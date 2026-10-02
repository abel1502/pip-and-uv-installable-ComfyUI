"""Qwen Image 2.1, Flux.2 Dev and Ideogram 4 sample as their authors' reference pipelines do.

- Qwen Image 2.1: diffusers QwenImage21Pipeline with Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json
  (revision d26bb61231c3).
- Flux.2 Dev: black-forest-labs/flux2 (50fe516) src/flux2/text_encoder.py Mistral3SmallEmbedder.forward.
- Ideogram 4: ideogram-oss/ideogram4 (990fe1c) scheduler.py, sampler_configs.py and pipeline_ideogram4.py.
"""
import base64
import json

import numpy as np
import pytest
import torch

import comfy.model_base
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.supported_models
from comfy import sd1_clip
from comfy.text_encoders import flux, llama

# Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json at d26bb61231c3, verbatim
QWEN_IMAGE21_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": 0.9, "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


def _official_qwen_image21_sigmas(width, height, steps):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(QWEN_IMAGE21_SCHEDULER_CONFIG)
    # pipeline_qwenimage21: sigmas = np.linspace(1.0, 1 / N, N); mu = calculate_shift(latents.shape[1], ...) with one
    # packed latent per 16x16 pixels
    c = QWEN_IMAGE21_SCHEDULER_CONFIG
    m = (c["max_shift"] - c["base_shift"]) / (c["max_image_seq_len"] - c["base_image_seq_len"])
    mu = (height // 16) * (width // 16) * m + (c["base_shift"] - m * c["base_image_seq_len"])
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    return scheduler.sigmas


class _Patcher:
    load_device = torch.device("cpu")
    model_options = {}

    def __init__(self, model_sampling):
        self.model_sampling = model_sampling

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.model_sampling


@pytest.fixture(scope="module")
def qwen_image21_model_sampling():
    config = comfy.supported_models.QwenImage21({"image_model": "qwen_image21", "num_layers": 1, "attention_head_dim": 128,
                                                 "num_attention_heads": 1, "context_in_dim": 16})
    return comfy.model_base.QwenImage21(config, device="meta").model_sampling


@pytest.mark.parametrize("steps", [40, 25])
@pytest.mark.parametrize("width,height", [(1024, 1024), (512, 512), (2048, 2048), (1536, 1024), (2752, 1536), (64, 64)])
def test_qwen_image21_ksampler_simple_is_the_official_schedule(qwen_image21_model_sampling, width, height, steps):
    expected = _official_qwen_image21_sigmas(width, height, steps)
    assert expected[-2].item() == pytest.approx(0.02)  # shift_terminal
    sampler = comfy.samplers.KSampler(_Patcher(qwen_image21_model_sampling), steps, "cpu", sampler="euler", scheduler="simple",
                                      latent_shape=(1, 64, height // 16, width // 16))
    torch.testing.assert_close(sampler.sigmas, expected, rtol=0, atol=0)


def test_qwen_image21_sample_uses_the_latents_resolution(qwen_image21_model_sampling, monkeypatch):
    seen = []

    def sample(self, noise, *args, **kwargs):
        seen.append(self.sigmas)
        return noise

    monkeypatch.setattr(comfy.samplers.KSampler, "sample", sample)
    latent = torch.zeros(1, 64, 96, 64)  # 1024x1536
    comfy.sample.sample(_Patcher(qwen_image21_model_sampling), latent, 40, 1.0, "euler", "simple", [], [], latent)
    torch.testing.assert_close(seen[0], _official_qwen_image21_sigmas(1024, 1536, 40), rtol=0, atol=0)


def test_qwen_image21_denoise_takes_the_tail_of_the_official_schedule(qwen_image21_model_sampling):
    sampler = comfy.samplers.KSampler(_Patcher(qwen_image21_model_sampling), 20, "cpu", sampler="euler", scheduler="simple",
                                      denoise=0.5, latent_shape=(1, 64, 64, 64))
    torch.testing.assert_close(sampler.sigmas, _official_qwen_image21_sigmas(1024, 1024, 40)[-21:], rtol=0, atol=0)


def test_qwen_image21_model_sampling_node_overrides_the_official_schedule():
    # ModelSamplingFlux / ModelSamplingAuraFlow patch the model with their own fixed shift, which the scheduler keeps using
    model_sampling = comfy.model_sampling.ModelSamplingFlux()
    model_sampling.set_parameters(shift=1.15)
    sampler = comfy.samplers.KSampler(_Patcher(model_sampling), 40, "cpu", sampler="euler", scheduler="simple",
                                      latent_shape=(1, 64, 64, 64))
    torch.testing.assert_close(sampler.sigmas, comfy.samplers.simple_scheduler(model_sampling, 40), rtol=0, atol=0)


def _tiny_tekken():
    special = {0: "<unk>", 1: "<s>", 2: "</s>", 3: "[INST]", 4: "[/INST]", 11: "<pad>", 17: "[SYSTEM_PROMPT]", 18: "[/SYSTEM_PROMPT]"}
    return json.dumps({
        "config": {"default_num_special_tokens": 1000, "default_vocab_size": 1256},
        "vocab": [{"rank": i, "token_bytes": base64.b64encode(bytes([i])).decode()} for i in range(256)],
        "special_tokens": [{"rank": rank, "token_str": token} for rank, token in special.items()],
    }).encode()


def test_flux2_prompt_is_left_padded_to_512_with_pad_tokens():
    # text_encoder.py: processor.apply_chat_template(..., padding="max_length", truncation=True, max_length=512); the
    # Mistral-Small-3.1 processor's LlamaTokenizerFast pads on the left with <pad> (id 11)
    tokenizer = flux.Flux2Tokenizer(tokenizer_data={"tekken_model": _tiny_tekken()})
    unpadded = [t for t, _ in tokenizer.tokenize_with_weights("a red fox", tokenizer_options={"mistral3_24b_min_length": 1})["mistral3_24b"][0]]
    padded = [t for t, _ in tokenizer.tokenize_with_weights("a red fox")["mistral3_24b"][0]]
    assert unpadded[0] == 1 and 11 not in unpadded
    assert len(padded) == 512
    assert padded == [11] * (512 - len(unpadded)) + unpadded


def test_flux2_scheduler_defaults_to_the_reference_step_count():
    # util.py FLUX2_MODEL_INFO["flux.2-dev"]: "defaults": {"guidance": 4.0, "num_steps": 50}
    from comfy_extras.nodes.nodes_flux import Flux2Scheduler
    assert {i.id: i.default for i in Flux2Scheduler.define_schema().inputs}["steps"] == 50


def _tiny_mistral(num_layers=3):
    from transformers import MistralConfig, MistralModel
    torch.manual_seed(0)
    config = MistralConfig(vocab_size=64, hidden_size=512, intermediate_size=256, num_hidden_layers=num_layers, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=128, rope_theta=1000000000.0, rms_norm_eps=1e-5,
                           attn_implementation="sdpa")
    model = MistralModel(config).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    comfy_config = {"vocab_size": 64, "hidden_size": 512, "intermediate_size": 256, "num_hidden_layers": num_layers,
                    "num_attention_heads": 4, "num_key_value_heads": 2}
    return model, comfy_config


def test_flux2_text_encoder_pad_rows_match_the_reference_encoder():
    """The official encoder runs the 512 left-padded tokens with their attention mask: pad keys are masked, and a pad
    query, which has no key it may attend to, gets a zero attention output from SDPA, so its hidden state is the MLP
    stack applied to the pad embedding. The DiT attends to all 512 rows."""
    reference, config = _tiny_mistral()
    special_tokens = flux.Mistral3_24BModel(device="meta").special_tokens
    assert special_tokens == {"start": 1, "pad": 11}
    encoder = sd1_clip.SDClipModel(layer=[1, 2], textmodel_json_config=config, dtype=torch.float32, model_class=llama.Mistral3Small24B,
                                   special_tokens=special_tokens, layer_norm_hidden_state=False, enable_attention_masks=True,
                                   return_attention_masks=True)
    encoder.transformer.model.load_state_dict(reference.state_dict(), strict=False)
    ids = [11] * 6 + [1, 5, 7, 9, 3]
    out, _, extra = encoder.forward([ids])
    with torch.no_grad():
        expected = reference(input_ids=torch.tensor([ids]), attention_mask=(torch.tensor([ids]) != 11).long(), output_hidden_states=True).hidden_states
    assert extra["attention_mask"].tolist() == [[0] * 6 + [1] * 5]
    for i, layer in enumerate((1, 2)):
        torch.testing.assert_close(out[:, i], expected[layer], rtol=1e-5, atol=1e-5)


def test_flux2_text_encoder_without_padding_is_unchanged():
    reference, config = _tiny_mistral()
    encoder = sd1_clip.SDClipModel(layer=[1, 2], textmodel_json_config=config, dtype=torch.float32, model_class=llama.Mistral3Small24B,
                                   special_tokens={"start": 1, "pad": 11}, layer_norm_hidden_state=False, enable_attention_masks=True,
                                   return_attention_masks=True)
    encoder.transformer.model.load_state_dict(reference.state_dict(), strict=False)
    ids = [1, 5, 7, 9, 3]
    out, _, _ = encoder.forward([ids])
    with torch.no_grad():
        expected = reference(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    for i, layer in enumerate((1, 2)):
        torch.testing.assert_close(out[:, i], expected[layer], rtol=1e-5, atol=1e-5)
