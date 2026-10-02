"""Qwen Image 2.1, Flux.2 Dev and Ideogram 4 sample as their authors' reference pipelines do.

- Qwen Image 2.1: diffusers QwenImage21Pipeline with Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json
  (revision d26bb61231c3).
- Flux.2 Dev: black-forest-labs/flux2 (50fe516) src/flux2/text_encoder.py Mistral3SmallEmbedder.forward.
- Ideogram 4: ideogram-oss/ideogram4 (990fe1c) scheduler.py, sampler_configs.py and pipeline_ideogram4.py.
"""
import base64
import json
import math

import numpy as np
import pytest
import torch

import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.ops
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.supported_models
from comfy import sd1_clip
from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
from comfy.nodes.base_nodes import ConditioningZeroOut
from comfy.text_encoders import flux, llama
from comfy_extras.nodes.nodes_custom_sampler import (CFGOverride, DualModelGuider, KSamplerSelect, RandomNoise,
                                                     SamplerCustomAdvanced)
from comfy_extras.nodes.nodes_ideogram4 import Ideogram4Scheduler

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


# ideogram4 scheduler.LogitNormalSchedule, get_schedule_for_resolution and make_step_intervals, verbatim
def _official_ideogram4_schedule(width, height, mu, std):
    mean = mu + 0.5 * math.log((height * width) / (512 * 512))

    def schedule(t):
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = mean + std * z
        t_ = torch.special.expit(y)
        t_ = 1 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * 18.0))
        t_max = 1.0 / (1 + math.exp(0.5 * -15.0))
        return t_.clamp(t_min, t_max).to(torch.float32)

    return schedule


def _official_ideogram4_sigmas(width, height, steps, mu, std):
    # pipeline_ideogram4.__call__: for i in range(num_steps - 1, -1, -1): t = schedule(step_intervals[i + 1]),
    # s = schedule(step_intervals[i]); the reference time runs 0 (noise) to 1 (clean), sigma = 1 - t
    schedule = _official_ideogram4_schedule(width, height, mu, std)
    step_intervals = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)
    return torch.stack([1 - schedule(step_intervals[i:i + 1])[0] for i in range(steps, -1, -1)])


def test_ideogram4_scheduler_defaults_to_the_v4_quality_48_preset():
    # docs/inference.md: "V4_QUALITY_48 is the default"; sampler_configs.py: num_steps=48, mu=0.0, std=1.5
    defaults = {i.id: i.default for i in Ideogram4Scheduler.define_schema().inputs}
    assert (defaults["steps"], defaults["mu"], defaults["std"]) == (48, 0.0, 1.5)


@pytest.mark.parametrize("width,height", [(512, 512), (1024, 1024), (2048, 2048), (256, 416), (1600, 400), (64, 64)])
@pytest.mark.parametrize("steps,mu,std", [(48, 0.0, 1.5), (20, 0.0, 1.75), (12, 0.5, 1.75)])
def test_ideogram4_sigmas_are_the_official_loop(width, height, steps, mu, std):
    sigmas = Ideogram4Scheduler.execute(steps, width, height, mu, std).args[0]
    torch.testing.assert_close(sigmas, _official_ideogram4_sigmas(width, height, steps, mu, std), rtol=0, atol=0)
    assert sigmas[-1].item() == pytest.approx(1 / (1 + math.exp(7.5)), abs=1e-7)  # the loop stops at 1 - t_max, not 0


TINY_IDEOGRAM4 = {"num_attention_heads": 2, "attention_head_dim": 32, "intermediate_size": 128, "adaln_dim": 32,
                  "llm_features_dim": 24, "rope_theta": 5000000, "mrope_section": [4, 2, 2], "norm_eps": 1e-5}


@pytest.fixture
def tiny_ideogram4(monkeypatch):
    monkeypatch.setattr(comfy.supported_models.Ideogram4, "unet_extra_config", dict(TINY_IDEOGRAM4))
    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = Ideogram4Transformer2DModel(in_channels=128, num_layers=2, operations=comfy.ops.disable_weight_init, **TINY_IDEOGRAM4)
        state_dict = {k: (torch.randn_like(v) * 0.05).float() for k, v in model.state_dict().items()}
        models.append(comfy.sd.load_diffusion_model_state_dict(state_dict, model_options={"dtype": torch.float32}))
    return models


def test_ideogram4_cfg_override_last_steps_is_the_polish_schedule():
    # sampler_configs.py: guidance_schedule=(3.0,) * 3 + (7.0,) * 45 in loop-index order, index 0 being the last step
    sigmas = Ideogram4Scheduler.execute(48, 1024, 1024, 0.0, 1.5).args[0]

    class Guider:
        cfg = 7.0

    class Executor:
        class_obj = Guider()

        def __call__(self, *args, **kwargs):
            return self.class_obj.cfg

    class Model:
        model_options = {}

        def get_model_object(self, name):
            return comfy.model_sampling.ModelSamplingDiscreteFlow()

        def clone(self):
            return self

        def add_wrapper(self, wrapper_type, wrapper):
            self.wrapper = wrapper

    model = CFGOverride.execute(Model(), 3.0, 0.0, 1.0, last_steps=3).args[0]
    model_options = {"transformer_options": {"sample_sigmas": sigmas}}
    used = [model.wrapper(Executor(), torch.zeros(1), sigma.reshape(1), model_options, 0) for sigma in sigmas[:-1]]
    assert used == list(reversed((3.0,) * 3 + (7.0,) * 45))


@pytest.mark.parametrize("negative", ["zero_out", "unconnected"])
def test_ideogram4_samples_the_official_loop(tiny_ideogram4, negative):
    """The native graph (Ideogram4Scheduler, CFGOverride on the last 3 steps, DualModelGuider with the unconditional
    model, SamplerCustomAdvanced, euler) against pipeline_ideogram4's loop on the same models: z starts as the noise
    itself, each step uses the preset's guidance, the unconditional pass is image-only, and the result is z where the
    loop stops, undivided."""
    cond, uncond = tiny_ideogram4
    width = height = 64
    steps, mu, std = 48, 0.0, 1.5
    positive = [[torch.randn(1, 5, 24, generator=torch.Generator().manual_seed(1)), {}]]
    neg = ConditioningZeroOut().zero_out(positive)[0] if negative == "zero_out" else None
    sigmas = Ideogram4Scheduler.execute(steps, width, height, mu, std).args[0]
    model = CFGOverride.execute(cond, 3.0, 0.0, 1.0, last_steps=3).args[0]
    guider = DualModelGuider.execute(model, positive, 7.0, model_negative=uncond, negative=neg).args[0]
    latent = {"samples": torch.zeros(1, 128, height // 16, width // 16)}
    out = SamplerCustomAdvanced.execute(RandomNoise.execute(7).args[0], guider, KSamplerSelect.execute("euler").args[0], sigmas, latent).args[0]

    comfy.model_management.load_models_gpu([cond, uncond], force_full_load=True)
    device = cond.load_device
    z = comfy.sample.prepare_noise(latent["samples"], 7).to(device)
    context = positive[0][0].to(device)
    guidance = list(reversed((3.0,) * 3 + (7.0,) * 45))
    for j in range(steps):
        t = sigmas[j:j + 1].to(device)
        with torch.no_grad():
            pos_v = -cond.model.diffusion_model(z, t, context=context)
            neg_v = -uncond.model.diffusion_model(z, t)
        z = z + (guidance[j] * pos_v + (1 - guidance[j]) * neg_v) * (sigmas[j] - sigmas[j + 1]).item()
    torch.testing.assert_close(out["samples"], cond.model.process_latent_out(z.cpu()), rtol=1e-5, atol=1e-6)


# ideogram4 latent_norm.py LATENT_SHIFT and LATENT_SCALE as float32 bytes, and the Flux.2 VAE's bn statistics
# (black-forest-labs/FLUX.2-dev ae.safetensors and vae/diffusion_pytorch_model.safetensors)
IDEOGRAM4_LATENT_NORM_SHA256 = {
    "LATENT_SHIFT": "500785259e56d2e507f450e2a81b4e714b747104100a4ddac1c9681e20c62620",
    "LATENT_SCALE": "8aff847687d2712d2363ec1267bf7daf83efe14a9aa7f28547e72b8363e899cc",
    "FLUX2_VAE_BN_MEAN": "4c1979e1636ba68bd878a327b7e84e3c8d9ea2ca6422b7e1455e3eba02b977da",
    "FLUX2_VAE_BN_VAR": "a1cfa04c03e4b0c58b858d3e72b0d869a763fad85d37f8eeb130d0feb4d64f78",
}


def test_ideogram4_latent_norm_is_the_reference_and_the_flux2_vae():
    import hashlib
    from comfy.ldm.ideogram4 import latent_norm
    for name, digest in IDEOGRAM4_LATENT_NORM_SHA256.items():
        values = torch.tensor(getattr(latent_norm, name), dtype=torch.float32)
        assert values.shape == (128,)
        assert hashlib.sha256(values.numpy().tobytes()).hexdigest() == digest, name


def test_ideogram4_vae_decodes_the_reference_latent():
    """pipeline_ideogram4._decode: tokens (pi, pj, c) * LATENT_SCALE + LATENT_SHIFT, unpatched, into the decoder.
    ComfyUI decodes process_out(z) through the VAE, which first applies bn: z * sqrt(var + 1e-4) + mean, then
    unpatches (c pi pj) i j -> c (i pi) (j pj)."""
    from comfy.ldm.ideogram4 import latent_norm
    latent_format = comfy.supported_models.Ideogram4.latent_format()
    gh, gw = 3, 5
    z = torch.randn(2, 128, gh, gw, generator=torch.Generator().manual_seed(0))
    # the reference token layout, as Ideogram4Transformer2DModel packs it
    tokens = z.view(2, 32, 2, 2, gh, gw).permute(0, 4, 5, 2, 3, 1).reshape(2, gh * gw, 128)
    shift = torch.tensor(latent_norm.LATENT_SHIFT, dtype=torch.float32)
    scale = torch.tensor(latent_norm.LATENT_SCALE, dtype=torch.float32)
    reference = tokens * scale + shift
    reference = reference.view(2, gh, gw, 2, 2, 32).permute(0, 5, 1, 3, 2, 4).reshape(2, 32, gh * 2, gw * 2)

    out = latent_format.process_out(z)
    bn_std = torch.sqrt(torch.tensor(latent_norm.FLUX2_VAE_BN_VAR).view(1, -1, 1, 1) + 1e-4)
    bn_mean = torch.tensor(latent_norm.FLUX2_VAE_BN_MEAN).view(1, -1, 1, 1)
    vae_in = (out * bn_std + bn_mean).view(2, 32, 2, 2, gh, gw).permute(0, 1, 4, 2, 5, 3).reshape(2, 32, gh * 2, gw * 2)
    torch.testing.assert_close(vae_in, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(latent_format.process_in(out), z, rtol=1e-5, atol=1e-5)


def _tiny_qwen3vl_text(num_layers=36):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
    torch.manual_seed(0)
    config = Qwen3VLTextConfig(vocab_size=151936, hidden_size=256, intermediate_size=128, num_hidden_layers=num_layers,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=128, rms_norm_eps=1e-6,
                               rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                                "mrope_section": [24, 20, 20], "mrope_interleaved": True},
                               attn_implementation="sdpa")
    model = Qwen3VLTextModel(config).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    comfy_config = {"vocab_size": 151936, "hidden_size": 256, "intermediate_size": 128, "num_hidden_layers": num_layers,
                    "num_attention_heads": 2, "num_key_value_heads": 1}
    return model, comfy_config


def _official_ideogram4_text_features(language_model, token_ids):
    """pipeline_ideogram4._get_qwen3_vl_embeddings and _encode_text for one unpadded prompt: each tap is the
    output of decoder layer i for i in QWEN3_VL_ACTIVATION_LAYERS = (0, 3, ..., 33, 35)."""
    from transformers.masking_utils import create_causal_mask
    with torch.no_grad():
        inputs_embeds = language_model.embed_tokens(token_ids)
        pos_2d = torch.arange(token_ids.shape[1]).unsqueeze(0)
        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        causal_mask = create_causal_mask(config=language_model.config, inputs_embeds=inputs_embeds,
                                         attention_mask=torch.ones_like(token_ids), past_key_values=None,
                                         position_ids=position_ids_4d[0])
        position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids_4d[1:])
        captured = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids_4d[0],
                                          past_key_values=None, position_embeddings=position_embeddings)
            if layer_idx in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35):
                captured[layer_idx] = hidden_states
    stacked = torch.stack([captured[i] for i in sorted(captured)], dim=0).permute(1, 2, 3, 0)
    return stacked.reshape(token_ids.shape[0], token_ids.shape[1], -1)


@pytest.mark.parametrize("encoder", ["qwen3vl", "text_only"])
def test_ideogram4_text_features_are_the_reference_taps(encoder, monkeypatch):
    from comfy.text_encoders import ideogram4, qwen3vl
    reference, config = _tiny_qwen3vl_text()
    if encoder == "qwen3vl":
        monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, "qwen3vl_8b", dict(hidden_size=32, intermediate_size=32, depth=1, deepstack_visual_indexes=[0]))
        te = ideogram4.Ideogram4Qwen3VLTEModel(dtype=torch.float32, textmodel_json_config=config)
        tokenizer = ideogram4.Ideogram4Qwen3VLTokenizer()
    else:
        te = ideogram4.Ideogram4TEModel(dtype=torch.float32, model_options={"qwen3vl_8b_model_config": config})
        tokenizer = ideogram4.Ideogram4Tokenizer()
    te.qwen3vl_8b.transformer.model.load_state_dict(reference.state_dict(), strict=False)
    tokens = tokenizer.tokenize_with_weights('{"high_level_description": "a red fox in the snow"}')
    ids = torch.tensor([[t[0] for t in tokens["qwen3vl_8b"][0]]])
    out, _, _ = te.encode_token_weights(tokens)
    torch.testing.assert_close(out, _official_ideogram4_text_features(reference, ids), rtol=1e-5, atol=1e-5)
