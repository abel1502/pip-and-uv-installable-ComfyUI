"""Qwen Image 2.1, Flux.2 Dev and Ideogram 4 sample as their authors' reference pipelines do.

- Qwen Image 2.1: diffusers QwenImage21Pipeline with Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json
  (revision d26bb61231c3).
- Flux.2 Dev: black-forest-labs/flux2 (50fe516) src/flux2/text_encoder.py Mistral3SmallEmbedder.forward.
- Ideogram 4: ideogram-oss/ideogram4 (990fe1c) scheduler.py, sampler_configs.py and pipeline_ideogram4.py.
"""
import numpy as np
import pytest
import torch

import comfy.model_base
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.supported_models

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
