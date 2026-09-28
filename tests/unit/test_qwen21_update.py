from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfy import model_management
from comfy.sd import VAE
from comfy.taesd.taesd import TAESD
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
from comfy_extras.nodes import nodes_upscale_model


def test_qwen_vision_rgba_matches_rgb_patches():
    rgb = torch.rand(1, 224, 224, 3)
    rgba = torch.cat((rgb, torch.rand(1, 224, 224, 1)), dim=-1)
    options = dict(patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    patches, grid = process_qwen2vl_images(rgb, **options)
    rgba_patches, rgba_grid = process_qwen2vl_images(rgba, **options)
    torch.testing.assert_close(rgba_grid, grid)
    torch.testing.assert_close(rgba_patches, patches)


@pytest.fixture
def qwen_tiny_vae():
    model = TAESD(latent_channels=64)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.vae_scale.data.fill_(1)
    return model


def test_qwen_tiny_vae_rgba_roundtrip_shape(qwen_tiny_vae):
    latent = qwen_tiny_vae.encode(torch.zeros(1, 4, 32, 48))
    assert latent.shape == (1, 64, 2, 3)
    decoded = qwen_tiny_vae.decode(latent)
    assert decoded.shape == (1, 4, 32, 48)
    assert torch.isfinite(decoded).all()


def test_qwen_tiny_vae_loader_configuration(qwen_tiny_vae):
    model = VAE(sd=qwen_tiny_vae.state_dict(), device=torch.device('cpu'))
    assert model.latent_channels == 64
    assert model.downscale_ratio == model.upscale_ratio == 16
    assert model.output_channels == 4
    assert model.pad_channel_value == 1.0


def test_upscaler_preserves_alpha(monkeypatch):
    class RGBUpscaler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, image):
            assert image.shape[1] == 3
            return F.interpolate(image, scale_factor=2, mode='nearest') * self.weight

    monkeypatch.setattr(model_management, 'get_torch_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(model_management, 'unet_offload_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(nodes_upscale_model, 'load_models_gpu', lambda *a, **kw: None)
    descriptor = SimpleNamespace(model=RGBUpscaler(), device=torch.device('cpu'),
                                 input_channels=3, output_channels=3, scale=2)
    model = nodes_upscale_model.UpscaleModelManageable(descriptor, 'test')
    image = torch.full((1, 64, 64, 4), 0.3)
    result = nodes_upscale_model.ImageUpscaleWithModel.execute(model, image)[0]
    assert result.shape == (1, 128, 128, 4)
    torch.testing.assert_close(result, torch.full_like(result, 0.3))
