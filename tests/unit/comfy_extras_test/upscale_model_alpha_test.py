import torch

from types import SimpleNamespace

from comfy import model_management
from comfy_extras.nodes import nodes_upscale_model
from comfy_extras.nodes.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelManageable


class StubUpscaleModel(torch.nn.Module):
    """Stands in for a spandrel ImageModelDescriptor that only accepts RGB."""

    scale = 2

    class patcher:
        load_device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.seen_channels = None

    def forward(self, image):
        self.seen_channels = image.shape[1]
        assert image.shape[1] == 3, "model was handed a non-RGB tensor"
        return torch.nn.functional.interpolate(image, scale_factor=self.scale, mode="nearest")


def rgba_image():
    """16x16 RGBA where the top half is transparent and the bottom half opaque."""
    image = torch.zeros(1, 16, 16, 4)
    image[..., :3] = 0.5
    image[0, 8:, :, 3] = 1.0
    return image


def upscale(image, monkeypatch):
    monkeypatch.setattr(nodes_upscale_model, "load_models_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(model_management, "unet_offload_device", lambda: torch.device("cpu"))
    model = StubUpscaleModel()
    descriptor = SimpleNamespace(model=model, device=torch.device("cpu"), input_channels=3, output_channels=3, scale=2)
    return model, ImageUpscaleWithModel.execute(UpscaleModelManageable(descriptor, "test"), image)


def test_rgba_input_does_not_crash_and_alpha_is_preserved(monkeypatch):
    model, out = upscale(rgba_image(), monkeypatch)
    image = out.result[0]

    assert model.seen_channels == 3
    assert image.shape == (1, 32, 32, 4)
    assert image[0, :14, :, 3].max() < 0.1
    assert image[0, 18:, :, 3].min() > 0.9


def test_rgb_input_is_untouched(monkeypatch):
    model, out = upscale(torch.zeros(1, 16, 16, 3), monkeypatch)
    image = out.result[0]

    assert model.seen_channels == 3
    assert image.shape == (1, 32, 32, 3)
