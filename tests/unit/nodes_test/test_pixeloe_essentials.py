import sys
from types import ModuleType

import pytest

from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1


@pytest.mark.parametrize('legacy_layout', [False, True])
def test_essentials_pixelize_executes_with_both_pixeloe_layouts(tmp_path, monkeypatch, legacy_layout):
    package = ModuleType('pixeloe')
    package.__path__ = []
    legacy = ModuleType('pixeloe.legacy')
    legacy.__path__ = []
    implementation = ModuleType('pixeloe.legacy.pixelize')
    implementation.pixelize = lambda image: image + 1
    monkeypatch.setitem(sys.modules, 'pixeloe', package)
    monkeypatch.setitem(sys.modules, 'pixeloe.legacy', legacy)
    monkeypatch.setitem(sys.modules, 'pixeloe.legacy.pixelize', implementation)
    monkeypatch.setitem(sys.modules, 'pixeloe.pixelize', None if legacy_layout else implementation)
    node_package = tmp_path / 'ComfyUI_essentials'
    node_package.mkdir()
    (node_package / '__init__.py').write_text(
        'class PixelOEPixelize:\n'
        '    def execute(self, image):\n'
        '        from pixeloe.pixelize import pixelize\n'
        '        return pixelize(image)\n'
        'NODE_CLASS_MAPPINGS = {"PixelOEPixelize+": PixelOEPixelize}\n'
    )
    monkeypatch.setitem(sys.modules, 'ComfyUI_essentials', None)
    exported = _vanilla_load_custom_nodes_1(str(node_package))
    assert exported.NODE_CLASS_MAPPINGS['PixelOEPixelize+']().execute(41) == 42
