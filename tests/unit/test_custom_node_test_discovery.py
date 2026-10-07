import importlib
from types import SimpleNamespace

import pytest

from comfy.nodes.package_typing import ExportedNodes


def _preprocessor_module(monkeypatch, tmp_path, exported):
    def wrong_package():
        raise AssertionError('Advanced ControlNet is not the auxiliary preprocessor package')

    entries = [
        SimpleNamespace(name='comfyui-advanced-controlnet', load=wrong_package),
        SimpleNamespace(name='comfyui-controlnet-aux', load=lambda: object()),
    ]
    monkeypatch.setattr('importlib.metadata.entry_points', lambda: SimpleNamespace(select=lambda **kwargs: entries))
    monkeypatch.setattr('comfy_compatibility.vanilla.prepare_vanilla_environment', lambda: None)
    monkeypatch.setattr('comfy.nodes.package._extract_vanilla_custom_node_roots', lambda module: [str(tmp_path)])
    monkeypatch.setattr('comfy.nodes.vanilla_node_importing._vanilla_load_custom_nodes_1', lambda path: exported)
    module = importlib.import_module('tests.custom_nodes.test_controlnet_aux_preprocessors')
    monkeypatch.setattr(module, '_ALL_PREPROCESSORS', [])
    return module


def test_preprocessor_discovery_selects_aux_instead_of_advanced_controlnet(monkeypatch, tmp_path):
    class Selector:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'preprocessor': (['none', 'Canny', 'Depth'],)}}

    exported = ExportedNodes(NODE_CLASS_MAPPINGS={'ControlNetPreprocessorSelector': Selector})
    module = _preprocessor_module(monkeypatch, tmp_path, exported)
    assert module._discover_preprocessors() == ['Canny', 'Depth']


def test_preprocessor_discovery_fails_when_selector_is_missing(monkeypatch, tmp_path):
    module = _preprocessor_module(monkeypatch, tmp_path, ExportedNodes())
    with pytest.raises(RuntimeError, match='ControlNetPreprocessorSelector'):
        module._discover_preprocessors()


def test_custom_node_installation_does_not_accept_partial_success(monkeypatch, tmp_path):
    from tests.custom_nodes import conftest

    monkeypatch.setattr(conftest, 'CUSTOM_NODE_REGISTRY', [SimpleNamespace(node_id='broken')])

    def fail_install(spec, base):
        raise RuntimeError('fixture installation failed')

    monkeypatch.setattr(conftest, 'install_custom_node_from_spec', fail_install)
    with pytest.raises(RuntimeError, match='broken'):
        conftest.install_all_nodes(tmp_path)
