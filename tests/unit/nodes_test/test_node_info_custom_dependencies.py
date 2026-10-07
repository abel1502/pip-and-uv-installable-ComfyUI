import sys

from comfy.cmd.node_info import node_info
from comfy.nodes.package_typing import ExportedNodes
from comfy_compatibility.vanilla import prepare_vanilla_environment


def test_node_info_resolves_other_custom_nodes(monkeypatch):
    prepare_vanilla_environment()

    class Dependency:
        choices = ['registered custom node']

    class Consumer:
        RETURN_TYPES = ('STRING',)

        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'choice': (sys.modules['nodes'].NODE_CLASS_MAPPINGS['SchemaDependency'].choices,)}}

    exported = ExportedNodes(NODE_CLASS_MAPPINGS={'SchemaConsumer': Consumer, 'SchemaDependency': Dependency})
    monkeypatch.setattr('comfy.nodes_context.get_nodes', lambda: exported)
    info = node_info('SchemaConsumer', exported.NODE_CLASS_MAPPINGS, {})
    assert info['input']['required']['choice'][0] == Dependency.choices
