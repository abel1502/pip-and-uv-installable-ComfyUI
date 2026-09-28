import logging
import sys
from textwrap import dedent

import pytest

from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('ignored', [False, True])
def test_v3_custom_nodes_do_not_report_missing_legacy_mapping(tmp_path, caplog, asynchronous, ignored):
    module = tmp_path / 'v3_registration_test.py'
    module.write_text(dedent('''\
        from comfy_api.latest import ComfyExtension, io

        class Example(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id='V3RegistrationTest', display_name='V3 Registration Test')

            @classmethod
            def execute(cls):
                return io.NodeOutput()

        class Extension(ComfyExtension):
            async def get_node_list(self):
                return [Example]
    ''') + ('async ' if asynchronous else '') + 'def comfy_entrypoint():\n    return Extension()\n')
    try:
        with caplog.at_level(logging.ERROR, logger='comfy.nodes.vanilla_node_importing'):
            exported = _vanilla_load_custom_nodes_1(str(module), ignore={'V3RegistrationTest'} if ignored else set())
        assert set(exported.NODE_CLASS_MAPPINGS) == (set() if ignored else {'V3RegistrationTest'})
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    finally:
        sys.modules.pop(module.stem, None)
