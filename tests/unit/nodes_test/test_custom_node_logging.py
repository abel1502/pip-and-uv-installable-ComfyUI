import logging
import sys

from comfy.nodes.vanilla_node_importing import _stdout_intercept


def test_custom_node_logger_can_capture_stdout_during_import(capsys):
    logger = logging.getLogger('custom_node_logger_test')
    handler = None
    original_level = logger.level
    logger.setLevel(logging.INFO)
    original_propagate = logger.propagate
    logger.propagate = False
    try:
        with _stdout_intercept(logger.name):
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.warning('during import')
        logger.warning('after import')
        captured = capsys.readouterr().out
        assert 'during import' in captured
        assert 'after import' in captured
    finally:
        if handler is not None:
            logger.removeHandler(handler)
        logger.propagate = original_propagate
        logger.setLevel(original_level)
