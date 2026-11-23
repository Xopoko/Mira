from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import ImageFont

from context_operator.core.contracts import Intent


class CheckModelsTests(unittest.TestCase):
    def test_wrong_dictation_fails_the_command_even_with_the_right_target(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_models.py'
        with TemporaryDirectory() as folder:
            local = Path(folder)
            with (
                patch('context_operator.settings.LOCAL', local),
                patch('context_operator.settings.load', return_value={'model_mode': 'local'}),
                patch('context_operator.providers.models.Ranker') as ranker,
                patch('context_operator.providers.models.ContextModel') as model,
                patch('context_operator.foundations.perception.runtime.PerceptionClient') as perception,
                patch.object(ImageFont, 'truetype', return_value=ImageFont.load_default()),
                patch.object(sys, 'path', sys.path.copy()),
                redirect_stdout(io.StringIO()),
            ):
                ranker.return_value.rank.return_value = {'plus': 1.0, 'minus': 0.0, 'field': 0.0}
                perception.return_value.observe.return_value = {
                    'items': [{'source': 'ocr', 'label': label} for label in ('Plus', 'Minus')],
                    'timings': {},
                }
                model.return_value.interpret.side_effect = [
                    Intent('click', target_id='plus'),
                    Intent('click', target_id='minus'),
                    Intent('type_text', 'Different text', target_id='field'),
                ]
                with self.assertRaises(SystemExit) as exit_status:
                    runpy.run_path(str(script), run_name='__main__')
            self.assertEqual(exit_status.exception.code, 1)
            report = json.loads((local / 'model-check.json').read_text(encoding='utf-8'))
            self.assertEqual([case['passed'] for case in report['intent']], [True, True, False])
            self.assertFalse(report['passed'])
