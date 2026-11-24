from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from PIL import ImageFont

from context_operator.core.contracts import Focus, Intent, Rect, Scene
from context_operator.providers.models import ContextModel


class CheckModelsTests(unittest.TestCase):
    def test_context_model_distinguishes_transport_http_and_malformed_responses(self):
        address = 'https://model.example:9443/v1'
        model = ContextModel(dict(model_mode='remote',model_url=address,model_name='fixture'))
        scene = Scene(1,10,Rect(0,0,800,600),())
        unreachable = 'Couldn’t reach the model at model.example:9443. You can still use short commands and gaze.'
        cases = (
            (dict(side_effect=URLError('offline')), unreachable),
            (dict(side_effect=TimeoutError()), unreachable),
            (dict(side_effect=OSError()), unreachable),
            (dict(side_effect=HTTPError(address,503,'Unavailable',{},None)),
             'The model server answered with HTTP 503.'),
            (dict(return_value=io.StringIO('not JSON')),
             'The model returned an invalid response. You can still use short commands and gaze. JSONDecodeError'),
        )
        for fixture, message in cases:
            with self.subTest(fixture=fixture), patch('context_operator.providers.models.urlopen', **fixture):
                with self.assertRaises(RuntimeError) as error:
                    model.interpret('fixture',scene,Focus())
                self.assertEqual(str(error.exception),message)
        model.settings['model_url'] = 'https://[::1]/v1'
        with patch('context_operator.providers.models.urlopen', side_effect=URLError('offline')):
            with self.assertRaises(RuntimeError) as error:
                model.interpret('fixture',scene,Focus())
        self.assertEqual(str(error.exception),unreachable.replace('model.example:9443','[::1]:443'))

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
