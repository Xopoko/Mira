import json
import tempfile
import unittest
from pathlib import Path

from context_operator.personalization.history import preserve_attempt, latest_result
from context_operator.personalization.session import STARTER_PROTOCOL


class SetupHistoryTests(unittest.TestCase):
    def test_failed_attempt_survives_new_collection_and_remains_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'session.json'
            original = {'protocol': STARTER_PROTOCOL, 'display': [0, 0, 1920, 1080],
                        'train': [{'target': [50, 50]}], 'check': []}
            path.write_text(json.dumps(original), encoding='utf-8')
            report = {'passed': False, 'median_px': 250., 'p90_px': 900., 'coverage': .8}
            history = preserve_attempt(path, report)
            path.write_text('{"train":[],"check":[]}', encoding='utf-8')
            self.assertEqual(json.loads(history.read_text())['collection'], original)
            self.assertEqual(latest_result(folder, [0, 0, 1920, 1080]), report)
            self.assertIsNone(latest_result(folder, [0, 0, 1280, 720]))
            self.assertEqual(len(list(history.parent.glob('*.json'))), 1)

    def test_attempt_from_the_earlier_model_is_kept_but_not_shown_as_the_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'session.json'
            path.write_text(json.dumps({'protocol': 'personal-arkit-starter-v1', 'display': [0, 0, 1920, 1080],
                                        'train': [{'target': [50, 50]}], 'check': []}), encoding='utf-8')
            report = {'protocol': 'personal-arkit-starter-v1', 'usable': True,
                      'median_px': 250., 'p90_px': 500., 'coverage': 1.}
            self.assertTrue(preserve_attempt(path, report).is_file())
            self.assertIsNone(latest_result(folder, [0, 0, 1920, 1080]))
            (Path(folder)/'starter-last-result.json').write_text(json.dumps(
                {'version': 1, 'display': [0, 0, 1920, 1080], 'report': report}), encoding='utf-8')
            self.assertIsNone(latest_result(folder, [0, 0, 1920, 1080]))

    def test_corrupt_latest_result_does_not_block_profile_page(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'starter-last-result.json').write_text('{bad', encoding='utf-8')
            self.assertIsNone(latest_result(folder, [0, 0, 1920, 1080]))
