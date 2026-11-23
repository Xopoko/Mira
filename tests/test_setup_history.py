import json
import tempfile
import unittest
from pathlib import Path

from context_operator.personalization.history import preserve_attempt, latest_result


class SetupHistoryTests(unittest.TestCase):
    def test_failed_attempt_survives_new_collection_and_remains_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'session.json'
            original = {'protocol': 'fixture', 'display': [0, 0, 1920, 1080],
                        'train': [{'target': [50, 50]}], 'check': []}
            path.write_text(json.dumps(original), encoding='utf-8')
            report = {'passed': False, 'median_px': 250., 'p90_px': 900., 'coverage': .8}
            history = preserve_attempt(path, report)
            path.write_text('{"train":[],"check":[]}', encoding='utf-8')
            self.assertEqual(json.loads(history.read_text())['collection'], original)
            self.assertEqual(latest_result(folder, [0, 0, 1920, 1080]), report)
            self.assertIsNone(latest_result(folder, [0, 0, 1280, 720]))
            self.assertEqual(len(list(history.parent.glob('*.json'))), 1)

    def test_corrupt_latest_result_does_not_block_profile_page(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'personal-last-result.json').write_text('{bad', encoding='utf-8')
            self.assertIsNone(latest_result(folder, [0, 0, 1920, 1080]))

    def test_latest_starter_attempt_is_visible_without_replacing_older_history(self):
        with tempfile.TemporaryDirectory() as folder:
            bounds = [0, 0, 1920, 1080]
            old = {'passed': False, 'median_px': 250., 'p90_px': 900., 'coverage': .8}
            new = {'quality': 'starter', 'usable': False, 'median_px': 200., 'p90_px': 800., 'coverage': .8}
            for name, date, report in [('personal', '2026-09-23T10:00:00+00:00', old),
                                       ('starter', '2026-09-24T10:00:00+00:00', new)]:
                (Path(folder)/f'{name}-last-result.json').write_text(json.dumps(
                    {'version': 1, 'display': bounds, 'saved_at': date, 'report': report}), encoding='utf-8')
            self.assertEqual(latest_result(folder, bounds), new)
            (Path(folder)/'starter-last-result.json').write_text('{bad', encoding='utf-8')
            self.assertEqual(latest_result(folder, bounds), old)
