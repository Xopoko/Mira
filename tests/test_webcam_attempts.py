import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from context_operator.core.contracts import Rect
from context_operator.core.webcam import WebcamObservation
from context_operator.personalization.webcam import (
    CHECK_TARGETS,
    TRAIN_TARGETS,
    WebcamSession,
)
from context_operator.personalization.webcam_attempts import WebcamAttemptStore


BOUNDS = Rect(-1920, 0, 3840, 2160)


def feature_for(target):
    x, y = target
    return (x - .5, y - .5, 0., 0., 0., .5, .5, .25)


def completed_session(*, failed_check=False):
    session = WebcamSession(BOUNDS, 'camera-1')
    seq = 0
    start = 0.0
    for target in TRAIN_TARGETS:
        feature = feature_for(target)
        for offset in range(12):
            stamp = start + offset * .08
            session.feed(WebcamObservation(seq, stamp, feature), stamp)
            seq += 1
        if not session.confirm(stamp + .01):
            raise AssertionError(session.hold_reason)
        start = stamp + .30
    for index, target in enumerate(CHECK_TARGETS):
        observed = TRAIN_TARGETS[-1] if failed_check and index == 0 else target
        feature = feature_for(observed)
        for offset in range(12):
            stamp = start + offset * .08
            session.feed(WebcamObservation(seq, stamp, feature), stamp)
            seq += 1
        if not session.confirm(stamp + .01):
            raise AssertionError(session.hold_reason)
        start = stamp + .30
    if not session.complete:
        raise AssertionError('Synthetic webcam session did not complete')
    return session


class WebcamAttemptStoreTests(unittest.TestCase):
    def test_failed_complete_attempt_is_saved_without_adopting_profile(self):
        session = completed_session(failed_check=True)
        self.assertFalse(session.report.usable)
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamAttemptStore(folder)
            archive_path = store.save(session)
            latest = store.load_latest()
            self.assertTrue(archive_path.is_file())
            self.assertFalse(latest['usable'])
            self.assertIs(latest['passed'], False)
            self.assertEqual(latest['device_id'], 'camera-1')
            self.assertEqual(latest['engine_version'], session.candidate.engine_version)
            self.assertEqual(latest['bounds'], list(session.candidate.bounds))
            self.assertEqual(len(latest['train_rows']), 9)
            self.assertEqual(latest['train_rows'], latest['model']['training_rows'])
            self.assertIn('mapping', latest['model'])
            self.assertEqual(len(latest['model']['training_rows'][0]['samples']), 12)
            self.assertEqual(len(latest['check_rows']), 5)
            self.assertEqual(latest['report']['median_px'], session.report.median_px)
            self.assertEqual(len(latest['check_rows'][0]['features']), 12)
            self.assertFalse((Path(folder) / 'webcam-profile.json').exists())
            self.assertFalse(any(key in json.dumps(latest).lower()
                                 for key in ('image', 'pixels', 'frame_bytes')))

    def test_rejects_incomplete_attempt_before_writing(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamAttemptStore(folder)
            with self.assertRaisesRegex(ValueError, 'complete'):
                store.save(session)
            self.assertFalse(store.root.exists())

    def test_rejects_nonfinite_numeric_result_before_writing(self):
        session = completed_session()
        session.report = replace(session.report, median_px=math.nan)
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamAttemptStore(folder)
            with self.assertRaises(ValueError):
                store.save(session)
            self.assertFalse(store.root.exists())

    def test_reader_rejects_tampered_or_nonfinite_record(self):
        session = completed_session()
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamAttemptStore(folder)
            store.save(session)
            envelope = json.loads(store.latest_path.read_text(encoding='utf-8'))
            envelope['report']['median_px'] += 1
            store.latest_path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                store.load_latest()

            envelope.pop('sha256')
            envelope['report']['median_px'] = 'nan'
            body = {key: value for key, value in envelope.items() if key != 'sha256'}
            envelope['sha256'] = hashlib.sha256(store._canonical(body)).hexdigest()
            store.latest_path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'constant|numeric|report'):
                store.load_latest()

    def test_archive_retains_only_ten_latest_completed_attempts(self):
        session = completed_session()
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamAttemptStore(folder)
            saved = [store.save(session) for _ in range(12)]
            archives = list(store.archive.glob('webcam-attempt-*.json'))
            self.assertEqual(len(archives), store.archive_limit)
            self.assertTrue(saved[-1].exists())
            latest = store.load_latest()
            self.assertEqual(latest['attempt_id'], saved[-1].stem.split('-')[-1])
            self.assertTrue(all(path.stat().st_size <= store.max_attempt_bytes
                                for path in archives))


if __name__ == '__main__':
    unittest.main()
