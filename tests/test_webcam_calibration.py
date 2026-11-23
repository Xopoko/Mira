import json
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from context_operator.core.contracts import Rect
from context_operator.core.webcam import WebcamObservation
from context_operator.personalization.webcam import (
    CHECK_TARGETS, TRAIN_TARGETS, WebcamProfile, WebcamProfileStore, WebcamSession,
)


BOUNDS = Rect(-1920, 0, 3840, 2160)


def feature_for(target):
    x, y = target
    return (x - .5, y - .5, 0., 0., 0., .5, .5, .25)


def fitted_profile(device_id='camera-1'):
    features = [feature_for(target) for target in TRAIN_TARGETS]
    profile = WebcamProfile.fit(features, TRAIN_TARGETS, BOUNDS, device_id)
    groups = [[feature_for(target)] * 12 for target in CHECK_TARGETS]
    report = profile.validate(groups, CHECK_TARGETS)
    return replace(profile, validation_p90_px=report.p90_px), report


class WebcamProfileTests(unittest.TestCase):
    def test_low_capacity_fit_and_joint_support(self):
        profile, report = fitted_profile()
        point = profile.predict(feature_for((.23, .23)))
        self.assertAlmostEqual(point[0], BOUNDS.x + BOUNDS.w * .23, places=7)
        self.assertAlmostEqual(point[1], BOUNDS.y + BOUNDS.h * .23, places=7)
        self.assertTrue(report.usable)
        self.assertFalse(report.passed)
        self.assertEqual(report.coverage, (1.,) * 5)
        moved_camera = list(feature_for((.23, .23)))
        moved_camera[5] += .30
        self.assertFalse(profile.supports(moved_camera))
        with self.assertRaises(ValueError):
            profile.predict(moved_camera)

    def test_targets_are_fixed_and_disjoint(self):
        self.assertFalse(set(TRAIN_TARGETS) & set(CHECK_TARGETS))
        with self.assertRaises(ValueError):
            WebcamProfile.fit([feature_for(t) for t in TRAIN_TARGETS],
                              [*TRAIN_TARGETS[:-1], CHECK_TARGETS[0]], BOUNDS, 'camera-1')

    def test_coverage_counts_missing_observations(self):
        profile, _ = fitted_profile()
        groups = [[feature_for(target)] * 10 for target in CHECK_TARGETS]
        report = profile.validate(groups, CHECK_TARGETS, invalid_counts=[0, 0, 0, 0, 3])
        self.assertLess(report.coverage[-1], .80)
        self.assertFalse(report.usable)

    def test_store_rechecks_receipt_and_identity_and_uses_backup(self):
        profile, report = fitted_profile()
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamProfileStore(folder)
            store.adopt(profile, report)
            first = store.load(BOUNDS, 'camera-1')
            self.assertEqual(first.validation_p90_px, report.p90_px)
            other = replace(profile, coefficients=((0., 0.),) * 3)
            with self.assertRaises(ValueError):
                store.adopt(other, report)
            # Preserve the valid incumbent as a verified backup on replacement.
            store.adopt(profile, report)
            store.path.write_text('{broken', encoding='utf-8')
            recovered = store.load(BOUNDS, 'camera-1')
            self.assertEqual(recovered.device_id, 'camera-1')
            with self.assertRaises(ValueError):
                store.load(BOUNDS, 'camera-2')

    def test_store_rejects_validation_scalar_that_disagrees_with_check_data(self):
        profile, report = fitted_profile()
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamProfileStore(folder)
            store.adopt(profile, report)
            payload = json.loads(store.path.read_text(encoding='utf-8'))
            payload['model']['validation_p90_px'] = 123.
            payload.pop('sha256')
            payload['sha256'] = __import__('hashlib').sha256(store._canonical(payload)).hexdigest()
            store.path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                store.load(BOUNDS, 'camera-1')

    def test_missing_profile_message_is_actionable(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Set up the webcam first'):
                WebcamProfileStore(folder).load(BOUNDS, 'camera-1')

    def test_adopt_does_not_trust_mutated_usable_flag(self):
        profile, report = fitted_profile()
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamProfileStore(folder)
            with self.assertRaises(ValueError):
                store.adopt(profile, replace(report, usable=False))
            self.assertFalse(store.path.exists())


class WebcamSessionTests(unittest.TestCase):
    def _fixate(self, session, target, seq, start):
        feature = feature_for(target)
        for offset in range(12):
            stamp = start + offset * .07
            session.feed(WebcamObservation(seq + offset, stamp, feature), stamp)
        self.assertTrue(session.ready, session.hold_reason)
        self.assertTrue(session.confirm(start + 12 * .07))
        self.assertFalse(session.confirm(start + 12 * .07))
        return seq + 12, start + 1.

    def test_all_targets_need_unique_fresh_stable_fixations(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        first = WebcamObservation(1, 0., feature_for(TRAIN_TARGETS[0]))
        session.feed(first, 0.)
        session.feed(first, 1.)
        self.assertFalse(session.ready)
        self.assertEqual(session.index, 0)
        seq, now = 10, 1.
        for target in TRAIN_TARGETS:
            seq, now = self._fixate(session, target, seq, now)
        self.assertEqual(session.phase, 'check')
        for target in CHECK_TARGETS:
            seq, now = self._fixate(session, target, seq, now)
        self.assertTrue(session.complete)
        self.assertTrue(session.report.usable)
        self.assertFalse(session.report.passed)
        self.assertEqual(session.candidate.device_id, 'camera-1')

    def test_stale_and_blink_observation_clear_streak(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        for i in range(9):
            stamp = i * .08
            session.feed(WebcamObservation(i, stamp, feature_for(TRAIN_TARGETS[0])), stamp)
        self.assertFalse(session.ready)
        session.feed(WebcamObservation(10, .8, feature_for(TRAIN_TARGETS[0]), 'blink'), .8)
        self.assertFalse(session.frames)
        self.assertFalse(session.confirm(.8))
        self.assertIn('blink', session.hold_reason)

    def test_fast_ui_duplicate_polling_preserves_fixation(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        for i in range(10):
            stamp = i * .08
            obs = WebcamObservation(i, stamp, feature)
            session.feed(obs, stamp)
            session.feed(obs, stamp + .035)
        self.assertTrue(session.ready)
        self.assertTrue(session.confirm(.755))

    def test_six_fps_with_inference_latency_and_stale_ui_polls_can_finish(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        became_ready = False
        for i in range(11):
            stamp = i / 6
            observation = WebcamObservation(i, stamp, feature)
            session.feed(observation, stamp + .06)
            became_ready |= session.ready
            if i == 9:
                self.assertTrue(session.ready)
            session.feed(observation, stamp + .21)
            self.assertFalse(session.ready)
            self.assertFalse(session.confirm(stamp + .21))
            self.assertEqual(len(session.frames), i+1)
        self.assertTrue(became_ready)
        # Historical rows never bridge an actual capture outage.
        session.feed(WebcamObservation(11, 3., feature), 3.06)
        self.assertFalse(session.ready)
        self.assertEqual(len(session.frames), 1)

    def test_six_fps_stable_fixation_keeps_ten_distinct_frames(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        for i in range(10):
            stamp = i / 6
            session.feed(WebcamObservation(i, stamp, feature), stamp)
        self.assertTrue(session.ready, session.hold_reason)
        self.assertEqual(session.sample_count, 10)
        self.assertGreaterEqual(session.duration, .65)
        self.assertEqual(session.progress, 1.0)
        self.assertTrue(session.confirm(10 / 6))
        np.testing.assert_allclose(session.train[0]['features'], feature, atol=1e-12)

    def test_terminal_spike_cannot_auto_confirm_until_new_coherent_frame(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        for i in range(10):
            stamp = i * .08
            session.feed(WebcamObservation(i, stamp, feature), stamp)
        spike = list(feature)
        spike[0] += .10
        stamp = 10 * .08
        session.feed(WebcamObservation(10, stamp, tuple(spike)), stamp)
        self.assertFalse(session.ready)
        self.assertEqual(session.collection_state, 'outliers')
        stamp += .08
        session.feed(WebcamObservation(11, stamp, feature), stamp)
        self.assertTrue(session.ready, session.hold_reason)
        self.assertTrue(session.confirm(stamp + .01))

    def test_single_tracking_spike_is_trimmed_from_fit_but_not_raw_check(self):
        feature = feature_for(TRAIN_TARGETS[0])
        session = WebcamSession(BOUNDS, 'camera-1')
        for i in range(11):
            row = list(feature)
            if i == 5:
                row[0] += .10
            stamp = i * .08
            session.feed(WebcamObservation(i, stamp, tuple(row)), stamp)
        self.assertTrue(session.ready, session.hold_reason)
        self.assertEqual(session.sample_count, 10)
        self.assertEqual(session.excluded_count, 1)
        self.assertLessEqual(session.excluded_count / len(session.frames), .15)
        self.assertTrue(session.confirm(11 * .08))
        np.testing.assert_allclose(session.train[0]['features'], feature, atol=1e-12)

        profile, _ = fitted_profile()
        session.phase = 'check'
        session.index = 0
        session.candidate = profile
        check_feature = feature_for(CHECK_TARGETS[0])
        spike = list(check_feature)
        spike[0] += .10
        for i in range(11):
            row = tuple(spike) if i == 5 else check_feature
            stamp = 2.0 + i * .08
            session.feed(WebcamObservation(100 + i, stamp, row), stamp)
        self.assertTrue(session.ready, session.hold_reason)
        self.assertTrue(session.confirm(2.95))
        raw_rows = session.check[0]['features']
        self.assertEqual(len(raw_rows), 11)
        self.assertIn(tuple(spike), raw_rows)

    def test_persistent_motion_restarts_candidate_without_readying(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = list(feature_for(TRAIN_TARGETS[0]))
        for i in range(30):
            row = list(feature)
            row[5] += i * .008
            stamp = i * .12
            session.feed(WebcamObservation(i, stamp, tuple(row)), stamp)
        self.assertFalse(session.ready)
        self.assertLess(session.sample_count, 10)
        self.assertEqual(session.collection_state, 'moving')
        self.assertIn('after moving', session.hold_reason)

    def test_out_of_order_frame_clears_current_fixation(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        for i in range(8):
            stamp = i * .08
            session.feed(WebcamObservation(i, stamp, feature), stamp)
        session.feed(WebcamObservation(3, .64, feature), .64)
        self.assertFalse(session.ready)
        self.assertEqual(session.sample_count, 0)
        self.assertEqual(session.frames, [])
        self.assertFalse(session.confirm(.64))

    def test_stale_new_sequence_and_stream_gap_clear_fixation(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = feature_for(TRAIN_TARGETS[0])
        for i in range(5):
            stamp = i * .04
            session.feed(WebcamObservation(i, stamp, feature), stamp)
        session.feed(WebcamObservation(5, .20, feature), .50)
        self.assertFalse(session.frames)
        for i in range(6, 9):
            stamp = .50 + (i - 6) * .04
            session.feed(WebcamObservation(i, stamp, feature), stamp)
        self.assertEqual(len(session.frames), 3)
        session.feed(WebcamObservation(9, .78, feature), .78)
        self.assertEqual(len(session.frames), 1)
        self.assertFalse(session.ready)

    def test_rolling_window_detects_gradual_pose_drift(self):
        session = WebcamSession(BOUNDS, 'camera-1')
        feature = list(feature_for(TRAIN_TARGETS[0]))
        for i in range(20):
            sample = list(feature)
            sample[5] += i * .006
            stamp = i * .04
            session.feed(WebcamObservation(i, stamp, tuple(sample)), stamp)
        self.assertFalse(session.ready)
        self.assertLess(len(session.frames), 20)
        matrix = np.asarray([row for _, row in session.frames])
        self.assertLessEqual(float(np.ptp(matrix[:, 5])), .06 + 1e-12)


if __name__ == '__main__':
    unittest.main()
