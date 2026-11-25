import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from context_operator.core.contracts import Rect
from context_operator.personalization import StarterSession
from context_operator.personalization.session import STARTER_PROTOCOL
from context_operator.providers.gaze import learning_path
from context_operator.personalization import geometry
from context_operator.personalization.model import StarterProfile
from context_operator.personalization.history import preserve_attempt
from gaze_fixture import PIXEL_MM, burst, reading, turned


BOUNDS = Rect(0, 0, 1920, 1080)
METADATA = {'display': BOUNDS.as_list(), 'engine': 'arkit-truedepth-v1',
            'features': 22, 'protocol': STARTER_PROTOCOL}


def pixels(fractions, bounds=BOUNDS):
    return np.asarray([[bounds.x+bounds.w*x, bounds.y+bounds.h*y] for x, y in fractions])


def feature_for(target, *, bounds=BOUNDS, **seat):
    return reading(target, bounds, **seat)[0]


def starter_profile(bounds=BOUNDS, *, offset=0, pixel_mm=PIXEL_MM, camera=None, **seat):
    targets = pixels(StarterSession.train_targets, bounds)
    bursts, origins = zip(*(burst(target, bounds, camera=camera, **seat) for target in targets))
    return StarterProfile.fit(bursts, origins, targets+[offset, 0], dict(METADATA, display=bounds.as_list()), pixel_mm)


class UncertainCandidate:
    def predict(self, features):
        x = np.asarray(features)
        return x[..., :2]

    def support_reason(self, features):
        return 'eye_direction' if float(np.asarray(features)[0]) == -100. else ''


class StarterTests(unittest.TestCase):
    def assert_lands_on(self, profile, targets, limit, **seat):
        for target in targets:
            prediction = profile.predict(feature_for(target, **seat))
            self.assertLess(np.linalg.norm(prediction-target), limit, (target, seat))

    def test_setup_finds_the_screen_and_round_trips(self):
        profile = starter_profile()
        self.assertEqual(profile.training_x.shape, (len(StarterSession.train_targets), 22))
        self.assertAlmostEqual(profile.settings['viewing_m'], .76, delta=.03)
        np.testing.assert_allclose(profile.arrays['gains'], geometry.EYE_GAIN, atol=.02)
        self.assert_lands_on(profile, pixels(StarterSession.check_targets), 12.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'starter.npz'
            profile.save(path)
            loaded = StarterProfile.load(path, profile.metadata)
            self.assertTrue(all(loaded.supports(row) for row in profile.training_x))
            np.testing.assert_array_equal(loaded.predict(profile.training_x), profile.predict(profile.training_x))
        grazing = feature_for([960, 540])
        grazing[7:13] = [0, 0, 0, 1, -1, 0]
        grazing[13:19] = [1, 0, .01, 1, 0, .01]
        self.assertTrue(np.isnan(profile.predict(grazing)).all())

    def test_slow_drift_left_by_the_plane_is_corrected_between_setup_points(self):
        def drift(points):
            return np.c_[40*np.sin(points[:, 0]/BOUNDS.w*np.pi), 50*np.cos(points[:, 1]/BOUNDS.h*np.pi)]
        targets = pixels(StarterSession.train_targets)
        bursts, origins = zip(*(burst(target, BOUNDS) for target in targets))
        profile = StarterProfile.fit(bursts, origins, targets+drift(targets), METADATA, PIXEL_MM)
        checks = pixels(StarterSession.check_targets)
        rows = np.asarray([feature_for(target) for target in checks])
        plane = geometry.project(rows, profile.display, profile.arrays['frame'], profile.arrays['center'],
                                 profile.settings['pixel_m'], profile.arrays['eye_offsets'], profile.arrays['gains'])
        wanted = checks+drift(checks)
        self.assertLess(np.linalg.norm(profile.predict(rows)-wanted, axis=1).mean(),
                        np.linalg.norm(plane-wanted, axis=1).mean()/2)

    def test_one_stray_frame_does_not_fail_setup(self):
        targets = pixels(StarterSession.train_targets)
        bursts, origins = zip(*(burst(target, BOUNDS) for target in targets))
        bursts[4][0, 13:19] = [0, 0, -1, 0, 0, -1]
        profile = StarterProfile.fit(bursts, origins, targets, METADATA, PIXEL_MM)
        self.assert_lands_on(profile, pixels(StarterSession.check_targets), 15.)

    def test_head_turn_and_eyes_alone_reach_the_same_pixel(self):
        checks = pixels(StarterSession.check_targets)
        self.assert_lands_on(starter_profile(turn=1.), checks, 12., turn=0.)
        self.assert_lands_on(starter_profile(turn=0.), checks, 12., turn=1.)

    def test_moving_the_head_after_setup_keeps_the_mapping(self):
        profile = starter_profile(turn=.5)
        moved = dict(head=(.05, .02, -.60), turn=.2)
        self.assertTrue(profile.supports(feature_for([960, 540], **moved)))
        self.assert_lands_on(profile, pixels(StarterSession.check_targets), 15., **moved)

    def test_phone_on_its_side_and_leaning_back_fits_the_same(self):
        lean, side = np.radians(30), np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        camera = side@np.array([[1, 0, 0], [0, np.cos(lean), -np.sin(lean)], [0, np.sin(lean), np.cos(lean)]])
        profile = starter_profile(turn=.5, camera=camera)
        for target in pixels(StarterSession.check_targets):
            row, _ = turned(*reading(target, BOUNDS, turn=.2), camera)
            self.assertLess(np.linalg.norm(profile.predict(row)-target), 12.)

    def test_screen_of_unknown_size_fits_and_is_trusted_only_near_the_seat(self):
        profile = starter_profile(pixel_mm=None, pitch_mm=.15)
        self.assertFalse(profile.settings['screen_size_known'])
        self.assert_lands_on(profile, pixels(StarterSession.check_targets), 12., pitch_mm=.15)
        self.assert_lands_on(starter_profile(pixel_mm=40.), pixels(StarterSession.check_targets), 12.)
        shifted = dict(head=(.1, 0., -.7))
        self.assertEqual(profile.support_reason(feature_for([960, 540], pitch_mm=.15, **shifted)), 'head_position')
        self.assertEqual(starter_profile().support_reason(feature_for([960, 540], **shifted)), '')

    def test_support_follows_the_seat_used_in_setup(self):
        profile = starter_profile(turn=.8)
        feature = feature_for([960, 540], turn=.8)
        self.assertEqual(profile.support_reason(feature), '')
        far = feature.copy()
        far[6] -= .26
        self.assertEqual(profile.support_reason(far), 'head_position')
        turned = feature.copy()
        turned[7:13] = [0, 0, 0, 1, -1, 0]
        self.assertEqual(profile.support_reason(turned), 'head_rotation')
        self.assertEqual(profile.support_reason(feature[:21]), 'invalid_features')
        # That turn seen only from another seat does not make it known here.
        profile.training_x[:6, 4] += .3
        profile.training_x[:6, 7:13] = turned[7:13]
        self.assertEqual(profile.support_reason(turned), 'head_rotation')

    def test_incomplete_fixations_are_rejected(self):
        targets = pixels(StarterSession.train_targets)
        bursts, origins = zip(*(burst(target, BOUNDS) for target in targets))
        with self.assertRaises(ValueError):
            StarterProfile.fit(bursts[:5], origins[:5], targets[:5], METADATA, PIXEL_MM)
        with self.assertRaises(ValueError):
            StarterProfile.fit(bursts, origins[:-1], targets, METADATA, PIXEL_MM)

    def test_target_layout_is_thirteen_plus_five_and_disjoint(self):
        train = set(StarterSession.train_targets)
        check = set(StarterSession.check_targets)
        self.assertEqual(len(train), 13)
        self.assertEqual(len(check), 5)
        self.assertFalse(train & check)

    def test_superseded_setup_archive_starts_over(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'session.json'
            archive.write_text(json.dumps({'protocol': 'personal-arkit-starter-v1', 'display': BOUNDS.as_list(),
                                           'train': [{'phase': 'train'}], 'check': []}), encoding='utf-8')
            session = StarterSession.resume(BOUNDS, archive)
            self.assertEqual((session.phase, session.index, session.train), ('train', 0, []))

    def test_setup_archive_with_wrong_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'session.json'
            archive.write_text(json.dumps({'protocol': 'unknown', 'display': BOUNDS.as_list(),
                                           'train': [], 'check': []}), encoding='utf-8')
            with self.assertRaises(ValueError):
                StarterSession.resume(BOUNDS, archive)

    def test_short_capture_resumes_and_keeps_five_checks_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'starter.json'
            session = StarterSession(BOUNDS, archive, PIXEL_MM)
            seq = 0
            stamp = 100.

            def collect(target):
                nonlocal seq, stamp
                self.assertEqual(session.target, target)
                feature, origins = reading(target, BOUNDS)
                sample = None
                advanced = False
                for frame in range(36):
                    seq += 1
                    captured = stamp+1.+frame*.05
                    sample = SimpleNamespace(seq=seq, captured=captured, features=feature.copy(),
                                             eye_origins=origins, error='', blink=False)
                    session.feed(sample, captured+.005)
                    if session.ready_to_confirm:
                        advanced = session.confirm(captured+.01)
                        if advanced:
                            stamp = captured+.5
                            break
                self.assertTrue(advanced, session.hold_reason)

            for index, fraction in enumerate(session.train_targets):
                if index == 3:
                    resumed = StarterSession.resume(BOUNDS, archive)
                    self.assertEqual(resumed.index, 3)
                    session = resumed
                collect([fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h])
            self.assertEqual(session.phase, 'fit')
            self.assertEqual(len(session.train), len(session.train_targets))
            candidate = session.fit()
            session.begin_check(candidate, stamp)
            for fraction in session.check_targets:
                collect([fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h])
            self.assertTrue(session.complete)
            self.assertFalse(set(session.train_targets) & set(session.check_targets))
            report = session.evaluate()
            self.assertEqual(report['scope'], 'starter-independent-targets')
            self.assertEqual(report['quality'], 'starter')
            self.assertFalse(report['passed'])
            self.assertTrue(report['usable'], report)
            session.persist_result()
            preserve_attempt(archive, report)
            self.assertTrue((Path(folder)/'starter-last-result.json').is_file())
            self.assertNotIn('video', archive.read_text(encoding='utf-8'))
            self.assertEqual(learning_path('abcdef1234567890-extra', folder).name,
                             'gaze-adaptation-personal-abcdef1234567890.json')
            loaded = StarterSession.resume(BOUNDS, archive)
            self.assertEqual(loaded.phase, 'result')
            self.assertIsNotNone(loaded.candidate)
            probe = feature_for([960, 540])
            np.testing.assert_array_equal(loaded.candidate.predict(probe), candidate.predict(probe))

            candidate_path = archive.parent/'starter-candidates'/f'{loaded.candidate_key}.npz'
            candidate_path.write_bytes(b'corrupt')
            session = StarterSession.resume(BOUNDS, archive)
            self.assertIsNone(session.candidate)
            self.assertTrue(session.report['usable'])
            session.phase = 'fit'
            session.begin_check(session.fit(), stamp)

            pending = json.loads(archive.read_text(encoding='utf-8'))
            self.assertEqual(pending['check'], [])
            self.assertIsNone(pending.get('report'))
            self.assertIsNone(pending.get('candidate_sha256'))
            self.assertIsNone(session.report)
            for fraction in session.check_targets:
                collect([fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h])
            session.evaluate()
            session.persist_result()
            loaded = StarterSession.resume(BOUNDS, archive)
            self.assertTrue(loaded.report['usable'])
            self.assertIsNotNone(loaded.candidate)
            self.assertEqual(loaded.candidate_key, candidate_path.stem)

    def test_one_bad_check_target_blocks_coarse_usability(self):
        session = StarterSession(BOUNDS)
        session.phase = 'result'
        session.candidate = UncertainCandidate()
        for index, fraction in enumerate(session.check_targets):
            target = [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h]
            x = np.zeros((12, 22))
            x[:, :2] = target
            if index == 3:
                x[:, 0] = -100.
            session.check.append({'phase': 'check', 'index': index,
                                  'target': target, 'features': x.tolist(),
                                  'eye_origins': np.zeros((12, 2, 3)).tolist()})
        report = session.evaluate()
        self.assertEqual(report['quality'], 'starter')
        self.assertFalse(report['passed'])
        self.assertFalse(report['usable'])
        self.assertEqual(report['targets'][3]['coverage'], 0.)
        self.assertEqual(report['targets'][3]['supported_fraction'],0.)
        self.assertEqual(report['targets'][3]['onscreen_fraction'],0.)
        self.assertTrue(any('gaze often fell outside the screen' in reason for reason in report['problems']))
        self.assertTrue(any('eye and head position fell outside the setup range' in reason for reason in report['problems']))

if __name__ == '__main__':
    unittest.main()
