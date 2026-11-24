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
from context_operator.personalization.model import StarterProfile
from context_operator.personalization.history import preserve_attempt


BOUNDS = Rect(0, 0, 1920, 1080)


def feature_for(target, *, bounds=BOUNDS):
    fx = target[0] / bounds.w
    fy = target[1] / bounds.h
    px = (fx-.5)*.16
    py = (fy-.5)*.10
    direction = np.array([px/.7, py/.7, 1.])
    direction /= np.linalg.norm(direction)
    row = np.zeros(22)
    row[0:4] = [px-.005, py, px+.005, py]
    row[4:7] = [0., 0., -.7]
    row[7:13] = [1., 0., 0., 1., 0., 0.]
    row[13:19] = [*direction, *direction]
    row[19:22] = direction
    return row


def burst_for(target, *, bounds=BOUNDS):
    center = feature_for(target, bounds=bounds)
    burst = np.tile(center, (12, 1))
    noise = np.linspace(-.0004, .0004, len(burst))
    burst[:, 0] += noise
    burst[:, 2] -= noise
    return burst


def starter_profile(bounds=BOUNDS, *, offset=0):
    training_targets = np.asarray([
        [(bounds.x+bounds.w*x), (bounds.y+bounds.h*y)]
        for x, y in StarterSession.train_targets
    ])
    bursts = [burst_for(target, bounds=bounds) for target in training_targets]
    metadata = {'display': bounds.as_list(), 'engine': 'arkit-truedepth-v1',
                'features': 22, 'protocol': STARTER_PROTOCOL}
    return StarterProfile.fit(bursts, training_targets + [offset, 0], metadata)


class UncertainCandidate:
    def predict(self, features):
        x = np.asarray(features)
        return x[..., :2]

    def support_reason(self, features):
        return 'eye_direction' if float(np.asarray(features)[0]) == -100. else ''


class StarterTests(unittest.TestCase):
    def turning_head_profile(self):
        targets=np.array([[BOUNDS.w*x,BOUNDS.h*y] for x,y in StarterSession.train_targets])
        bursts=[]
        for target in targets:
            burst=burst_for(target)
            angle=(target[0]/BOUNDS.w-.5)*1.65
            c,s=np.cos(angle),np.sin(angle)
            burst[:,7:13]=[c,0,0,1,-s,0]
            bursts.append(burst)
        metadata={'display':BOUNDS.as_list(),'engine':'arkit-truedepth-v1','features':22,'protocol':STARTER_PROTOCOL}
        return StarterProfile.fit(bursts,targets,metadata)

    def test_training_head_turns_are_supported_after_reload(self):
        profile=self.turning_head_profile()
        self.assertTrue(all(profile.supports(row) for row in profile.training_x))
        extreme=profile.training_x[0].copy();extreme[7:13]=[0,0,0,1,-1,0]
        self.assertEqual(profile.support_reason(extreme),'head_rotation')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'profile.npz';profile.save(path)
            loaded=StarterProfile.load(path,profile.metadata)
            self.assertTrue(all(loaded.supports(row) for row in profile.training_x))
            np.testing.assert_array_equal(profile.predict(profile.training_x), loaded.predict(profile.training_x))

    def test_separate_known_head_and_eye_positions_do_not_imply_known_combination(self):
        profile=self.turning_head_profile()
        mixed=profile.training_x[0].copy();mixed[7:13]=profile.training_x[2,7:13]
        self.assertEqual(profile.support_reason(mixed),'head_eye_combination')

    def test_starter_is_baseline_free_affine_and_round_trips(self):
        profile = starter_profile()
        self.assertEqual(profile.arrays['training_x'].shape, (9, 22))
        self.assertEqual(profile.arrays['coef'].shape, (2, 2))
        self.assertEqual(profile.arrays['intercept'].shape, (2,))
        self.assertEqual(profile.metadata['protocol'], STARTER_PROTOCOL)
        for fraction in StarterSession.train_targets:
            target = [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h]
            prediction = profile.predict(feature_for(target))
            self.assertLess(np.linalg.norm(prediction-target), 35.)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'starter.npz'
            profile.save(path)
            loaded = StarterProfile.load(path, profile.metadata)
            np.testing.assert_allclose(loaded.predict(feature_for([960, 540])),
                                       profile.predict(feature_for([960, 540])))

    def test_starter_support_is_single_pose_and_bounds_large_head_motion(self):
        profile = starter_profile()
        feature = feature_for([BOUNDS.w*.45, BOUNDS.h*.45])
        self.assertTrue(profile.supports(feature))
        modest = feature.copy()
        modest[6] -= .05
        self.assertTrue(profile.supports(modest))
        far = feature.copy()
        far[6] -= .26
        self.assertFalse(profile.supports(far))

    def test_target_layout_is_nine_plus_five_and_disjoint(self):
        train = set(StarterSession.train_targets)
        check = set(StarterSession.check_targets)
        self.assertEqual(len(train), 9)
        self.assertEqual(len(check), 5)
        self.assertFalse(train & check)

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
            session = StarterSession(BOUNDS, archive)
            seq = 0
            stamp = 100.

            def collect(target):
                nonlocal seq, stamp
                self.assertEqual(session.target, target)
                feature = feature_for(target)
                sample = None
                advanced = False
                for frame in range(36):
                    seq += 1
                    captured = stamp+1.+frame*.05
                    sample = SimpleNamespace(seq=seq, captured=captured, features=feature.copy(),
                                             eye_origins=((0., 0., -.7), (.06, 0., -.7)),
                                             error='', blink=False)
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
            self.assertEqual(len(session.train), 9)
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
            np.testing.assert_array_equal(loaded.candidate.predict(np.zeros(22)), candidate.predict(np.zeros(22)))

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
