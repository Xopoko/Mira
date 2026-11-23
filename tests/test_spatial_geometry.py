import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from context_operator.foundations.gaze.spatial import SpatialProfile


BOUNDS = [0, 0, 1200, 800]
CENTER = np.array([.015, .06, .045])
RIGHT = np.array([.56, 0, 0])
DOWN = np.array([0, -.36, .04])
EYES = np.array([[-.032, .018, .025], [.032, .018, .025]])
ULTRA_BOUNDS = [0, 0, 5120, 1440]
G9_GEOMETRY = {'kind': 'cylinder', 'width_m': 1.191936,
               'height_m': .335232, 'radius_m': 1.0}
ULTRA_ROTATION = Rotation.from_rotvec([.22, .11, -.06]).as_matrix()
ULTRA_CENTER = np.array([.02, .24, .12])
ULTRA_RIGHT = ULTRA_ROTATION @ np.array([1.20, 0, 0])
ULTRA_DOWN = ULTRA_ROTATION @ np.array([0, -.3375, 0])
ANGULAR_GAIN = np.array([.56, .42])
ANGULAR_BIAS = np.array([[.075, -.055], [-.055, .045]])


def observation(u, v, face, pose=(0, 0, 0), camera_rotation=None,
                camera_translation=None, vertical=DOWN):
    """Generate physically consistent ARKit vectors for a known screen target."""
    head = Rotation.from_rotvec(pose).as_matrix()
    target = CENTER + (u - .5) * RIGHT + (v - .5) * vertical
    origins = face + EYES @ head.T
    directions = target - origins
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    if camera_rotation is not None:
        # Given current-camera -> canonical transform, encode the new camera view.
        r = camera_rotation
        t = camera_translation
        origins = (origins - t) @ r
        face = (face - t) @ r
        directions = directions @ r
        head = r.T @ head
    intersections = origins[:, :2] - origins[:, 2:3] * directions[:, :2] / directions[:, 2:3]
    look = np.mean(directions, axis=0)
    look /= np.linalg.norm(look)
    features = np.r_[intersections.ravel(), face, head[:, :2].ravel(),
                     directions.ravel(), look]
    return features, origins


def training_records(vertical=DOWN):
    result = []
    positions = [np.array([-.045, .005, -.58]), np.array([.055, -.028, -.78]),
                 np.array([.008, .048, -.67])]
    for v in (.1, .5, .9):
        for u in (.1, .5, .9):
            samples = [observation(u, v, face + np.array([0, 0, j*.001]),
                                   pose=(.03, -.04, .02), vertical=vertical)
                       for face in positions for j in range(3)]
            result.append({'target': [u*1200, v*800],
                           'features': np.array([sample[0] for sample in samples]),
                           'eye_origins': np.array([sample[1] for sample in samples])})
    return result


def reanchor_records(rotation, translation, good=True):
    result = []
    for u, v in ((.1, .1), (.9, .1), (.5, .5), (.1, .9), (.9, .9)):
        face = np.array([.025, .012, -.69])
        samples = [observation(u, v, face, camera_rotation=rotation,
                               camera_translation=translation) for _ in range(7)]
        target = [u*1200, v*800]
        if not good and v == .1:
            target = [(1-u)*1200, v*800]
        result.append({'target': target,
                       'features': np.array([sample[0] for sample in samples]),
                       'eye_origins': np.array([sample[1] for sample in samples])})
    return result


def ultra_observation(u, v, face, pose=(0, 0, 0), camera_rotation=None,
                      camera_translation=None, angular_noise=(0, 0),
                      screen_center=ULTRA_CENTER, screen_right=ULTRA_RIGHT,
                      screen_down=ULTRA_DOWN, curved=False):
    head = Rotation.from_rotvec(pose).as_matrix()
    if curved:
        right = screen_right / np.linalg.norm(screen_right)
        down = screen_down / np.linalg.norm(screen_down)
        normal = np.cross(right, down)
        theta = (u - .5)*G9_GEOMETRY['width_m']/G9_GEOMETRY['radius_m']
        target = (screen_center + np.sin(theta)*right + (1-np.cos(theta))*normal +
                  (v-.5)*G9_GEOMETRY['height_m']*down)
    else:
        target = screen_center + (u - .5)*screen_right + (v - .5)*screen_down
    origins = face + EYES @ head.T
    true = target - origins
    true /= np.linalg.norm(true, axis=1, keepdims=True)
    local = true @ head
    slopes = local[:, :2] / local[:, 2:]
    measured_slopes = slopes * ANGULAR_GAIN + ANGULAR_BIAS + angular_noise
    measured_local = np.c_[measured_slopes, np.ones(2)]
    measured_local /= np.linalg.norm(measured_local, axis=1, keepdims=True)
    directions = measured_local @ head.T
    if camera_rotation is not None:
        r = camera_rotation
        t = camera_translation
        origins = (origins - t) @ r
        face = (face - t) @ r
        directions = directions @ r
        head = r.T @ head
    intersections = origins[:, :2] - origins[:, 2:3] * directions[:, :2] / directions[:, 2:3]
    look = np.mean(directions, axis=0)
    look /= np.linalg.norm(look)
    features = np.r_[intersections.ravel(), face, head[:, :2].ravel(),
                     directions.ravel(), look]
    return features, origins


def ultra_records(camera_rotation=None, camera_translation=None, reanchor=False, curved=False):
    rng = np.random.default_rng(113)
    result = []
    positions = [np.array([-.12, -.035, -.58]), np.array([.095, .052, -.91]),
                 np.array([.025, -.004, -.72])]
    cells = ((.10, .10), (.90, .10), (.50, .50), (.10, .90), (.90, .90)) if reanchor else [
        (u, v) for v in (.08, .50, .92) for u in (.08, .36, .64, .92)]
    for u, v in cells:
        samples = []
        for index in range(24):
            face = np.array([.025, .005, -.74]) if reanchor else positions[index % 3]
            face = face + rng.normal(0, .001, 3)
            pose = (.025, -.03, .015) if reanchor else (.04, -.05, .02)
            noise = rng.normal(0, .0035, (2, 2))
            samples.append(ultra_observation(u, v, face, pose, camera_rotation,
                                             camera_translation, noise, curved=curved))
        result.append({'target': [u*5120, v*1440],
                       'features': np.array([row[0] for row in samples]),
                       'eye_origins': np.array([row[1] for row in samples])})
    return result


class SpatialGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = SpatialProfile.fit_records(training_records(), BOUNDS)

    def test_predicts_changed_head_pose_and_distance(self):
        for face in (np.array([.09, -.035, -.49]), np.array([-.08, .04, -.95])):
            for u, v in ((.24, .66), (.75, .3)):
                features, _ = observation(u, v, face, pose=(.10, -.08, .06))
                self.assertTrue(self.profile.supports(features))
                np.testing.assert_allclose(self.profile.predict(features), [u*1200, v*800], atol=2)

    def test_skewed_effective_plane_uses_joint_screen_coordinates(self):
        vertical = np.array([.045, -.36, .04])
        profile = SpatialProfile.fit_records(training_records(vertical), BOUNDS)
        features, _ = observation(.73, .27, np.array([.04, -.02, -.88]),
                                  pose=(.06, -.04, .02), vertical=vertical)
        np.testing.assert_allclose(profile.predict(features), [.73*1200, .27*800], atol=2)

    def test_biased_ultrawide_rays_recover_metric_geometry_and_holdout(self):
        records = ultra_records()
        profile = SpatialProfile.fit_records(records, ULTRA_BOUNDS)
        self.assertEqual(profile.settings['kind'], 'spatial-v2')
        self.assertEqual(profile.settings['fit_diagnostics'].get('fit_mode'), None)
        # This fixture reproduces the previous post-collection rejection:
        # fitting the same uncorrected rays made the screen <15 cm tall.
        self.assertLess(profile.settings['fit_diagnostics']['raw_plane_height_m'], .15)
        self.assertTrue(.7 < np.linalg.norm(profile.arrays['plane_u']) < 1.7)
        np.testing.assert_allclose(profile.arrays['angular_gain'], ANGULAR_GAIN, atol=.12)
        rng = np.random.default_rng(91)
        for z in (-.49, -.99):
            errors = []
            for u, v in ((.22, .2), (.48, .73), (.77, .41), (.31, .59)):
                for _ in range(10):
                    raw, _ = ultra_observation(u, v, np.array([.06, -.02, z]),
                                               (.075, -.05, .03),
                                               angular_noise=rng.normal(0, .0035, (2, 2)))
                    self.assertTrue(profile.supports(raw))
                    errors.append(np.linalg.norm(profile.predict(raw) - [u*5120, v*1440]))
            self.assertLess(np.median(errors), 90)
            self.assertLess(np.percentile(errors, 90), 160)

    def test_biased_ultrawide_reanchor_keeps_personal_correction(self):
        profile = SpatialProfile.fit_records(ultra_records(), ULTRA_BOUNDS)
        camera = Rotation.from_rotvec([.12, -.14, .07]).as_matrix()
        shift = np.array([.085, -.04, .035])
        reanchored = profile.reanchor(ultra_records(camera, shift, reanchor=True))
        self.assertEqual(reanchored.fingerprint(), profile.fingerprint())
        errors = []
        for u, v in ((.23, .68), (.72, .31), (.51, .46)):
            raw, _ = ultra_observation(u, v, np.array([-.06, .04, -.86]),
                                       (-.06, .04, -.025), camera, shift)
            canonical = reanchored.canonicalize(raw)
            self.assertTrue(reanchored.supports(canonical))
            errors.append(np.linalg.norm(reanchored.predict(canonical) - [u*5120, v*1440]))
        self.assertLess(max(errors), 100)

    def test_known_1000r_cylinder_predicts_heldout_targets_and_depths(self):
        profile = SpatialProfile.fit_records(ultra_records(curved=True), ULTRA_BOUNDS,
                                             screen_geometry=G9_GEOMETRY)
        self.assertEqual(profile.settings['kind'], 'spatial-v3')
        self.assertEqual(profile.metadata['screen_geometry'], G9_GEOMETRY)
        self.assertAlmostEqual(np.linalg.norm(profile.arrays['plane_u']), G9_GEOMETRY['width_m'], places=5)
        self.assertAlmostEqual(np.linalg.norm(profile.arrays['plane_v']), G9_GEOMETRY['height_m'], places=5)
        rng = np.random.default_rng(44)
        for depth in (-.5, -.95):
            errors = []
            for u, v in ((.18, .68), (.79, .22), (.47, .81), (.93, .48)):
                for _ in range(10):
                    raw, _ = ultra_observation(u, v, np.array([-.035, .03, depth]),
                                               (.05, -.06, .02),
                                               angular_noise=rng.normal(0, .0035, (2, 2)),
                                               curved=True)
                    self.assertTrue(profile.supports(raw))
                    errors.append(np.linalg.norm(profile.predict(raw) - [u*5120, v*1440]))
            self.assertLess(np.median(errors), 90)
            self.assertLess(np.percentile(errors, 90), 160)

    def test_known_cylinder_rigid_reanchor_and_load(self):
        profile = SpatialProfile.fit_records(ultra_records(curved=True), ULTRA_BOUNDS,
                                             screen_geometry=G9_GEOMETRY)
        camera = Rotation.from_rotvec([.11, -.13, .06]).as_matrix()
        shift = np.array([.08, -.045, .03])
        candidate = profile.reanchor(ultra_records(camera, shift, reanchor=True, curved=True))
        self.assertEqual(candidate.fingerprint(), profile.fingerprint())
        raw, _ = ultra_observation(.74, .32, np.array([.02, -.03, -.82]),
                                   (-.04, .03, .02), camera, shift, curved=True)
        canonical = candidate.canonicalize(raw)
        self.assertTrue(candidate.supports(canonical))
        self.assertLess(np.linalg.norm(candidate.predict(canonical)-[.74*5120, .32*1440]), 100)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'curved.npz'
            candidate.save(path)
            restored = SpatialProfile.load(path, candidate.metadata)
            self.assertEqual(restored.fingerprint(), profile.fingerprint())
            np.testing.assert_allclose(restored.predict(restored.canonicalize(raw)),
                                       candidate.predict(canonical), atol=1e-6)

    def test_angular_scale_without_depth_evidence_is_rejected(self):
        # Lateral motion with a frontoparallel plane leaves a metric depth/gain
        # gauge despite many frames and apparent head-translation coverage.
        positions = [np.array([-.12, -.04, -.74]), np.array([.10, .05, -.74]),
                     np.array([.02, -.02, -.74])]
        records = []
        for v in (.08, .50, .92):
            for u in (.08, .36, .64, .92):
                samples = [ultra_observation(u, v, face,
                                             screen_center=np.array([0, .2, .1]),
                                             screen_right=np.array([1.2, 0, 0]),
                                             screen_down=np.array([0, -.3375, 0]))
                           for face in positions for _ in range(8)]
                records.append({'target': [u*5120, v*1440],
                                'features': np.array([row[0] for row in samples]),
                                'eye_origins': np.array([row[1] for row in samples])})
        from context_operator.foundations.gaze.spatial import SpatialFitError
        with self.assertRaises(SpatialFitError) as failure:
            SpatialProfile.fit_records(records, ULTRA_BOUNDS)
        self.assertEqual(failure.exception.code, 'underdetermined_angular_fit')
        self.assertGreater(failure.exception.diagnostics['fit_condition'], 1e5)

    def test_rigid_reanchor_preserves_personal_frame_and_fingerprint(self):
        r = Rotation.from_rotvec([.13, -.17, .08]).as_matrix()
        t = np.array([.075, -.045, .035])
        new = self.profile.reanchor(reanchor_records(r, t))
        self.assertEqual(new.fingerprint(), self.profile.fingerprint())
        for u, v in ((.3, .25), (.7, .62)):
            raw, _ = observation(u, v, np.array([-.04, .03, -.84]),
                                 pose=(-.08, .06, -.03), camera_rotation=r,
                                 camera_translation=t)
            canonical = new.canonicalize(raw)
            self.assertTrue(new.supports(canonical))
            np.testing.assert_allclose(new.predict(canonical), [u*1200, v*800], atol=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'spatial.npz'
            new.save(path)
            restored = SpatialProfile.load(path, new.metadata)
            self.assertEqual(restored.fingerprint(), self.profile.fingerprint())
            np.testing.assert_allclose(restored.predict(restored.canonicalize(raw)),
                                       new.predict(canonical), atol=1e-6)

    def test_reverse_grazing_and_malformed_rays_are_rejected(self):
        features, _ = observation(.5, .5, np.array([0, 0, -.7]))
        for direction in ([0, 0, -1], [1, 0, .01]):
            bad = features.copy()
            bad[13:16] = direction
            bad[16:19] = direction
            self.assertFalse(self.profile.supports(bad))
            with self.assertRaises(ValueError):
                self.profile.predict(bad)
        bad = features.copy()
        bad[7:13] = 0
        self.assertFalse(self.profile.supports(bad))

    def test_degenerate_motion_bad_scale_and_bad_reanchor_fail(self):
        stationary = training_records()
        for record in stationary:
            u, v = np.asarray(record['target']) / [1200, 800]
            samples = [observation(u, v, np.array([0, 0, -.7])) for _ in range(9)]
            record['features'] = np.array([row[0] for row in samples])
            record['eye_origins'] = np.array([row[1] for row in samples])
        with self.assertRaisesRegex(ValueError, 'several distinct positions'):
            SpatialProfile.fit_records(stationary, BOUNDS)
        parallel = training_records()
        for record in parallel:
            record['features'][:, 13:19] = np.tile([0, 0, 1], 2)
            record['features'][:, 19:22] = [0, 0, 1]
            record['features'][:, :4] = record['eye_origins'][:, :, :2].reshape(-1, 4)
        with self.assertRaisesRegex(ValueError, 'locate the screen'):
            SpatialProfile.fit_records(parallel, BOUNDS)
        scaled = training_records()
        for record in scaled:
            record['eye_origins'] *= .2
            record['features'][:, 4:7] *= .2
            # The directions still encode a much larger monitor and cannot
            # yield a plausible physical width with the scaled origins.
        with self.assertRaises(ValueError):
            SpatialProfile.fit_records(scaled, BOUNDS)
        r = Rotation.from_rotvec([.13, -.17, .08]).as_matrix()
        with self.assertRaises(ValueError):
            self.profile.reanchor(reanchor_records(r, np.array([.075, -.045, .035]), good=False))

    def test_malformed_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'broken.npz'
            for key, value in [('camera_rotation', lambda a: a.__setitem__((0, 0), 2)),
                               ('camera_translation', lambda a: a.__setitem__(0, 2)),
                               ('eye_offsets', lambda a: a.__setitem__((0, 0), -.2)),
                               ('plane_center', lambda a: a.__setitem__(0, .2))]:
                with self.subTest(key=key):
                    arrays = {name: np.array(data, copy=True) for name, data in self.profile.arrays.items()}
                    value(arrays[key])
                    np.savez(path, **arrays, metadata=json_text(self.profile.metadata),
                             settings=json_text(self.profile.settings))
                    with self.assertRaises(ValueError):
                        SpatialProfile.load(path, self.profile.metadata)
            arrays=dict(self.profile.arrays);arrays.pop('training_y')
            np.savez(path,**arrays,metadata=json_text(self.profile.metadata),settings=json_text(self.profile.settings))
            with self.assertRaises(ValueError):SpatialProfile.load(path,self.profile.metadata)
            arrays = dict(self.profile.arrays)
            arrays.pop('angular_gain')
            arrays.pop('angular_bias')
            legacy = dict(self.profile.settings, kind='spatial-v1')
            np.savez(path, **arrays, metadata=json_text(self.profile.metadata), settings=json_text(legacy))
            with self.assertRaisesRegex(ValueError, 'Run setup again'):
                SpatialProfile.load(path, self.profile.metadata)


def json_text(value):
    import json
    return json.dumps(value)


if __name__ == '__main__':
    unittest.main()
