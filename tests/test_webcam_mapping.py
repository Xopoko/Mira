import json
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from context_operator.core.contracts import Rect
from context_operator.personalization.webcam import (
    WebcamProfile, WebcamProfileStore, TRAIN_TARGETS, CHECK_TARGETS,
    LEGACY_WEBCAM_ENGINE_VERSION,
)

BOUNDS = Rect(0, 0, 3840, 2160)


def curved(target):
    u, y = np.asarray(target)-.5
    return (u, y-1.2*u*u, 0., 0., 0., .5, .5, .25)


class MappingTests(unittest.TestCase):
    def test_nonlinearity_selected_using_training_only_improves_unseen_targets(self):
        features = [curved(t) for t in TRAIN_TARGETS]
        model = WebcamProfile.fit(features, TRAIN_TARGETS, BOUNDS, 'camera')
        baseline = WebcamProfile.fit(features, TRAIN_TARGETS, BOUNDS, 'camera', legacy=True)
        self.assertEqual(model.mapping.kind, 'iris_quadratic')
        groups = [[curved(t)]*12 for t in CHECK_TARGETS]
        check = model.validate(groups, CHECK_TARGETS)
        self.assertLess(check.p90_px, baseline.validate(groups, CHECK_TARGETS).p90_px*.1)
        self.assertTrue(check.usable)
        frozen = (model.coefficients, model.mapping)
        wrong_groups = [[curved(target)]*12 for target in reversed(CHECK_TARGETS)]
        wrong_check = model.validate(wrong_groups, CHECK_TARGETS)
        self.assertGreater(wrong_check.p90_px, wrong_check.limits['p90_px'])
        self.assertIs(wrong_check.usable, False)
        self.assertEqual(frozen, (model.coefficients, model.mapping))

    def test_head_compensation_uses_per_frame_groups_and_grouped_cv(self):
        groups = []
        for index, target in enumerate(TRAIN_TARGETS):
            u, v = np.asarray(target)-.5
            rows = []
            for frame in range(20):
                yaw = .07*np.sin(index*1.7) + .012*np.sin(frame)
                pitch = .05*np.cos(index*1.3) + .012*np.cos(frame)
                rows.append((u-1.8*yaw, v-2*pitch, yaw, pitch, 0., .5, .5, .25))
            groups.append(rows)
        means = np.asarray(groups).mean(axis=1)
        model = WebcamProfile.fit(means, TRAIN_TARGETS, BOUNDS, 'camera', training_samples=groups)
        baseline = WebcamProfile.fit(means, TRAIN_TARGETS, BOUNDS, 'camera', legacy=True)
        self.assertIn(model.mapping.kind, ('iris_pose', 'iris_pose_quadratic'))
        # Fresh target positions with head movement not copied from training means.
        heldout = [(u-.5-1.8*.035, v-.5-2*(-.025), .035, -.025, 0., .5, .5, .25)
                   for u, v in CHECK_TARGETS]
        actual = np.asarray(CHECK_TARGETS)*np.asarray((BOUNDS.w, BOUNDS.h))
        errors = np.linalg.norm(model._predictions(heldout)-actual, axis=1)
        baseline_errors = np.linalg.norm(baseline._predictions(heldout)-actual, axis=1)
        self.assertLess(float(np.median(errors)), 10.)
        self.assertLess(float(np.median(errors)), float(np.median(baseline_errors))*.1)
        WebcamProfileStore._validate_model(model)
        # Nine means alone never select a saturated eight-feature head model.
        without_groups = WebcamProfile.fit(means, TRAIN_TARGETS, BOUNDS, 'camera')
        self.assertNotIn(without_groups.mapping.kind, ('iris_pose', 'iris_pose_quadratic'))
        duplicates = [[row]*20 for row in means]
        without_motion = WebcamProfile.fit(means, TRAIN_TARGETS, BOUNDS, 'camera', training_samples=duplicates)
        self.assertNotIn(without_motion.mapping.kind, ('iris_pose', 'iris_pose_quadratic'))

    def test_legacy_profile_remains_readable_and_new_mapping_receipt_recomputes(self):
        linear = [(x-.5, y-.5, 0., 0., 0., .5, .5, .25) for x, y in TRAIN_TARGETS]
        legacy = WebcamProfile.fit(linear, TRAIN_TARGETS, BOUNDS, 'camera', legacy=True)
        old_check = legacy.validate([[(x-.5, y-.5, 0., 0., 0., .5, .5, .25)]*12 for x, y in CHECK_TARGETS], CHECK_TARGETS)
        legacy = replace(legacy, validation_p90_px=old_check.p90_px)
        new = WebcamProfile.fit([curved(t) for t in TRAIN_TARGETS], TRAIN_TARGETS, BOUNDS, 'camera')
        check = new.validate([[curved(t)]*12 for t in CHECK_TARGETS], CHECK_TARGETS)
        new = replace(new, validation_p90_px=check.p90_px)
        with tempfile.TemporaryDirectory() as folder:
            store = WebcamProfileStore(folder)
            store.adopt(legacy, old_check)
            self.assertEqual(store.load(BOUNDS, 'camera').engine_version, LEGACY_WEBCAM_ENGINE_VERSION)
            store.adopt(new, check)
            self.assertEqual(store.load(BOUNDS, 'camera').mapping, new.mapping)
            payload = json.loads(store.path.read_text(encoding='utf-8'))
            payload['model']['mapping']['regularization'] = 99.
            payload.pop('sha256')
            import hashlib
            payload['sha256'] = hashlib.sha256(store._canonical(payload)).hexdigest()
            store.path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(store.load(BOUNDS, 'camera').engine_version, LEGACY_WEBCAM_ENGINE_VERSION)
            with self.assertRaises(ValueError):
                store._read(store.path, BOUNDS, 'camera')


if __name__ == '__main__':
    unittest.main()
