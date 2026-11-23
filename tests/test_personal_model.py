import tempfile
import unittest
from pathlib import Path

import numpy as np

from context_operator.personalization.model import PersonalProfile


METADATA = {'display': [0, 0, 1920, 1080], 'engine': 'arkit-truedepth-v1',
            'features': 22, 'protocol': 'personal-arkit-direct-v1'}


def observations(*, obscured_posture=False, unequal_bursts=False):
    bursts, targets, groups, postures = [], [], [], []
    for posture in range(3):
        for index in range(12):
            fx, fy = (index % 4 + 1)/5, (index // 4 + 1)/4
            count = 12
            frame = np.zeros((count, 22))
            measured_x = fx if not (obscured_posture and posture == 2) else .5
            frame[:, 0] = measured_x
            frame[:, 1] = fy
            frame[:, 2] = frame[:, 0] + .06
            frame[:, 3] = fy
            frame[:, 6] = -.7 + .1*(1-posture)
            frame[:, 13] = measured_x
            frame[:, 16] = measured_x
            frame[:, 19] = measured_x
            frame[:, 20] = fy
            frame[:, 0] += np.linspace(-.002, .002, count)
            if unequal_bursts and index == 0:
                frame = np.tile(frame, (2, 1))
            bursts.append(frame)
            targets.append([fx*1920, fy*1080])
            groups.append(index)
            postures.append(posture)
    return bursts, targets, groups, postures


class PersonalModelTests(unittest.TestCase):
    def test_one_equal_weight_observation_per_fixation_and_round_trip(self):
        bursts, targets, groups, postures = observations(unequal_bursts=True)
        profile = PersonalProfile.fit(bursts, targets, METADATA, groups, postures)
        self.assertEqual(profile.training_x.shape, (36, 22))
        self.assertEqual(profile.training_y.shape, (36, 2))
        self.assertEqual(profile.settings['model_kind'], PersonalProfile.MODEL_KIND)
        self.assertLessEqual(profile.arrays['coef'].shape[0], 54)
        self.assertTrue(profile.supports(bursts[0][0]))
        prediction = profile.predict(bursts[17][0])
        equal = observations()
        equal_profile = PersonalProfile.fit(*equal[:2], METADATA, *equal[2:])
        self.assertEqual(profile.settings['model'], equal_profile.settings['model'])
        np.testing.assert_allclose(profile.predict(bursts[17][0]),
                                   equal_profile.predict(bursts[17][0]), atol=1e-8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'personal.npz'
            profile.save(path)
            loaded = PersonalProfile.load(path, METADATA)
            np.testing.assert_allclose(loaded.predict(bursts[17][0]), prediction)

    def test_training_cv_keeps_bad_posture_visible(self):
        bursts, targets, groups, postures = observations(obscured_posture=True)
        profile = PersonalProfile.fit(bursts, targets, METADATA, groups, postures)
        self.assertEqual(len(profile.settings['cv_by_target']), len(PersonalProfile.CANDIDATES))
        for result in profile.settings['cv_by_target']:
            self.assertEqual(len(result['posture_p90_px']), 3)
            self.assertEqual(result['worst_posture_p90_px'], max(result['posture_p90_px']))
            self.assertGreaterEqual(result['score'], result['median_px'])
        chosen = min(profile.settings['cv_by_target'], key=lambda r: r['score'])
        self.assertGreater(chosen['posture_p90_px'][2], 250)

    def test_invalid_burst_and_insufficient_target_groups_fail(self):
        bursts, targets, groups, postures = observations()
        with self.assertRaises(ValueError):
            PersonalProfile.fit(bursts[:8], targets[:8], METADATA, groups[:8], postures[:8])
        bursts[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            PersonalProfile.fit(bursts, targets, METADATA, groups, postures)


if __name__ == '__main__':
    unittest.main()
