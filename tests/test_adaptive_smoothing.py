import math
import unittest

import numpy as np

from context_operator.foundations.gaze.smoothing import Smoother


def times_at_rate(rate, duration, *, irregular=False):
    values = [0.0]
    index = 1
    while values[-1] < duration:
        dt = 1.0 / rate
        if irregular:
            dt *= (0.65, 1.25, 0.85, 1.1, 0.75, 1.4)[(index - 1) % 6]
        values.append(values[-1] + dt)
        index += 1
    return values


def filter_trajectory(filter_factory, timestamps, points):
    smoother = filter_factory()
    return np.asarray([smoother.update(point, stamp)
                       for stamp, point in zip(timestamps, points)])


def jitter_rms(points, center):
    return float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))


def step_t90(timestamps, outputs, start, target):
    distance = float(np.linalg.norm(target - start))
    for stamp, point in zip(timestamps, outputs):
        if np.linalg.norm(target - point) <= distance * 0.1:
            return float(stamp - timestamps[0])
    raise AssertionError("step response did not reach 90 percent")


class AdaptiveSmoothingTests(unittest.TestCase):
    def test_stationary_jitter_stays_within_pixel_bounds_at_supported_rates(self):
        rng = np.random.default_rng(20260923)
        center = np.array([2500.0, 720.0])
        for rate, jitter_limit in ((30, 4.31), (60, 3.15), (120, 2.30)):
            with self.subTest(rate=rate):
                timestamps = times_at_rate(rate, 5.0)
                noise = rng.normal(0.0, 5.0, (len(timestamps), 2))
                points = center + noise
                adaptive = filter_trajectory(Smoother, timestamps, points)
                self.assertLessEqual(jitter_rms(adaptive[rate:], center), jitter_limit)

    def test_rapid_800_pixel_step_reaches_90_percent_without_overshoot(self):
        for rate in (30, 60, 120):
            with self.subTest(rate=rate):
                timestamps = times_at_rate(rate, 0.75)
                start = np.array([1000.0, 700.0])
                target = start + np.array([800.0, 0.0])
                points = np.asarray([start if stamp < 0.20 else target for stamp in timestamps])
                adaptive = filter_trajectory(Smoother, timestamps, points)
                post_step = [i for i, stamp in enumerate(timestamps) if stamp >= 0.20]
                adaptive_t90 = step_t90(timestamps[post_step[0]:], adaptive[post_step], start, target)
                self.assertLessEqual(adaptive_t90, .0434)
                self.assertTrue(np.all(adaptive[post_step, 0] >= start[0]))
                self.assertTrue(np.all(adaptive[post_step, 0] <= target[0]))

    def test_irregular_samples_keep_finite_bounded_step_response(self):
        timestamps = times_at_rate(60, 0.8, irregular=True)
        start = np.array([700.0, 700.0])
        target = np.array([1500.0, 1100.0])
        points = np.asarray([start if stamp < 0.2 else target for stamp in timestamps])
        output = filter_trajectory(Smoother, timestamps, points)
        self.assertTrue(np.isfinite(output).all())
        post_step = output[np.asarray(timestamps) >= 0.2]
        self.assertTrue(np.all(post_step >= start - 1e-9))
        self.assertTrue(np.all(post_step <= target + 1e-9))

    def test_repeated_or_earlier_timestamp_does_not_advance(self):
        smoother = Smoother()
        first = smoother.update((0, 0), 1.0)
        moved = smoother.update((800, 0), 1.02)
        last = smoother.last
        self.assertEqual(smoother.update((1200, 0), 1.02), moved)
        self.assertEqual(smoother.update((1200, 0), 1.01), moved)
        self.assertEqual(smoother.last, last)
        self.assertNotEqual(first, moved)

    def test_nonfinite_and_long_gap_reset_without_old_state_carryover(self):
        smoother = Smoother()
        smoother.update((100, 200), 1.0)
        self.assertIsNone(smoother.update((math.nan, 4), 1.01))
        self.assertIsNone(smoother.point)
        self.assertEqual(smoother.update((800, 900), 2.0), (800.0, 900.0))
        result = smoother.update((1200, 900), 2.4)
        self.assertEqual(result, (1200.0, 900.0))

    def test_reset_clears_public_and_internal_state(self):
        smoother = Smoother()
        smoother.update((1, 2), 0.0)
        smoother.reset()
        self.assertIsNone(smoother.point)
        self.assertIsNone(smoother.last)
        self.assertIsNone(smoother.update((float('inf'), 1), 1.0))


if __name__ == '__main__':
    unittest.main()
