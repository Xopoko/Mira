"""Print deterministic synthetic comparisons for fixed and adaptive smoothing."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_operator.foundations.gaze.profile import Smoother as FixedSmoother
from context_operator.foundations.gaze.smoothing import Smoother


def timestamps(rate, duration):
    result = [0.0]
    while result[-1] < duration:
        result.append(result[-1] + 1.0 / rate)
    return result


def run(factory, stamps, points):
    smoother = factory()
    return np.asarray([smoother.update(point, stamp)
                       for stamp, point in zip(stamps, points)])


def t90(stamps, output, start, target):
    distance = np.linalg.norm(target - start) * .1
    for stamp, point in zip(stamps, output):
        if np.linalg.norm(target - point) <= distance:
            return stamp - stamps[0]
    return float('nan')


def main():
    rng = np.random.default_rng(20260923)
    center = np.array([2500.0, 720.0])
    print('Hz | fixed jitter RMS px | adaptive jitter RMS px | jitter ratio | fixed step t90 ms | adaptive step t90 ms | t90 reduction')
    for rate in (30, 60, 120):
        stamps = timestamps(rate, 5.0)
        noise = rng.normal(0.0, 5.0, (len(stamps), 2))
        points = center + noise
        fixed = run(lambda: FixedSmoother(.065), stamps, points)
        adaptive = run(Smoother, stamps, points)
        begin = rate
        fixed_jitter = float(np.sqrt(np.mean(np.sum((fixed[begin:] - center) ** 2, axis=1))))
        adaptive_jitter = float(np.sqrt(np.mean(np.sum((adaptive[begin:] - center) ** 2, axis=1))))

        step_stamps = timestamps(rate, .75)
        start = np.array([1000.0, 700.0])
        target = start + np.array([800.0, 0.0])
        step_points = np.asarray([start if stamp < .20 else target for stamp in step_stamps])
        fixed_step = run(lambda: FixedSmoother(.065), step_stamps, step_points)
        adaptive_step = run(Smoother, step_stamps, step_points)
        first = next(index for index, stamp in enumerate(step_stamps) if stamp >= .20)
        response_stamps = step_stamps[first:]
        fixed_ms = 1000 * t90(response_stamps, fixed_step[first:], start, target)
        adaptive_ms = 1000 * t90(response_stamps, adaptive_step[first:], start, target)
        reduction = 100 * (1 - adaptive_ms / fixed_ms)
        print(f'{rate:>2} | {fixed_jitter:>18.3f} | {adaptive_jitter:>21.3f} | '
              f'{adaptive_jitter/fixed_jitter:>11.3f} | {fixed_ms:>16.1f} | '
              f'{adaptive_ms:>19.1f} | {reduction:>13.1f}%')


if __name__ == '__main__':
    main()
