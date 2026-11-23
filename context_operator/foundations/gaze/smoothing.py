"""Speed-adaptive low-pass smoothing for physical screen coordinates.

This is a vector form of the 1€ filter. Position and speed are normalized by
screen height so the tuning remains comparable across display resolutions.
"""
import math

import numpy as np


class Smoother:
    """Time-aware adaptive smoother with the legacy Smoother call contract.

    ``point`` and ``last`` remain available to callers. ``scale`` is the
    physical screen height in pixels; filter frequencies are tuned in screen
    heights per second.
    """

    def __init__(self, tau=0.065, *, scale=1440.0, beta=3.0,
                 derivative_cutoff=1.0, max_gap=0.25):
        if not all(math.isfinite(value) and value > 0 for value in
                   (tau, scale, beta, derivative_cutoff, max_gap)):
            raise ValueError("smoothing parameters must be finite and positive")
        self.scale = float(scale)
        self.min_cutoff = 1.0 / (2.0 * math.pi * float(tau))
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.max_gap = float(max_gap)
        self.reset()

    def reset(self):
        self.point = None
        self.last = None
        self._filtered = None
        self._derivative = None
        self._previous = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, point, now):
        p = np.asarray(point, dtype=float)
        try:
            stamp = float(now)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return None
        if p.shape != (2,) or not np.isfinite(p).all() or not math.isfinite(stamp):
            self.reset()
            return None

        normalized = p / self.scale
        if self.last is None:
            self._filtered = normalized.copy()
            self._derivative = np.zeros(2, dtype=float)
            self._previous = normalized.copy()
            self.point = tuple(p)
            self.last = stamp
            return self.point

        dt = stamp - self.last
        if dt <= 0.0:
            # Repeated or out-of-order timestamps cannot move the output or
            # change the baseline used by the next valid sample.
            return self.point
        if dt > self.max_gap:
            self.reset()
            return self.update(p, stamp)

        raw_derivative = (normalized - self._previous) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)
        speed = float(np.linalg.norm(self._derivative))
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)
        self._filtered += alpha * (normalized - self._filtered)
        self._previous = normalized.copy()
        self.point = tuple(self._filtered * self.scale)
        self.last = stamp
        return self.point
