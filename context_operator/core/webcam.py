"""Numeric secondary observations and primary-first gaze arbitration.

No camera, Qt or model dependencies. Agreement is a diagnostic, not an accuracy
probability. A second camera never changes a healthy primary's coordinates.
"""
from dataclasses import dataclass
import math

from .contracts import Gaze


@dataclass(frozen=True)
class WebcamObservation:
    seq: int
    captured: float
    features: tuple[float, ...] = ()
    reason: str = ''


@dataclass(frozen=True)
class GazeChoice:
    gaze: Gaze | None
    mode: str
    corroborated: bool | None = None


def fresh(gaze, now):
    try:
        return (isinstance(gaze, Gaze) and gaze.valid
                and all(not isinstance(v,bool) and math.isfinite(v) for v in (gaze.x, gaze.y, gaze.captured, gaze.radius, now))
                and gaze.radius > 0 and 0 <= now - gaze.captured < .20)
    except (TypeError,ValueError):
        return False


class GazeArbitrator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.first = None
        self.last = None
        self.count = 0

    def choose(self, primary, secondary, now):
        primary = primary if fresh(primary, now) else None
        secondary = secondary if fresh(secondary, now) and secondary.source == 'webcam' else None
        if secondary is None:
            self.reset()
        elif self.last is None or secondary.captured > self.last:
            if self.last is None or secondary.captured - self.last > .20:
                self.first = secondary.captured
                self.count = 0
            self.last = secondary.captured
            self.count += 1
        elif secondary.captured < self.last:
            self.reset()
            secondary = None
        if primary is not None:
            agreement = None
            if secondary is not None and abs(primary.captured - secondary.captured) <= .08:
                agreement = math.hypot(primary.x - secondary.x, primary.y - secondary.y) <= max(primary.radius, secondary.radius)
            return GazeChoice(primary, 'iphone', agreement)
        if secondary is not None and self.count >= 3 and self.last - self.first >= .12:
            return GazeChoice(secondary, 'webcam')
        return GazeChoice(None, 'none')
