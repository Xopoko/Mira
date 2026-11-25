"""Gaze profile from one setup: where the screen is and how this person's eyes read."""

import json
from pathlib import Path

import numpy as np

from . import geometry
from .session import STARTER_PROTOCOL


def _kernel(a, b, reach):
    return np.exp(-((a[:, None]-b[None])**2).sum(axis=2)/(2*reach**2))


def leftover_field(landed, targets, reach):
    """Weights of a smooth field through what the plane leaves over at each setup point."""
    # Equal signal and noise: about half of a lone fixation's error is
    # taken as real, the rest as that fixation's own scatter.
    return np.linalg.solve(_kernel(landed, landed, reach)+np.eye(len(landed)), targets-landed)


def add_field(pixels, landed, weights, reach):
    """Move the predictions that hit the screen by the field around them."""
    hit = np.isfinite(pixels).all(axis=1)
    pixels[hit] += _kernel(pixels[hit], landed, reach)@weights
    return pixels


class StarterProfile:
    """Screen plane and eye correction fitted from the setup fixations.

    Head pose and eye rotation stay separate, so a point looked at by turning
    the head and the same point looked at with the eyes alone land on the same
    pixel, and a moderate change of seat keeps the mapping.

    What the plane leaves over drifts slowly across the screen and repeats
    between setups, so a smooth field through the leftover error at each setup
    point is added on top. It fades to nothing away from those points.
    """

    MODEL_KIND = 'screen_geometry_v1'
    VERSION = 4
    MIN_TARGETS = 6
    POSITION_TOLERANCE_M = .2
    # One seat cannot tell a large far screen from a small near one, so a
    # fitted size is trusted only close to that seat.
    FITTED_SIZE_TOLERANCE_M = .08
    ROTATION_TOLERANCE = .6
    # Reach of one fixation's leftover error, as a share of the screen height.
    CORRECTION_REACH = .5
    ARRAYS = ('mean', 'scale', 'training_x', 'training_y', 'frame', 'center', 'eye_offsets', 'gains',
              'landed', 'correction')

    def __init__(self, arrays, metadata, settings):
        try:
            training_x = np.asarray(arrays['training_x'], dtype=float)
            training_y = np.asarray(arrays['training_y'], dtype=float)
            frame = np.asarray(arrays['frame'], dtype=float)
            pixel_m = float(settings['pixel_m'])
            tolerances = [float(settings['position_tolerance_m']), float(settings['rotation_tolerance'])]
            reach = float(settings['correction_reach_px'])
            valid = (settings.get('model_kind') == self.MODEL_KIND and
                     metadata.get('protocol') == STARTER_PROTOCOL and metadata.get('features') == 22 and
                     arrays['mean'].shape == (22,) and arrays['scale'].shape == (22,) and
                     training_x.ndim == 2 and training_x.shape[1:] == (22,) and
                     len(training_x) >= self.MIN_TARGETS and training_y.shape == (len(training_x), 2) and
                     frame.shape == (3, 3) and arrays['center'].shape == (3,) and
                     arrays['eye_offsets'].shape == (2, 3) and arrays['gains'].shape == (2,) and
                     arrays['landed'].shape == training_y.shape and
                     arrays['correction'].shape == training_y.shape and
                     all(np.isfinite(arrays[name]).all() for name in self.ARRAYS) and
                     np.allclose(frame.T@frame, np.eye(3), atol=1e-6) and
                     np.all(arrays['scale'] > 0) and np.all(arrays['gains'] > 0) and
                     np.isfinite([pixel_m, reach, *tolerances]).all() and
                     min(pixel_m, reach, *tolerances) > 0)
        except (KeyError, TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('Invalid gaze profile')
        self.arrays, self.metadata = arrays, metadata
        self.settings = dict(settings, pixel_m=pixel_m, position_tolerance_m=tolerances[0],
                             rotation_tolerance=tolerances[1], correction_reach_px=reach)
        self.training_x, self.training_y = training_x, training_y
        self.display = np.asarray(metadata['display'], dtype=float)

    @classmethod
    def fit(cls, bursts, origins, targets, metadata, pixel_mm=None):
        try:
            frames = [np.asarray(burst, dtype=float) for burst in bursts]
            eyes = [np.asarray(burst, dtype=float) for burst in origins]
            y = np.asarray(targets, dtype=float)
            display = np.asarray(metadata['display'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid setup observations') from exc
        if (metadata.get('protocol') != STARTER_PROTOCOL or len(frames) < cls.MIN_TARGETS or
                len(eyes) != len(frames) or y.shape != (len(frames), 2) or not np.isfinite(y).all() or
                display.shape != (4,) or
                any(f.ndim != 2 or f.shape[1] != 22 or len(f) < 2 or not np.isfinite(f).all() or
                    e.shape != (len(f), 2, 3) or not np.isfinite(e).all()
                    for f, e in zip(frames, eyes))):
            raise ValueError('Gaze setup needs steady fixations on every point')
        screen = geometry.fit(frames, eyes, y, display, pixel_mm)
        landed = np.asarray([np.nanmedian(geometry.project(
            f, display, screen['frame'], screen['center'], screen['pixel_m'],
            screen['eye_offsets'], screen['gains']), axis=0) for f in frames])
        reach = cls.CORRECTION_REACH*display[3]
        correction = leftover_field(landed, y, reach)
        medians = np.asarray([np.median(f, axis=0) for f in frames])
        floors = np.full(22, .005)
        floors[4:7] = .02
        floors[7:13] = .05
        floors[13:19] = .08
        arrays = {'mean': medians.mean(axis=0), 'scale': np.maximum(medians.std(axis=0), floors),
                  'training_x': medians, 'training_y': y, 'frame': screen['frame'],
                  'center': screen['center'], 'eye_offsets': screen['eye_offsets'],
                  'gains': np.asarray(screen['gains'], dtype=float),
                  'landed': landed, 'correction': correction}
        settings = {'version': cls.VERSION, 'model_kind': cls.MODEL_KIND,
                    'pixel_m': screen['pixel_m'], 'screen_size_known': screen['known_size'],
                    'viewing_m': screen['viewing_m'], 'fit_error_px': screen['fit_error_px'],
                    'correction_reach_px': float(reach),
                    'position_tolerance_m': (cls.POSITION_TOLERANCE_M if screen['known_size']
                                             else cls.FITTED_SIZE_TOLERANCE_M),
                    'rotation_tolerance': cls.ROTATION_TOLERANCE}
        return cls(arrays, metadata, settings)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        predicted = add_field(
            geometry.project(x, self.display, self.arrays['frame'], self.arrays['center'],
                             self.settings['pixel_m'], self.arrays['eye_offsets'], self.arrays['gains']),
            self.arrays['landed'], self.arrays['correction'], self.settings['correction_reach_px'])
        return predicted[0] if x.ndim == 1 else predicted

    def supports(self, features):
        return not self.support_reason(features)

    def support_reason(self, features):
        """Why a reading is too far from the seat used in setup, or ''."""
        x = np.asarray(features, dtype=float)
        if x.shape != (22,) or not np.isfinite(x).all():
            return 'invalid_features'
        poses = self.training_x[:, 4:13]
        near = np.linalg.norm(poses[:, :3]-x[4:7], axis=1) <= self.settings['position_tolerance_m']
        if not near.any():
            return 'head_position'
        # The turn has to match a fixation taken from about this position.
        if np.linalg.norm(poses[near, 3:]-x[7:13], axis=1).min() > self.settings['rotation_tolerance']:
            return 'head_rotation'
        return ''

    def save(self, path: Path):
        # Numeric arrays only. Never unpickle calibration files.
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as out:
            np.savez_compressed(out, **self.arrays, metadata=json.dumps(self.metadata),
                                settings=json.dumps(self.settings))
        temporary.replace(path)

    @classmethod
    def load(cls, path, metadata):
        with np.load(path, allow_pickle=False) as saved:
            actual = json.loads(str(saved['metadata']))
            settings = json.loads(str(saved['settings']))
            if actual != metadata or settings.get('version') != cls.VERSION:
                raise ValueError('Incompatible gaze profile')
            arrays = {name: saved[name] for name in cls.ARRAYS}
        return cls(arrays, actual, settings)
