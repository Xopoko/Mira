"""Small webcam mappings selected only by leave-one-target-out training error."""
from dataclasses import dataclass, asdict
import math

import numpy as np


KINDS = {'iris_affine': 3, 'iris_quadratic': 6, 'iris_pose': 9, 'iris_pose_quadratic': 12}
SCALE_FLOOR = np.asarray((.018, .018, .020, .020, .020, .006, .006, .006))
RIDGES = (.001, .01, .1, 1., 10.)


@dataclass(frozen=True)
class WebcamMapping:
    kind: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    regularization: float
    cv_error_px: float

    @property
    def columns(self):
        return KINDS[self.kind]

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'kind', 'center', 'scale', 'regularization', 'cv_error_px'}:
            raise ValueError('Invalid webcam mapping fields')
        if value['kind'] not in KINDS:
            raise ValueError('Unknown webcam mapping')
        center, scale = np.asarray(value['center'], dtype=float), np.asarray(value['scale'], dtype=float)
        alpha, error = float(value['regularization']), float(value['cv_error_px'])
        if (center.shape != (8,) or scale.shape != (8,) or not np.isfinite(center).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)
                or not math.isfinite(alpha) or alpha < 0 or not math.isfinite(error) or error < 0):
            raise ValueError('Invalid webcam mapping parameters')
        return cls(value['kind'], tuple(center), tuple(scale), alpha, error)

    def design(self, features):
        values = (np.atleast_2d(features) - np.asarray(self.center)) / np.asarray(self.scale)
        iris = values[:, :2]
        columns = [iris]
        if self.kind in ('iris_pose', 'iris_pose_quadratic'):
            columns.append(values[:, 2:])
        if self.kind in ('iris_quadratic', 'iris_pose_quadratic'):
            x, y = iris[:, 0], iris[:, 1]
            columns.append(np.column_stack((x*x, x*y, y*y)))
        columns.append(np.ones((len(values), 1)))
        return np.column_stack(columns)


def _fit(features, pixels, kind, alpha, groups=None):
    sample_weights = np.ones(len(features))
    if groups is not None and kind in ('iris_pose', 'iris_pose_quadratic'):
        sample_weights = np.concatenate([np.full(len(group), 1./len(group)) for group in groups])
        pixels = np.concatenate([np.repeat([target], len(group), axis=0) for target, group in zip(pixels, groups)])
        features = np.concatenate(groups)
    if kind == 'iris_affine':
        mapping = WebcamMapping(kind, (0.,)*8, (1.,)*8, 0., 0.)
    else:
        center = np.average(features, axis=0, weights=sample_weights)
        scale = np.maximum(np.sqrt(np.average((features-center)**2, axis=0, weights=sample_weights)), SCALE_FLOOR)
        mapping = WebcamMapping(kind, tuple(center), tuple(scale), float(alpha), 0.)
    design = mapping.design(features)
    design = design * np.sqrt(sample_weights)[:, None]
    pixels = pixels * np.sqrt(sample_weights)[:, None]
    if kind == 'iris_affine':
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError('Training iris offsets do not span an affine fit')
        weights = np.linalg.lstsq(design, pixels, rcond=None)[0]
    else:
        penalty = np.eye(design.shape[1]) * alpha
        penalty[-1, -1] = 0.  # The intercept is not regularized.
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ pixels)
    return weights, mapping


def _cross_validation(features, pixels, kind, alpha, groups=None):
    squared_errors = []
    for index in range(len(features)):
        keep = np.arange(len(features)) != index
        # Fold-specific normalization avoids letting the omitted target shape its fit.
        try:
            fold_groups = [group for i, group in enumerate(groups) if i != index] if groups is not None else None
            weights, mapping = _fit(features[keep], pixels[keep], kind, alpha, fold_groups)
        except (ValueError, np.linalg.LinAlgError):
            return math.inf
        heldout = groups[index] if groups is not None else features[index:index+1]
        prediction = mapping.design(heldout) @ weights
        squared_errors.append(float(np.mean(np.sum((prediction - pixels[index])**2, axis=1))))
    return math.sqrt(float(np.mean(squared_errors)))


def fit_mapping(features, pixels, groups=None):
    """Freeze the family/regularization before receiving any held-out observations.

    The original affine fit wins ties and small improvements. Extra capacity is
    admitted only for a >=20% and >=15 physical-pixel cross-validation improvement.
    The independent five-point acceptance remains a separate decision.
    """
    features, pixels = np.asarray(features, dtype=float), np.asarray(pixels, dtype=float)
    baseline_error = _cross_validation(features, pixels, 'iris_affine', 0., groups)
    if not math.isfinite(baseline_error):
        raise ValueError('Training gaze is not diverse enough across screen targets')
    selected = ('iris_affine', 0., baseline_error)
    best = selected
    # Head terms need within-fixation measurements, not a saturated fit to nine means.
    pose_varies = False
    if groups is not None:
        variation = np.asarray([np.std(group, axis=0)[2:] / SCALE_FLOOR[2:] for group in groups])
        pose_varies = np.count_nonzero(np.any(variation >= .25, axis=1)) >= 3
    kinds = ('iris_quadratic', 'iris_pose', 'iris_pose_quadratic') if pose_varies else ('iris_quadratic',)
    for kind in kinds:
        for alpha in RIDGES:
            error = _cross_validation(features, pixels, kind, alpha, groups)
            if error < best[2]:
                best = (kind, alpha, error)
    if best[2] < .8*baseline_error and best[2] < baseline_error-15.:
        selected = best
    weights, mapping = _fit(features, pixels, selected[0], selected[1], groups)
    if not np.isfinite(weights).all():
        raise ValueError('Webcam mapping produced non-finite weights')
    return weights, WebcamMapping(mapping.kind, mapping.center, mapping.scale,
                                  mapping.regularization, selected[2])
