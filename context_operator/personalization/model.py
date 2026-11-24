"""Affine screen mapping fitted from nine independent TrueDepth fixations."""

import json
from pathlib import Path

import numpy as np

from .session import STARTER_PROTOCOL


class StarterProfile:
    """Small one-pose affine map from camera-plane eye-ray intersections.

    It is a coarse first-use model. Its nine fixations are intentionally too
    few to fit a high-dimensional predictor or establish full head-pose coverage.
    """

    MODEL_KIND = 'starter_affine_v1'
    SUPPORT_COLUMNS = tuple(range(13, 19))
    RIDGE_ALPHA = .25
    NOISE_PENALTY = 2.
    SUPPORT_POLICY = 'joint_training_poses_v2'

    def __init__(self, arrays, metadata, settings):
        if settings.get('model_kind') != self.MODEL_KIND:
            raise ValueError('Unknown starter gaze model')
        try:
            input_mean = np.asarray(settings['input_mean'], dtype=float)
            input_scale = np.asarray(settings['input_scale'], dtype=float)
            pose_tolerance = np.asarray(settings['pose_tolerance_m'], dtype=float)
            rotation_tolerance = float(settings['rotation_tolerance'])
            support_limit = float(settings['support_limit'])
            training_x = np.asarray(arrays['training_x'])
            training_y = np.asarray(arrays['training_y'])
            valid = (metadata.get('protocol') == STARTER_PROTOCOL and
                     metadata.get('features') == 22 and
                     arrays['mean'].shape == (22,) and arrays['scale'].shape == (22,) and
                     training_x.ndim == 2 and training_x.shape[1:] == (22,) and
                     training_y.shape == (len(training_x), 2) and len(training_x) == 9 and
                     arrays['coef'].shape == (2, 2) and arrays['intercept'].shape == (2,) and
                     arrays['support'].shape == (len(training_x), 22) and
                     arrays['target_mean'].shape == (2,) and arrays['target_scale'].shape == (2,) and
                     settings.get('support_columns') == list(self.SUPPORT_COLUMNS) and
                     settings.get('support_policy') == self.SUPPORT_POLICY and
                     input_mean.shape == (2,) and input_scale.shape == (2,) and
                     pose_tolerance.shape == (3,) and
                     np.isfinite(np.r_[input_mean, input_scale, pose_tolerance,
                                       rotation_tolerance, support_limit]).all() and
                     np.all(input_scale > 0) and np.all(pose_tolerance > 0) and
                     rotation_tolerance > 0 and support_limit > 0)
        except (KeyError, TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('Invalid starter gaze model')
        self.arrays, self.metadata, self.settings = arrays, metadata, settings
        self.training_x, self.training_y = training_x, training_y
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Invalid calibration profile')
        if np.any(arrays['scale'] <= 0) or np.any(arrays['target_scale'] <= 0):
            raise ValueError('Invalid model scaling')

    @staticmethod
    def _intersections(features):
        x = np.atleast_2d(np.asarray(features, dtype=float))
        # The four values are left and right eye-ray intersections with the
        # camera plane; average both eyes before mapping to display pixels.
        return (x[:, :2] + x[:, 2:4]) * .5

    @classmethod
    def fit(cls, bursts, targets, metadata):
        try:
            frames = [np.asarray(burst, dtype=float) for burst in bursts]
            y = np.asarray(targets, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid starter observations') from exc
        if (metadata.get('protocol') != STARTER_PROTOCOL or len(frames) != 9 or
                y.shape != (9, 2) or not np.isfinite(y).all() or
                any(f.ndim != 2 or f.shape[1] != 22 or len(f) < 2 or not np.isfinite(f).all()
                    for f in frames)):
            raise ValueError('Starter model needs nine finite fixation bursts')
        medians = np.asarray([np.median(f, axis=0) for f in frames])
        plane = cls._intersections(medians)
        mean = medians.mean(axis=0)
        floors = np.full(22, .005)
        floors[4:7] = .02
        floors[7:13] = .05
        floors[13:19] = .08
        scale = np.maximum(medians.std(axis=0), floors)
        z = (medians-mean)/scale
        input_mean = plane.mean(axis=0)
        input_scale = np.maximum(plane.std(axis=0), .003)
        design = np.c_[(plane-input_mean)/input_scale, np.ones(len(plane))]
        target_mean = y.mean(axis=0)
        target_scale = np.maximum(y.std(axis=0), 1.)
        normalized_y = (y-target_mean)/target_scale

        # Penalize unstable rays within each fixation without giving long
        # captures extra weight. Every accepted fixation contributes once.
        covariance = np.zeros((3, 3))
        for burst in frames:
            rays = cls._intersections(burst)
            ray_z = (rays-input_mean)/input_scale
            deviation = np.c_[ray_z-ray_z.mean(axis=0), np.zeros(len(ray_z))]
            covariance += deviation.T@deviation/max(len(ray_z), 1)
        penalty = np.diag([cls.RIDGE_ALPHA, cls.RIDGE_ALPHA, 0.])
        coef = np.linalg.solve(design.T@design + penalty + cls.NOISE_PENALTY*covariance,
                               design.T@normalized_y)

        support_z = z[:, cls.SUPPORT_COLUMNS]
        distances = np.sqrt(np.mean((support_z[:, None]-support_z[None])**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        support_limit = max(.35, float(np.percentile(nearest, 90))*1.25)
        arrays = {'mean': mean, 'scale': scale, 'coef': coef[:2], 'intercept': coef[2],
                  'target_mean': target_mean, 'target_scale': target_scale,
                  'support': z, 'training_x': medians, 'training_y': y}
        settings = {'version': 2, 'model_kind': cls.MODEL_KIND,
                    'support_policy': cls.SUPPORT_POLICY,
                    'support_columns': list(cls.SUPPORT_COLUMNS),
                    'input_mean': input_mean.tolist(), 'input_scale': input_scale.tolist(),
                    'pose_tolerance_m': [.05, .05, .07], 'rotation_tolerance': .35,
                    'support_limit': support_limit, 'ridge_alpha': cls.RIDGE_ALPHA,
                    'noise_penalty': cls.NOISE_PENALTY}
        return cls(arrays, metadata, settings)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        plane = self._intersections(x)
        z = (plane-self.settings['input_mean'])/self.settings['input_scale']
        design = np.c_[z, np.ones(len(z))]
        predicted = (design@np.vstack((self.arrays['coef'], self.arrays['intercept'])))*self.arrays['target_scale'] + self.arrays['target_mean']
        return predicted[0] if x.ndim == 1 else predicted

    def supports(self, features):
        return not self.support_reason(features)

    def support_reason(self, features):
        """Compare head and eyes with the same training fixation.

        A comfortable seat includes looking toward the sides of a wide screen,
        so the fixed tolerances are anchored in each collected pose. The
        held-out check never widens them.
        """
        x = np.asarray(features, dtype=float)
        if x.shape != (22,) or not np.isfinite(x).all():
            return 'invalid_features'
        z = (x-self.arrays['mean'])/self.arrays['scale']
        columns = np.asarray(self.SUPPORT_COLUMNS, dtype=int)
        eyes = np.sqrt(np.mean((self.arrays['support'][:, columns]-z[columns])**2, axis=1)) <= self.settings['support_limit']
        poses = self.training_x[:, 4:13]
        translation = np.all(np.abs(poses[:, :3]-x[4:7]) <= self.settings['pose_tolerance_m'], axis=1)
        if not translation.any():return 'head_position'
        rotation = np.linalg.norm(poses[:, 3:]-x[7:13], axis=1) <= self.settings['rotation_tolerance']
        if not (translation & rotation).any():return 'head_rotation'
        if not eyes.any():return 'eye_direction'
        if not (translation & rotation & eyes).any():return 'head_eye_combination'
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
            if actual != metadata or settings.get('version') != 2:
                raise ValueError('Incompatible starter profile')
            arrays = {name: saved[name] for name in
                      ('mean', 'scale', 'coef', 'intercept', 'target_mean', 'target_scale',
                       'support', 'training_x', 'training_y')}
        return cls(arrays, actual, settings)
