"""Personal gaze fit using independent fixations as the training observations.

The recorded frames remain available to measure within-fixation instability,
but a long burst cannot outweigh another target during fitting. Candidate
selection uses training targets only; prospective check targets stay separate.
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

import json

from ..foundations.gaze.profile import Profile
from .session import STARTER_PROTOCOL


FEATURE_SETS = {
    'intersections_head': (0, 1, 2, 3, 4, 5, 6),
    'directions_head': (4, 5, 6, 13, 14, 15, 16, 17, 18),
    'look_head': (4, 5, 6, 19, 20, 21),
    'eyes_head': (0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18),
    'all22': tuple(range(22)),
}


class PersonalProfile(Profile):
    """Small ridge models chosen by held-out training locations and postures."""

    MODEL_KIND = 'personal_fixation_v1'
    CANDIDATES = (
        ('linear', 'intersections_head', 10.),
        ('linear', 'directions_head', 10.),
        ('linear', 'look_head', 10.),
        ('linear', 'eyes_head', 10.),
        ('linear', 'all22', 10.),
        ('quadratic', 'intersections_head', 30.),
        ('quadratic', 'directions_head', 30.),
        ('quadratic', 'look_head', 30.),
    )

    def __init__(self, arrays, metadata, settings):
        if settings.get('model_kind') != self.MODEL_KIND:
            raise ValueError('Unknown personal gaze model')
        try:
            kind, feature_set, alpha = settings['model']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid personal gaze model') from exc
        if ((kind, feature_set, alpha) not in self.CANDIDATES or
                arrays['training_x'].shape[1:] != (22,)):
            raise ValueError('Invalid personal gaze model')
        expected = len(FEATURE_SETS[feature_set])
        if kind == 'quadratic':
            expected += expected * (expected + 1) // 2
        if arrays['coef'].shape != (expected, 2):
            raise ValueError('Invalid personal gaze coefficients')
        super().__init__(arrays, metadata, settings)

    @staticmethod
    def _basis(z, kind):
        return (PolynomialFeatures(2, include_bias=False).fit_transform(z)
                if kind == 'quadratic' else z)

    @classmethod
    def _fit_fixations(cls, x, y, spec, bursts):
        kind, feature_set, alpha = spec
        columns = FEATURE_SETS[feature_set]
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), .025)
        z = (x-mean)/scale
        target_mean = y.mean(axis=0)
        target_scale = np.maximum(y.std(axis=0), 1.)
        design = cls._basis(z[:, columns], kind)
        augmented = np.c_[design, np.ones(len(design))]
        covariance = np.zeros((augmented.shape[1], augmented.shape[1]))
        for burst in bursts:
            basis = cls._basis(((burst-mean)/scale)[:, columns], kind)
            deviation = np.c_[basis-basis.mean(axis=0), np.zeros(len(basis))]
            covariance += deviation.T@deviation/len(basis)
        penalty = np.diag(np.r_[np.full(design.shape[1], alpha), 0.])
        coefficients = np.linalg.solve(augmented.T@augmented + penalty + 2.*covariance,
                                       augmented.T@((y-target_mean)/target_scale))
        return {'mean': mean, 'scale': scale, 'coef': coefficients[:-1],
                'intercept': coefficients[-1], 'target_mean': target_mean,
                'target_scale': target_scale, 'support': z,
                'training_x': x, 'training_y': y}

    @classmethod
    def _predict_fixations(cls, x, arrays, spec):
        kind, feature_set, _ = spec
        z = (np.atleast_2d(x)-arrays['mean'])/arrays['scale']
        basis = cls._basis(z[:, FEATURE_SETS[feature_set]], kind)
        return ((basis @ arrays['coef'] + arrays['intercept']) *
                arrays['target_scale'] + arrays['target_mean'])

    @classmethod
    def fit(cls, bursts, targets, metadata, groups=None, postures=None):
        """Fit from one numeric burst per fixation, with location-grouped CV.

        `bursts` has one (frames, 22) array per target/posture. No check burst
        may enter this method. All fixation medians have equal training weight.
        """
        try:
            frames = [np.asarray(burst, dtype=float) for burst in bursts]
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid fixation bursts') from exc
        if (not frames or any(f.ndim != 2 or f.shape[1] != 22 or len(f) < 2 or
                              not np.isfinite(f).all() for f in frames)):
            raise ValueError('Invalid fixation bursts')
        y = np.asarray(targets, dtype=float)
        if y.shape != (len(frames), 2) or not np.isfinite(y).all():
            raise ValueError('Invalid fixation targets')
        x = np.asarray([np.median(f, axis=0) for f in frames])
        if groups is None:
            _, groups = np.unique(y, axis=0, return_inverse=True)
        else:
            groups = np.asarray(groups)
        if postures is None:
            postures = np.zeros(len(frames), dtype=int)
        else:
            postures = np.asarray(postures)
        if (groups.shape != (len(frames),) or
                postures.shape != (len(frames),) or
                len(np.unique(groups)) < 9 or len(np.unique(postures)) < 1):
            raise ValueError('Invalid fixation grouping')
        folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=23).split(x, y, groups))
        scores = []
        distances = []
        with threadpool_limits(limits=2):
            for spec in cls.CANDIDATES:
                errors = np.full(len(frames), np.nan)
                jitters = np.full(len(frames), np.nan)
                for train, test in folds:
                    arrays = cls._fit_fixations(x[train], y[train], spec, [frames[i] for i in train])
                    for i in test:
                        predictions = cls._predict_fixations(frames[i], arrays, spec)
                        center = np.median(predictions, axis=0)
                        errors[i] = np.linalg.norm(center-y[i])
                        jitters[i] = np.percentile(np.linalg.norm(predictions-center, axis=1), 90)
                        if spec == cls.CANDIDATES[0]:
                            z = (x[i]-arrays['mean'])/arrays['scale']
                            distances.append(float(np.sqrt(np.mean((arrays['support']-z)**2, axis=1)).min()))
                posture_p90 = [float(np.percentile(errors[postures == p], 90))
                               for p in np.unique(postures)]
                median = float(np.median(errors))
                p90 = float(np.percentile(errors, 90))
                jitter_p90 = float(np.percentile(jitters, 90))
                worst_posture_p90 = max(posture_p90)
                score = median + .35*p90 + .35*worst_posture_p90 + .2*jitter_p90
                scores.append({'model': list(spec), 'median_px': median,
                               'p90_px': p90, 'posture_p90_px': posture_p90,
                               'worst_posture_p90_px': worst_posture_p90,
                               'jitter_p90_px': jitter_p90, 'score': score})
            best = min(scores, key=lambda result: result['score'])
            arrays = cls._fit_fixations(x, y, best['model'], frames)
        settings = {'version': 2, 'model_kind': cls.MODEL_KIND,
                    'model': best['model'], 'cv_by_target': scores,
                    'support_limit': max(.8, float(np.percentile(distances, 95))*1.3)}
        return cls(arrays, metadata, settings)

    def predict_base(self, features):
        predicted = self._predict_fixations(features, self.arrays, self.settings['model'])
        return predicted[0] if np.asarray(features).ndim == 1 else predicted


class StarterProfile(Profile):
    """Small one-pose affine map from camera-plane eye-ray intersections.

    It is a coarse first-use model. Its nine fixations are intentionally too
    few to fit a high-dimensional predictor or establish full head-pose coverage.
    """

    MODEL_KIND = 'starter_affine_v1'
    INPUT_COLUMNS = (0, 1, 2, 3)
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
            pose_center = np.asarray(settings['pose_center'], dtype=float)
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
                     settings.get('support_policy', 'median_pose_v1') in ('median_pose_v1', self.SUPPORT_POLICY) and
                     input_mean.shape == (2,) and input_scale.shape == (2,) and
                     pose_center.shape == (9,) and pose_tolerance.shape == (3,) and
                     np.isfinite(np.r_[input_mean, input_scale, pose_center, pose_tolerance,
                                       rotation_tolerance, support_limit]).all() and
                     np.all(input_scale > 0) and np.all(pose_tolerance > 0) and
                     rotation_tolerance > 0 and support_limit > 0)
        except (KeyError, TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('Invalid starter gaze model')
        super().__init__(arrays, metadata, settings)

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
        pose_center = np.median(medians[:, 4:13], axis=0)
        settings = {'version': 2, 'model_kind': cls.MODEL_KIND,
                    'support_policy': cls.SUPPORT_POLICY,
                    'support_columns': list(cls.SUPPORT_COLUMNS),
                    'input_mean': input_mean.tolist(), 'input_scale': input_scale.tolist(),
                    'pose_center': pose_center.tolist(),
                    'pose_tolerance_m': [.05, .05, .07], 'rotation_tolerance': .35,
                    'support_limit': support_limit, 'ridge_alpha': cls.RIDGE_ALPHA,
                    'noise_penalty': cls.NOISE_PENALTY}
        return cls(arrays, metadata, settings)

    def predict_base(self, features):
        x = np.asarray(features, dtype=float)
        plane = self._intersections(x)
        z = (plane-self.settings['input_mean'])/self.settings['input_scale']
        design = np.c_[z, np.ones(len(z))]
        predicted = (design@np.vstack((self.arrays['coef'], self.arrays['intercept'])))*self.arrays['target_scale'] + self.arrays['target_mean']
        return predicted[0] if x.ndim == 1 else predicted

    def supports(self, features):
        return not self.support_reason(features)

    def support_reason(self, features):
        """Compare head and eyes with the same observed training neighborhood.

        A comfortable seat includes looking toward the sides of a wide screen.
        A fixed envelope around the median head rotation wrongly rejected those
        very training poses. Keep the tolerances, but anchor them in the poses
        actually collected; never enlarge them using the held-out check.
        """
        x = np.asarray(features, dtype=float)
        if x.shape != (22,) or not np.isfinite(x).all():
            return 'invalid_features'
        z = (x-self.arrays['mean'])/self.arrays['scale']
        columns = np.asarray(self.SUPPORT_COLUMNS, dtype=int)
        eyes = np.sqrt(np.mean((self.arrays['support'][:, columns]-z[columns])**2, axis=1)) <= self.settings['support_limit']
        if self.settings.get('support_policy', 'median_pose_v1') == 'median_pose_v1':
            poses = np.asarray([self.settings['pose_center']], dtype=float)
        else:
            poses = self.training_x[:, 4:13]
        translation = np.all(np.abs(poses[:, :3]-x[4:7]) <= self.settings['pose_tolerance_m'], axis=1)
        if not translation.any():return 'head_position'
        rotation = np.linalg.norm(poses[:, 3:]-x[7:13], axis=1) <= self.settings['rotation_tolerance']
        if not (translation & rotation).any():return 'head_rotation'
        if not eyes.any():return 'eye_direction'
        if len(poses)>1 and not (translation & rotation & eyes).any():return 'head_eye_combination'
        return ''

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
