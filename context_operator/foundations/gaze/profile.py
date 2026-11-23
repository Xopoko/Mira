from pathlib import Path
import json, math
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
FEATURE_SCALE_FLOOR = np.array([.035,.025,.035,.025]+[.045]*6+[.025,.025,.06,.03,.03,.08])

class Profile:
    """Small regularized model selected by holding out whole screen locations.

    Repeated frames/poses of the same target never cross the validation split.
    Files contain arrays and JSON only, including compact numerical support data.
    """
    CANDIDATES = [('linear', 10., 0.), ('quadratic', 30., 0.), ('quadratic', 150., 0.),
                  ('rbf', .05, .04), ('rbf', .3, .04), ('rbf', .1, .12)]

    def __init__(self, arrays, metadata, settings):
        self.arrays, self.metadata, self.settings = arrays, metadata, settings
        self.training_x, self.training_y = arrays['training_x'], arrays['training_y']
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Invalid calibration profile')
        if self.training_x.ndim != 2 or self.training_y.shape != (len(self.training_x), 2):
            raise ValueError('Invalid support data')
        if np.any(arrays['scale'] <= 0) or np.any(arrays['target_scale'] <= 0):
            raise ValueError('Invalid model scaling')
        self.rebuild_correction()

    @staticmethod
    def design(z, kind):
        return PolynomialFeatures(2, include_bias=False).fit_transform(z) if kind == 'quadratic' else z

    @classmethod
    def _fit(cls, x, y, spec):
        kind, alpha, gamma = spec
        scaler = StandardScaler().fit(x)
        floor = (FEATURE_SCALE_FLOOR if x.shape[1] == len(FEATURE_SCALE_FLOOR) else
                 np.full(x.shape[1], max(.001, float(np.median(scaler.scale_))*.05)))
        scale = np.maximum(scaler.scale_, floor)
        z = (x-scaler.mean_)/scale
        target_mean, target_scale = y.mean(axis=0), np.maximum(y.std(axis=0), 1.)
        normalized = (y-target_mean)/target_scale
        _, ids, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
        weights = 1/counts[ids]
        weights /= weights.mean()
        if kind == 'rbf':
            model = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf').fit(z, normalized, sample_weight=weights)
            coef, intercept = model.dual_coef_, np.zeros(2)
        else:
            model = Ridge(alpha=alpha).fit(cls.design(z, kind), normalized, sample_weight=weights)
            coef, intercept = model.coef_.T, model.intercept_
        return dict(mean=scaler.mean_, scale=scale, coef=coef, intercept=intercept,
                    target_mean=target_mean, target_scale=target_scale, support=z,
                    training_x=x, training_y=y)

    @classmethod
    def _predict(cls, x, arrays, spec):
        z = (np.atleast_2d(x)-arrays['mean'])/arrays['scale']
        basis = rbf_kernel(z, arrays['support'], gamma=spec[2]) if spec[0] == 'rbf' else cls.design(z, spec[0])
        return (basis @ arrays['coef'] + arrays['intercept'])*arrays['target_scale'] + arrays['target_mean']

    @classmethod
    def fit(cls, features, targets, metadata, groups=None):
        x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid training samples')
        if groups is None:
            _, groups = np.unique(y, axis=0, return_inverse=True)
        else:
            groups = np.asarray(groups)
            if groups.shape != (len(x),):
                raise ValueError('Invalid calibration groups')
        if len(np.unique(groups)) < 9:
            raise ValueError('At least nine distinct targets are required')
        folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=23).split(x, y, groups))
        scores = []
        support_distances = []
        with threadpool_limits(limits=2):
            candidates = cls.CANDIDATES if x.shape[1] <= 32 else [
                ('linear', .1, 0.), ('linear', 1., 0.), ('linear', 10., 0.),
                ('linear', 100., 0.), ('rbf', .02, .005), ('rbf', .1, .005)]
            for spec in candidates:
                target_errors = []
                for train, test in folds:
                    fitted = cls._fit(x[train], y[train], spec)
                    pred = cls._predict(x[test], fitted, spec)
                    errors = np.linalg.norm(pred-y[test], axis=1)
                    target_errors.extend(float(np.median(errors[groups[test] == g])) for g in np.unique(groups[test]))
                    if spec == candidates[0]:
                        z = (x[test]-fitted['mean'])/fitted['scale']
                        distances = np.sqrt(np.mean((z[:, None]-fitted['support'][None])**2, axis=2)).min(axis=1)
                        support_distances.extend(distances.tolist())
                median, p90 = float(np.median(target_errors)), float(np.percentile(target_errors, 90))
                scores.append({'model': list(spec), 'median_px': median, 'p90_px': p90,
                               'score': median+.25*p90})
            best = min(scores, key=lambda s: s['score'])
            arrays = cls._fit(x, y, best['model'])
        settings = {'version': 2, 'model': best['model'], 'cv_by_target': scores,
                    'support_limit': max(1.2, float(np.percentile(support_distances, 95))*1.3)}
        return cls(arrays, metadata, settings)

    def predict_base(self, features):
        predicted = self._predict(features, self.arrays, self.settings['model'])
        return predicted[0] if np.asarray(features).ndim == 1 else predicted

    def predict(self, features):
        predicted = self.predict_base(features)
        if self.correction is not None:
            predicted = self.correction.adjust(features,predicted)
        return np.asarray(predicted).reshape(-1,2)[0] if np.asarray(features).ndim == 1 else predicted

    @property
    def click_count(self):
        return len(self.arrays.get('click_targets',[]))

    def rebuild_correction(self):
        self.correction = None
        if not self.click_count:
            return
        from .fixation import CorrectionMap
        bursts,targets = self.arrays['click_features'],self.arrays['click_targets']
        if bursts.shape != (len(targets),8,len(self.arrays['mean'])) or targets.shape != (len(bursts),2):
            raise ValueError('Invalid click correction data')
        representatives = np.median(bursts,axis=1)
        bounds = self.metadata.get('display',[0,0,5120,1440])
        self.correction = CorrectionMap(representatives,self.predict_base(representatives),targets,bounds)

    def add_fixation(self, burst, target):
        burst,target = np.asarray(burst),np.asarray(target)
        if burst.shape != (8,len(self.arrays['mean'])) or target.shape != (2,) or not np.isfinite(burst).all() or not np.isfinite(target).all():
            raise ValueError('Invalid fixation')
        predictions = self.predict(burst)
        center = np.median(predictions,axis=0)
        report = {'before_error_px':float(np.linalg.norm(center-target)),
                  'jitter_px':float(np.median(np.linalg.norm(predictions-center,axis=1))),
                  'base_error_px':float(np.linalg.norm(np.median(self.predict_base(burst),axis=0)-target))}
        if report['jitter_px'] > 150:
            return dict(report,accepted=False)
        self.arrays['click_features'] = np.concatenate([self.arrays.get('click_features',np.empty((0,8,len(self.arrays['mean'])))),burst[None]])[-100:]
        self.arrays['click_targets'] = np.concatenate([self.arrays.get('click_targets',np.empty((0,2))),target[None]])[-100:]
        self.arrays['click_errors'] = np.concatenate([self.arrays.get('click_errors',np.empty((0,3))),
                                                     [[report['before_error_px'],report['jitter_px'],report['base_error_px']]]])[-100:]
        self.rebuild_correction()
        return dict(report,accepted=True)

    def undo_fixation(self):
        if not self.click_count:
            return False
        for key in ('click_features','click_targets','click_errors'):
            self.arrays[key] = self.arrays[key][:-1]
        self.rebuild_correction()
        return True

    def supports(self, features):
        x = np.asarray(features)
        if x.shape != self.arrays['mean'].shape or not np.isfinite(x).all():
            return False
        z = (x-self.arrays['mean'])/self.arrays['scale']
        distance = np.sqrt(np.mean((self.arrays['support']-z)**2, axis=1)).min()
        return bool(distance <= self.settings['support_limit'])

    def save(self, path: Path):
        # Numeric arrays only. Never unpickle calibration files.
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as out:
            np.savez_compressed(out, **self.arrays, metadata=json.dumps(self.metadata),
                                settings=json.dumps(self.settings))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, metadata):
        with np.load(path, allow_pickle=False) as saved:
            actual = json.loads(str(saved['metadata']))
            if actual != metadata:
                raise ValueError('Camera or display changed; calibrate again')
            settings = json.loads(str(saved['settings']))
            if settings.get('version') != 2:
                raise ValueError('Old profile format; recalibrate')
            names = ['mean', 'scale', 'coef', 'intercept', 'target_mean', 'target_scale',
                     'support', 'training_x', 'training_y']
            names += [key for key in ('click_features','click_targets','click_errors') if key in saved.files]
            return cls({n: saved[n] for n in names}, actual, settings)

class Smoother:
    def __init__(self, tau=0.10):
        self.tau = tau
        self.reset()

    def reset(self):
        self.point = None
        self.last = None

    def update(self, point, now):
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            self.reset()
            return None
        if self.last is None or now - self.last > 0.25:
            self.point = p
        else:
            # No three-frame median: at 10 FPS it delayed each genuine gaze shift.
            alpha = 1 - math.exp(-max(0, now - self.last) / self.tau)
            self.point = self.point + alpha * (p - self.point)
        self.last = now
        return tuple(self.point)
