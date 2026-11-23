"""Conservative numeric calibration for the optional webcam gaze source.

Training selects a bounded iris/pose mapping without consulting held-out targets.
All eye/head values retain the fixed joint support guard, so a model cannot
extrapolate into unrelated poses. Its independent check is a usability screen,
never physical accuracy acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.contracts import Rect
from ..core.webcam import WebcamObservation
from .webcam_mapping import WebcamMapping, fit_mapping

LEGACY_WEBCAM_ENGINE_VERSION = 'webcam-iris-affine-support-v1'
WEBCAM_ENGINE_VERSION = 'webcam-iris-pose-mapping-v2'
WEBCAM_PROTOCOL = 'webcam-secondary-calibration-v1'
FEATURE_COUNT = 8
TRAIN_TARGETS = tuple((x, y) for y in (.10, .50, .90) for x in (.10, .50, .90))
CHECK_TARGETS = ((.23, .23), (.77, .23), (.50, .35), (.23, .77), (.77, .77))
# Fixed joint eye/head neighborhood; the last three terms detect camera-relative
# movement and scale changes as well as the iris and nose features.
_SUPPORT_TOLERANCES = (.18, .18, .20, .20, .20, .06, .06, .06)
_FRESH_SECONDS = .20
_MIN_FRAMES = 10
_MIN_DURATION = .65
_WINDOW_SECONDS = 2.40
_MAX_WINDOW_FRAMES = 90
_STABILITY_SPREAD = np.asarray((.06, .06, .06, .06, .08, .06, .06, .06))
_MAX_TRIM_FRACTION = .15


def _display(bounds):
    values = np.asarray(bounds.as_list() if hasattr(bounds, 'as_list') else bounds, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or values[2] <= 0 or values[3] <= 0:
        raise ValueError('Invalid display bounds')
    return tuple(float(v) for v in values)


def _feature_row(features):
    values = np.asarray(features, dtype=float)
    if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
        raise ValueError('Webcam features must contain exactly eight finite numbers')
    return values


def _matrix(rows, columns=FEATURE_COUNT):
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != columns or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f'Expected a non-empty finite matrix with {columns} columns')
    return values


def _target_rows(targets, count, name):
    values = np.asarray(targets, dtype=float)
    if values.shape != (count, 2) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f'{name} must contain {count} fractional screen points')
    return values


def _same_targets(values, expected):
    return values.shape == (len(expected), 2) and np.allclose(values, np.asarray(expected), atol=1e-9, rtol=0)


def _plain(rows):
    return [[float(v) for v in row] for row in rows]


@dataclass(frozen=True)
class WebcamValidationReport:
    usable: bool
    passed: bool
    coverage: tuple[float, ...]
    errors_px: tuple[tuple[float, ...], ...]
    median_px: float
    p90_px: float
    jitter_px: float
    targets: tuple[dict, ...]
    check_rows: tuple[dict, ...]
    limits: dict
    scope: str = 'webcam-secondary-heldout-usability'

    def as_dict(self):
        return {
            'usable': self.usable, 'passed': False, 'scope': self.scope,
            'coverage': list(self.coverage), 'errors_px': [list(row) for row in self.errors_px],
            'median_px': self.median_px, 'p90_px': self.p90_px,
            'jitter_px': self.jitter_px, 'targets': list(self.targets),
            'check_rows': list(self.check_rows), 'limits': dict(self.limits),
        }


@dataclass(frozen=True)
class WebcamProfile:
    bounds: tuple[float, float, float, float]
    device_id: str
    engine_version: str
    coefficients: tuple[tuple[float, float], ...]
    support_anchors: tuple[tuple[float, ...], ...]
    training_rows: tuple[dict, ...]
    validation_p90_px: float = 0.0
    mapping: WebcamMapping | None = None

    @property
    def validation_radius(self):
        return min(650., max(180., float(self.validation_p90_px) * 1.25))

    @classmethod
    def fit(cls, train_features, train_targets, bounds: Rect, device_id: str, *, training_samples=None, legacy=False):
        display = _display(bounds)
        if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 512:
            raise ValueError('A stable webcam device identity is required')
        features = _matrix(train_features)
        targets = _target_rows(train_targets, len(TRAIN_TARGETS), 'Training targets')
        if features.shape != (len(TRAIN_TARGETS), FEATURE_COUNT):
            raise ValueError('Webcam fit requires exactly nine target-averaged training rows')
        if not _same_targets(targets, TRAIN_TARGETS):
            raise ValueError('Training targets must match the nine fixed training points')
        samples = None
        if training_samples is not None:
            if legacy or len(training_samples) != len(TRAIN_TARGETS):
                raise ValueError('Invalid per-target training samples')
            samples = [_matrix(group) for group in training_samples]
            if any(not _MIN_FRAMES <= len(group) <= _MAX_WINDOW_FRAMES for group in samples):
                raise ValueError('Invalid training fixation sample count')
            if not np.allclose([group.mean(axis=0) for group in samples], features, atol=1e-12, rtol=0):
                raise ValueError('Training means differ from their fixation samples')
        design = np.column_stack((features[:, :2], np.ones(len(features))))
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError('Training iris offsets do not span an affine fit')
        pixels = np.column_stack((display[0] + display[2] * targets[:, 0],
                                  display[1] + display[3] * targets[:, 1]))
        if legacy:
            coefficients = np.linalg.lstsq(design, pixels, rcond=None)[0]
            mapping = None
        else:
            coefficients, mapping = fit_mapping(features, pixels, samples)
        if not np.isfinite(coefficients).all():
            raise ValueError('Webcam fit produced a non-finite model')
        train_rows = tuple({'target_fraction': list(map(float, target)), 'features': list(map(float, feature))}
                           for feature, target in zip(features, targets))
        if samples is not None:
            train_rows = tuple({**row, 'samples': _plain(group)} for row, group in zip(train_rows, samples))
        return cls(display, device_id.strip(), LEGACY_WEBCAM_ENGINE_VERSION if legacy else WEBCAM_ENGINE_VERSION,
                   tuple(tuple(float(v) for v in row) for row in coefficients),
                   tuple(tuple(float(v) for v in row) for row in features), train_rows, mapping=mapping)

    def _identity(self, bounds, device_id):
        if self.engine_version not in (WEBCAM_ENGINE_VERSION, LEGACY_WEBCAM_ENGINE_VERSION):
            raise ValueError('Webcam profile engine is incompatible')
        if self.bounds != _display(bounds) or self.device_id != device_id:
            raise ValueError('Webcam profile belongs to another camera or display')

    def supports(self, features):
        try:
            row = _feature_row(features)
        except (TypeError, ValueError):
            return False
        if not self.support_anchors or len(self.support_anchors) != len(TRAIN_TARGETS):
            return False
        anchors = np.asarray(self.support_anchors, dtype=float)
        if anchors.shape != (len(TRAIN_TARGETS), FEATURE_COUNT) or not np.isfinite(anchors).all():
            return False
        delta = (anchors - row) / np.asarray(_SUPPORT_TOLERANCES)
        distance2 = np.sum(delta * delta, axis=1)
        return bool(np.any(np.all(np.abs(delta) <= 1.0, axis=1) &
                           (distance2 <= float(len(_SUPPORT_TOLERANCES)))))

    def predict(self, features):
        row = _feature_row(features)
        if not self.supports(row):
            raise ValueError('Webcam observation is outside the recorded eye/head support')
        return tuple(float(v) for v in self._predictions([row])[0])

    def _predictions(self, features):
        coefficients = np.asarray(self.coefficients, dtype=float)
        columns = self.mapping.columns if self.mapping is not None else 3
        if coefficients.shape != (columns, 2) or not np.isfinite(coefficients).all():
            raise ValueError('Webcam model coefficients are invalid')
        matrix = np.atleast_2d(features)
        design = self.mapping.design(matrix) if self.mapping is not None else np.column_stack((matrix[:, :2], np.ones(len(matrix))))
        return design @ coefficients

    def validate(self, check_features, check_targets, *, invalid_counts=None):
        targets = _target_rows(check_targets, len(CHECK_TARGETS), 'Check targets')
        if not _same_targets(targets, CHECK_TARGETS):
            raise ValueError('Held-out targets must match the five fixed check points')
        if len(check_features) != len(CHECK_TARGETS):
            raise ValueError('Held-out checks require five per-target feature groups')
        invalid_counts = list(invalid_counts or [0] * len(CHECK_TARGETS))
        if len(invalid_counts) != len(CHECK_TARGETS) or any(type(n) is not int or n < 0 for n in invalid_counts):
            raise ValueError('Invalid held-out missing-observation counts')
        rows, all_errors, coverages, all_jitters = [], [], [], []
        display = self.bounds
        for index, (group, target, invalid) in enumerate(zip(check_features, targets, invalid_counts)):
            matrix = _matrix(group)
            if len(matrix) < _MIN_FRAMES:
                raise ValueError('Every held-out target needs at least ten observed frames')
            screen_target = np.asarray((display[0] + display[2] * target[0],
                                        display[1] + display[3] * target[1]))
            predictions = self._predictions(matrix)
            supported = np.asarray([self.supports(row) for row in matrix], dtype=bool)
            onscreen = ((predictions[:, 0] >= display[0]) & (predictions[:, 0] < display[0] + display[2]) &
                        (predictions[:, 1] >= display[1]) & (predictions[:, 1] < display[1] + display[3]))
            covered = supported & onscreen
            coverage = float(np.count_nonzero(covered) / (len(matrix) + invalid))
            errors = np.linalg.norm(predictions - screen_target, axis=1)
            median = float(np.median(errors))
            p90 = float(np.percentile(errors, 90))
            center = np.median(predictions, axis=0)
            jitter = float(np.median(np.linalg.norm(predictions - center, axis=1)))
            all_errors.extend(errors.tolist())
            all_jitters.append(jitter)
            coverages.append(coverage)
            rows.append({'index': index, 'target_fraction': list(map(float, target)),
                         'frames': len(matrix), 'invalid_frames': invalid,
                         'coverage': coverage, 'median_px': median, 'p90_px': p90,
                         'jitter_px': jitter})
        h = display[3]
        limits = {'median_px': min(400., h * .28), 'p90_px': min(700., h * .50),
                  'jitter_px': min(150., h * .11), 'coverage': .80}
        median = float(np.median(all_errors))
        p90 = float(np.percentile(all_errors, 90))
        jitter = float(np.median(all_jitters))
        usable = (all(v >= .80 for v in coverages) and median <= limits['median_px'] and
                  p90 <= limits['p90_px'] and jitter <= limits['jitter_px'] and
                  all(r['p90_px'] <= limits['p90_px'] and r['jitter_px'] <= limits['jitter_px'] for r in rows))
        check_rows = tuple({'target_fraction': r['target_fraction'],
                            'features': _plain(group), 'invalid_frames': int(invalid)}
                           for r, group, invalid in zip(rows, check_features, invalid_counts))
        return WebcamValidationReport(bool(usable), False, tuple(coverages),
            tuple(tuple(float(v) for v in _target_errors) for _target_errors in
                  [[float(v) for v in np.linalg.norm(self._predictions(group) -
                                                     np.asarray((display[0] + display[2] * target[0], display[1] + display[3] * target[1])), axis=1)]
                   for group, target in zip(check_features, targets)]),
            median, p90, jitter, tuple(rows), check_rows, limits)

    def validate_rows(self, check_rows):
        if not isinstance(check_rows, (list, tuple)) or len(check_rows) != len(CHECK_TARGETS):
            raise ValueError('A complete set of five held-out rows is required')
        targets, groups, invalid = [], [], []
        for row, expected in zip(check_rows, CHECK_TARGETS):
            if not isinstance(row, dict) or set(row) != {'target_fraction', 'features', 'invalid_frames'}:
                raise ValueError('Malformed held-out row')
            targets.append(row['target_fraction'])
            groups.append(row['features'])
            invalid.append(row['invalid_frames'])
        return self.validate(groups, targets, invalid_counts=invalid)


class WebcamSession:
    """Nine fit fixations followed by five disjoint, held-out check fixations."""
    train_targets = TRAIN_TARGETS
    check_targets = CHECK_TARGETS

    def __init__(self, bounds: Rect, device_id: str):
        self.bounds = _display(bounds)
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError('A stable webcam device identity is required')
        self.device_id = device_id.strip()
        self.phase = 'train'
        self.index = 0
        self.train = []
        self.check = []
        self.frames = []
        self.invalid_frames = 0
        self.last_seq = -1
        self.last_observation = None
        self.last_confirmed_seq = -1
        self.hold_reason = 'Waiting for a fresh camera frame'
        self.candidate = None
        self.report = None
        self.ready = False
        self._inlier_indices = ()
        self._excluded_indices = ()
        self._motion_detected = False

    @property
    def current_target_fraction(self):
        if self.phase == 'train':
            return self.train_targets[self.index]
        if self.phase == 'check':
            return self.check_targets[self.index]
        return None

    @property
    def target_fraction(self):
        return self.current_target_fraction

    @property
    def complete(self):
        return self.phase == 'result'

    @property
    def sample_count(self):
        """Number of coherent, distinct observations in the current fixation."""
        return len(self._inlier_indices)

    @property
    def excluded_count(self):
        """Raw observations retained for validation but excluded from fitting."""
        return len(self._excluded_indices)

    @property
    def duration(self):
        """Time span covered by coherent observations in the current fixation."""
        if not self._inlier_indices:
            return 0.0
        stamps = [float(self.frames[i][0].captured) for i in self._inlier_indices]
        return max(0.0, max(stamps) - min(stamps))

    @property
    def progress(self):
        """Fixation readiness progress from 0 to 1, without changing its gates."""
        if self.complete:
            return 1.0
        return min(1.0, self.sample_count / _MIN_FRAMES,
                   self.duration / _MIN_DURATION)

    @property
    def collection_state(self):
        if self.ready:
            return 'ready'
        if self._motion_detected:
            return 'moving'
        if self.excluded_count:
            return 'outliers'
        return 'collecting'

    def _analyze_fixation(self):
        """Return robust inliers, excluding at most 15% isolated feature spikes.

        Raw rows stay in ``frames``. That lets held-out validation see tracking
        jitter even when those rows do not influence the training mean.
        """
        if not self.frames:
            return (), (), False
        matrix = np.asarray([row for _, row in self.frames], dtype=float)
        center = np.median(matrix, axis=0)
        scores = np.max(np.abs(matrix - center) / _STABILITY_SPREAD, axis=1)
        outlier_mask = scores > 1.0
        outlier_indices = tuple(int(i) for i in np.flatnonzero(outlier_mask))
        # One candidate spike can be recognized before seven frames, but it
        # cannot make the fixation ready until the final exclusion ratio is
        # within the 15% bound.
        trim_limit = max(1, int(len(matrix) * _MAX_TRIM_FRACTION))
        if len(outlier_indices) > trim_limit:
            return (), outlier_indices, True
        inlier_indices = tuple(int(i) for i in np.flatnonzero(~outlier_mask))
        if not inlier_indices:
            return (), outlier_indices, True
        inliers = matrix[list(inlier_indices)]
        if np.any(np.ptp(inliers, axis=0) > _STABILITY_SPREAD):
            return (), outlier_indices, True
        return inlier_indices, outlier_indices, False

    def _clear(self, reason):
        self.frames.clear()
        # A lost fixation is discarded in full. Coverage is measured over the
        # eventual confirmed fixation, not over time spent arranging the camera.
        # Otherwise unbounded rejected polls divided by a bounded frame window
        # would make recovery impossible even after a long stable signal.
        self.invalid_frames = 0
        self.last_observation = None
        self.ready = False
        self._inlier_indices = ()
        self._excluded_indices = ()
        self._motion_detected = False
        self.hold_reason = reason

    def reset_stream(self):
        """A new explicitly opened stream has a new sequence counter."""
        self._clear('Keep looking at the point')
        self.last_seq = -1
        self.last_confirmed_seq = -1

    def feed(self, obs: WebcamObservation, now: float):
        if self.phase not in ('train', 'check'):
            self._clear('Setup complete')
            return
        if not isinstance(obs, WebcamObservation):
            self.invalid_frames += 1
            self._clear('Invalid camera reading')
            return
        try:
            captured = float(obs.captured)
            current = float(now)
        except (TypeError, ValueError):
            captured, current = math.nan, math.nan
        if (obs.reason or not math.isfinite(captured) or not math.isfinite(current) or
                current < captured):
            self.invalid_frames += 1
            self._clear(obs.reason or 'Waiting for a fresh camera frame')
            return
        if type(obs.seq) is not int:
            self.invalid_frames += 1
            self._clear('Invalid camera frame number')
            return
        if current - captured > _FRESH_SECONDS:
            if (obs.seq == self.last_seq and self.last_observation is not None
                    and captured == self.last_observation.captured):
                # Polling an old result is not another broken capture. Inference
                # latency can briefly age it out before the next fresh result.
                # Preserve bounded history, revoke readiness, then judge continuity
                # by distinct capture timestamps when that next result arrives.
                self.ready = False
                self.hold_reason = 'Waiting for a new gaze reading'
                return
            self._clear('Waiting for a fresh camera frame')
            return
        if obs.seq == self.last_seq:
            # Qt may poll faster than the camera. A repeated still-fresh frame
            # is not new evidence, but it must not erase a valid fixation.
            return
        if obs.seq < self.last_seq:
            self.invalid_frames += 1
            self._clear('Camera frame arrived out of order')
            return
        self.last_seq = obs.seq
        try:
            features = _feature_row(obs.features)
        except (TypeError, ValueError):
            self.invalid_frames += 1
            self._clear('The camera needs to see both eyes and your head')
            return
        if self.frames:
            previous = self.frames[-1][0]
            gap = captured - float(previous.captured)
            if gap <= 0 or gap >= _FRESH_SECONDS:
                self.invalid_frames += len(self.frames)
                self._clear('The camera stream stopped. Look at the point again.')
        if self.frames:
            window_start = captured - _WINDOW_SECONDS
            while self.frames and float(self.frames[0][0].captured) < window_start:
                self.frames.pop(0)
            if len(self.frames) >= _MAX_WINDOW_FRAMES:
                del self.frames[:len(self.frames) - _MAX_WINDOW_FRAMES + 1]
        self.frames.append((obs, tuple(float(v) for v in features)))
        self.last_observation = obs
        inliers, excluded, moving = self._analyze_fixation()
        if moving:
            # A sustained pose change starts a new candidate at this frame.
            # The prior target's rows cannot leak into the next fixation.
            self._clear('Gaze is moving. Keep looking at the point.')
            self.frames.append((obs, tuple(float(v) for v in features)))
            self.last_observation = obs
            inliers, excluded, _ = self._analyze_fixation()
            self._motion_detected = True
        self._inlier_indices = inliers
        self._excluded_indices = excluded
        newest_is_inlier = bool(self._inlier_indices and
                                self._inlier_indices[-1] == len(self.frames) - 1)
        self.ready = (newest_is_inlier and self.sample_count >= _MIN_FRAMES and
                      self.duration >= _MIN_DURATION)
        if self.ready:
            self.hold_reason = ''
        elif self._motion_detected:
            self.hold_reason = 'Keep looking at the point after moving'
        elif self.excluded_count:
            self.hold_reason = 'Keep looking at the point while the camera gets a steady reading'
        else:
            self.hold_reason = 'Keep looking at the point'

    def confirm(self, now: float) -> bool:
        if self.phase not in ('train', 'check') or not self.ready or not self.last_observation:
            return False
        obs = self.last_observation
        if (obs.seq == self.last_confirmed_seq or not math.isfinite(float(now)) or
                float(now) < obs.captured or float(now) - obs.captured > _FRESH_SECONDS):
            self._clear('A recent camera frame is needed to confirm')
            return False
        feature_rows = [self.frames[i][1] for i in self._inlier_indices]
        if self.phase == 'train':
            mean = np.mean(np.asarray(feature_rows), axis=0)
            target = self.train_targets[self.index]
            training = self.train + [{'target_fraction': target, 'features': tuple(float(v) for v in mean),
                                      'samples': tuple(feature_rows)}]
            if len(training) == len(self.train_targets):
                self.candidate = WebcamProfile.fit([row['features'] for row in training],
                    [row['target_fraction'] for row in training], Rect(*self.bounds), self.device_id,
                    training_samples=[row['samples'] for row in training])
                self.phase = 'check'
                self.index = 0
            else:
                self.index += 1
            self.train = training
        else:
            target = self.check_targets[self.index]
            # Validate every raw row, including rows the stability gate trimmed
            # from the fixation mean, so jitter still counts against usability.
            raw_feature_rows = tuple(row for _, row in self.frames)
            checks = self.check + [{'target_fraction': target, 'features': raw_feature_rows,
                                   'invalid_frames': self.invalid_frames}]
            if len(checks) == len(self.check_targets):
                self.report = self.candidate.validate_rows(checks)
                from dataclasses import replace
                self.candidate = replace(self.candidate, validation_p90_px=self.report.p90_px)
                self.phase = 'result'
                self.index = 0
            else:
                self.index += 1
            self.check = checks
        self.last_confirmed_seq = obs.seq
        self.frames = []
        self.invalid_frames = 0
        self.last_observation = None
        self.ready = False
        self._inlier_indices = ()
        self._excluded_indices = ()
        self._motion_detected = False
        self.hold_reason = ''
        return True


class WebcamProfileStore:
    """Single-file numeric model and receipt with an identity-checked backup."""
    schema_version = 1

    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / 'webcam-profile.json'
        self.backup_path = self.folder / 'webcam-profile.bak.json'

    @staticmethod
    def _canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

    @classmethod
    def _envelope(cls, profile, report):
        model = {'bounds': list(profile.bounds), 'device_id': profile.device_id,
                 'engine_version': profile.engine_version,
                 'coefficients': [list(row) for row in profile.coefficients],
                 'support_anchors': [list(row) for row in profile.support_anchors],
                 'training_rows': list(profile.training_rows),
                 'validation_p90_px': float(profile.validation_p90_px)}
        if profile.mapping is not None:
            model['mapping'] = profile.mapping.as_dict()
        body = {'schema_version': cls.schema_version, 'protocol': WEBCAM_PROTOCOL,
                'model': model, 'check_rows': list(report.check_rows),
                'report': report.as_dict()}
        return {**body, 'sha256': hashlib.sha256(cls._canonical(body)).hexdigest()}

    @classmethod
    def _read(cls, path, bounds, device_id):
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or set(raw) != {'schema_version', 'protocol', 'model', 'check_rows', 'report', 'sha256'}:
            raise ValueError('Malformed webcam profile receipt')
        digest = raw.pop('sha256')
        if type(raw.get('schema_version')) is not int or raw['schema_version'] != cls.schema_version or raw.get('protocol') != WEBCAM_PROTOCOL:
            raise ValueError('Unsupported webcam profile version')
        if not isinstance(digest, str) or len(digest) != 64 or hashlib.sha256(cls._canonical(raw)).hexdigest() != digest:
            raise ValueError('Webcam profile receipt hash does not match')
        model = raw['model']
        required = {'bounds', 'device_id', 'engine_version', 'coefficients', 'support_anchors',
                    'training_rows', 'validation_p90_px'}
        if isinstance(model, dict) and model.get('engine_version') == WEBCAM_ENGINE_VERSION:
            required.add('mapping')
        if not isinstance(model, dict) or set(model) != required:
            raise ValueError('Malformed webcam model')
        profile = WebcamProfile(tuple(_display(model['bounds'])), model['device_id'], model['engine_version'],
            tuple(tuple(float(v) for v in row) for row in model['coefficients']),
            tuple(tuple(float(v) for v in row) for row in model['support_anchors']),
            tuple(model['training_rows']), float(model['validation_p90_px']),
            WebcamMapping.from_dict(model['mapping']) if 'mapping' in model else None)
        profile._identity(bounds, device_id)
        cls._validate_model(profile)
        recomputed = profile.validate_rows(raw['check_rows'])
        if not math.isclose(profile.validation_p90_px, recomputed.p90_px, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Webcam validation radius differs from held-out measurements')
        from dataclasses import replace
        profile = replace(profile, validation_p90_px=recomputed.p90_px)
        if not recomputed.usable:
            raise ValueError('Saved webcam profile did not pass held-out usability checks')
        saved_report = raw['report']
        if not isinstance(saved_report, dict) or saved_report.get('usable') is not True or saved_report.get('passed') is not False:
            raise ValueError('Saved webcam report is not a usable, unverified result')
        if saved_report != recomputed.as_dict():
            raise ValueError('Saved webcam report differs from its held-out data')
        return profile

    @staticmethod
    def _validate_model(profile):
        coeff = np.asarray(profile.coefficients, dtype=float)
        anchors = np.asarray(profile.support_anchors, dtype=float)
        legacy = profile.engine_version == LEGACY_WEBCAM_ENGINE_VERSION
        if (profile.engine_version not in (WEBCAM_ENGINE_VERSION, LEGACY_WEBCAM_ENGINE_VERSION) or not profile.device_id or
                (legacy and profile.mapping is not None) or (not legacy and not isinstance(profile.mapping, WebcamMapping))):
            raise ValueError('Invalid webcam model engine or mapping')
        if profile.mapping is not None:
            WebcamMapping.from_dict(profile.mapping.as_dict())
        columns = 3 if legacy else profile.mapping.columns
        if (coeff.shape != (columns, 2) or anchors.shape != (len(TRAIN_TARGETS), FEATURE_COUNT) or
                not np.isfinite(coeff).all() or not np.isfinite(anchors).all() or
                not math.isfinite(profile.validation_p90_px) or profile.validation_p90_px < 0 or
                len(profile.training_rows) != len(TRAIN_TARGETS)):
            raise ValueError('Invalid webcam model dimensions or engine')
        feats, targets, samples = [], [], []
        for row in profile.training_rows:
            valid_keys = ({'target_fraction', 'features'},) if legacy else ({'target_fraction', 'features'}, {'target_fraction', 'features', 'samples'})
            if not isinstance(row, dict) or set(row) not in valid_keys:
                raise ValueError('Malformed webcam training row')
            feats.append(_feature_row(row['features']))
            targets.append(row['target_fraction'])
            if 'samples' in row:
                samples.append(row['samples'])
        if samples and len(samples) != len(TRAIN_TARGETS):
            raise ValueError('Incomplete webcam training samples')
        if not _same_targets(_target_rows(targets, len(TRAIN_TARGETS), 'Training targets'), TRAIN_TARGETS):
            raise ValueError('Saved webcam training targets are invalid')
        if not np.allclose(np.asarray(feats), anchors, atol=1e-12, rtol=0):
            raise ValueError('Webcam support guard differs from recorded training data')
        expected = WebcamProfile.fit(feats, targets, Rect(*profile.bounds), profile.device_id,
                                     training_samples=samples or None, legacy=legacy)
        if expected.mapping != profile.mapping or not np.allclose(np.asarray(expected.coefficients), coeff, atol=1e-9, rtol=1e-9):
            raise ValueError('Webcam coefficients differ from recorded training data')

    def load(self, bounds, device_id):
        errors = []
        for path in (self.path, self.backup_path):
            try:
                return self._read(path, bounds, device_id)
            except FileNotFoundError:
                errors.append('missing')
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, OverflowError) as exc:
                errors.append(str(exc))
        if errors and all(error == 'missing' for error in errors):
            raise ValueError('Set up the webcam first. Mira couldn’t find a profile for this webcam and screen.')
        raise ValueError('Set up the webcam first. The saved profile is damaged or uses a different webcam or screen (' + '; '.join(errors) + ')')

    def adopt(self, profile: WebcamProfile, report: WebcamValidationReport):
        if not isinstance(profile, WebcamProfile) or not isinstance(report, WebcamValidationReport):
            raise ValueError('A webcam candidate and typed held-out report are required')
        self._validate_model(profile)
        recomputed = profile.validate_rows(report.check_rows)
        if (not recomputed.usable or report.usable is not True or report.passed is not False or
                report.scope != recomputed.scope or report.targets != recomputed.targets or
                report.coverage != recomputed.coverage or report.median_px != recomputed.median_px or
                report.p90_px != recomputed.p90_px or report.jitter_px != recomputed.jitter_px):
            raise ValueError('Webcam candidate lacks a matching successful independent check')
        envelope = self._envelope(profile, recomputed)
        encoded = self._canonical(envelope)
        self.folder.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + '.tmp')
        temp.write_bytes(encoded)
        # Verify the exact staged receipt and model before replacing the active file.
        self._read(temp, Rect(*profile.bounds), profile.device_id)
        if self.path.exists():
            try:
                self._read(self.path, Rect(*profile.bounds), profile.device_id)
                os.replace(self.path, self.backup_path)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, OverflowError):
                # Replacing a corrupt/mismatched primary must not erase a known-good backup.
                pass
        try:
            os.replace(temp, self.path)
        except OSError:
            if not self.path.exists() and self.backup_path.exists():
                os.replace(self.backup_path, self.path)
            raise
        return hashlib.sha256(encoded).hexdigest()
