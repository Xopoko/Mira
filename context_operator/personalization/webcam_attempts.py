"""Bounded numeric archive of completed webcam calibration attempts.

Attempts are diagnostic records only. Saving one never adopts a profile.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .webcam import (
    CHECK_TARGETS,
    FEATURE_COUNT,
    TRAIN_TARGETS,
    WEBCAM_ENGINE_VERSION,
    WebcamMapping,
    WebcamProfile,
    WebcamProfileStore,
    WebcamSession,
    WebcamValidationReport,
)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate webcam attempt JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Invalid numeric constant in webcam attempt: {value}')


def _number(value, name, *, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'Invalid numeric webcam attempt field: {name}')
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f'Invalid numeric webcam attempt field: {name}')
    return result


def _vector(value, size, name, *, minimum=None):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'Invalid webcam attempt vector: {name}')
    return [_number(item, name, minimum=minimum) for item in value]


def _feature_matrix(value, name, *, minimum_rows=1, maximum_rows=90):
    if (not isinstance(value, (list, tuple)) or
            not minimum_rows <= len(value) <= maximum_rows):
        raise ValueError(f'Invalid webcam attempt feature rows: {name}')
    return [_vector(row, FEATURE_COUNT, name) for row in value]


def _json_tree(value, name, *, budget=None, depth=0):
    """Convert bounded model metadata to JSON-safe primitives and reject NaNs."""
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > 250_000 or depth > 16:
        raise ValueError(f'Webcam attempt model exceeds the bounded size: {name}')
    if value is None or type(value) in (str, bool):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError(f'Webcam attempt string is too long: {name}')
        return value
    if type(value) in (int, float):
        return _number(value, name)
    if isinstance(value, dict):
        if len(value) > 4096 or any(not isinstance(key, str) for key in value):
            raise ValueError(f'Invalid webcam attempt mapping: {name}')
        return {key: _json_tree(item, name, budget=budget, depth=depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 20_000:
            raise ValueError(f'Webcam attempt vector is too large: {name}')
        return [_json_tree(item, name, budget=budget, depth=depth + 1)
                for item in value]
    raise ValueError(f'Unsupported webcam attempt value: {name}')


def _model_training_rows(value):
    if not isinstance(value, (list, tuple)) or len(value) != len(TRAIN_TARGETS):
        raise ValueError('Invalid webcam attempt model training rows')
    rows = []
    for row, expected in zip(value, TRAIN_TARGETS):
        if (not isinstance(row, dict) or
                not {'target_fraction', 'features'} <= set(row)):
            raise ValueError('Malformed webcam attempt model training row')
        target = _vector(row['target_fraction'], 2, 'model training target')
        if target != list(expected):
            raise ValueError('Unexpected webcam model training target')
        clean = dict(row)
        clean['target_fraction'] = target
        clean['features'] = _vector(row['features'], FEATURE_COUNT,
                                    'model training features')
        if 'samples' in clean:
            clean['samples'] = _feature_matrix(clean['samples'],
                                               'model training samples',
                                               minimum_rows=1,
                                               maximum_rows=90)
        _json_tree(clean, 'model training row')
        rows.append(clean)
    return rows


def _check_rows(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != len(CHECK_TARGETS):
        raise ValueError(f'Invalid webcam attempt check rows: {name}')
    rows = []
    for index, (row, expected) in enumerate(zip(value, CHECK_TARGETS)):
        if not isinstance(row, dict) or set(row) != {
                'target_fraction', 'features', 'invalid_frames'}:
            raise ValueError(f'Malformed webcam attempt check row: {name}')
        target = _vector(row['target_fraction'], 2, name)
        if target != list(expected):
            raise ValueError(f'Unexpected webcam check target: {name}')
        if (type(row['invalid_frames']) is not int or
                not 0 <= row['invalid_frames'] <= 10_000):
            raise ValueError(f'Invalid webcam check invalid-frame count: {name}')
        rows.append({'target_fraction': target,
                     'features': _feature_matrix(row['features'], name,
                                                  minimum_rows=10),
                     'invalid_frames': row['invalid_frames']})
    return rows


def _validate_report(value, check_rows):
    expected_keys = {
        'usable', 'passed', 'scope', 'coverage', 'errors_px', 'median_px',
        'p90_px', 'jitter_px', 'targets', 'check_rows', 'limits',
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError('Malformed webcam attempt validation report')
    if type(value['usable']) is not bool or value['passed'] is not False:
        raise ValueError('Invalid webcam attempt validation state')
    if value['scope'] != 'webcam-secondary-heldout-usability':
        raise ValueError('Invalid webcam attempt validation scope')
    coverage = _vector(value['coverage'], len(CHECK_TARGETS), 'coverage')
    if any(score < 0 or score > 1 for score in coverage):
        raise ValueError('Invalid webcam attempt coverage')
    errors = value['errors_px']
    if not isinstance(errors, (list, tuple)) or len(errors) != len(CHECK_TARGETS):
        raise ValueError('Invalid webcam attempt error groups')
    errors = [_vector(row, len(check_rows[i]['features']), 'errors_px', minimum=0)
              for i, row in enumerate(errors)]
    targets = value['targets']
    if not isinstance(targets, (list, tuple)) or len(targets) != len(CHECK_TARGETS):
        raise ValueError('Invalid webcam attempt target metrics')
    checked_targets = []
    for i, (row, expected) in enumerate(zip(targets, CHECK_TARGETS)):
        if not isinstance(row, dict) or set(row) != {
                'index', 'target_fraction', 'frames', 'invalid_frames', 'coverage',
                'median_px', 'p90_px', 'jitter_px'}:
            raise ValueError('Malformed webcam attempt target metrics')
        if type(row['index']) is not int or row['index'] != i:
            raise ValueError('Unexpected webcam attempt target index')
        if _vector(row['target_fraction'], 2, 'target_fraction') != list(expected):
            raise ValueError('Unexpected webcam attempt metric target')
        if row['frames'] != len(check_rows[i]['features']) or type(row['frames']) is not int:
            raise ValueError('Webcam attempt metric frame count does not match data')
        if row['invalid_frames'] != check_rows[i]['invalid_frames']:
            raise ValueError('Webcam attempt invalid-frame count does not match data')
        if _number(row['coverage'], 'target coverage') != coverage[i]:
            raise ValueError('Webcam attempt target coverage does not match report')
        checked_targets.append({
            'index': i,
            'target_fraction': list(expected),
            'frames': row['frames'],
            'invalid_frames': row['invalid_frames'],
            'coverage': coverage[i],
            'median_px': _number(row['median_px'], 'target median', minimum=0),
            'p90_px': _number(row['p90_px'], 'target p90', minimum=0),
            'jitter_px': _number(row['jitter_px'], 'target jitter', minimum=0),
        })
    limits = value['limits']
    if not isinstance(limits, dict) or set(limits) != {
            'median_px', 'p90_px', 'jitter_px', 'coverage'}:
        raise ValueError('Malformed webcam attempt validation limits')
    clean_limits = {key: _number(item, 'limit', minimum=0)
                    for key, item in limits.items()}
    clean_checks = _check_rows(value['check_rows'], 'report check rows')
    if clean_checks != check_rows:
        raise ValueError('Webcam attempt report check rows do not match session')
    return {
        'usable': value['usable'], 'passed': False, 'scope': value['scope'],
        'coverage': coverage, 'errors_px': errors,
        'median_px': _number(value['median_px'], 'median', minimum=0),
        'p90_px': _number(value['p90_px'], 'p90', minimum=0),
        'jitter_px': _number(value['jitter_px'], 'jitter', minimum=0),
        'targets': checked_targets, 'check_rows': clean_checks,
        'limits': clean_limits,
    }


def _validate_payload(value):
    expected_keys = {
        'schema_version', 'protocol', 'attempt_id', 'created_at', 'usable',
        'passed', 'bounds', 'device_id', 'engine_version', 'train_rows',
        'check_rows', 'model', 'report',
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError('Malformed webcam attempt record')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Unsupported webcam attempt schema')
    if value['protocol'] != 'contextoperator.webcam-attempt.v1':
        raise ValueError('Unsupported webcam attempt protocol')
    attempt_id = value['attempt_id']
    if not isinstance(attempt_id, str) or len(attempt_id) != 32:
        raise ValueError('Invalid webcam attempt identity')
    try:
        uuid.UUID(hex=attempt_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError('Invalid webcam attempt identity') from exc
    created_at = value['created_at']
    if not isinstance(created_at, str) or len(created_at) > 64:
        raise ValueError('Invalid webcam attempt timestamp')
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError('Invalid webcam attempt timestamp') from exc
    if timestamp.tzinfo is None:
        raise ValueError('Webcam attempt timestamp needs a timezone')
    bounds = _vector(value['bounds'], 4, 'bounds')
    if bounds[2] <= 0 or bounds[3] <= 0:
        raise ValueError('Invalid webcam attempt display bounds')
    device_id = value['device_id']
    if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 512:
        raise ValueError('Invalid webcam attempt camera identity')
    engine_version = value['engine_version']
    if not isinstance(engine_version, str) or not engine_version or len(engine_version) > 128:
        raise ValueError('Invalid webcam attempt engine identity')
    train_rows = _model_training_rows(value['train_rows'])
    check_rows = _check_rows(value['check_rows'], 'check rows')
    report = _validate_report(value['report'], check_rows)
    if type(value['usable']) is not bool or value['usable'] != report['usable']:
        raise ValueError('Webcam attempt usability does not match report')
    if value['passed'] is not False:
        raise ValueError('Webcam attempt must remain an unpassed usability report')
    model = value['model']
    model_keys = {
        'bounds', 'device_id', 'engine_version', 'coefficients',
        'support_anchors', 'training_rows', 'validation_p90_px',
    }
    if not isinstance(model, dict) or not model_keys <= set(model):
        raise ValueError('Malformed webcam attempt numeric model')
    if (_vector(model['bounds'], 4, 'model bounds') != bounds or
            model['device_id'] != device_id or
            model['engine_version'] != engine_version):
        raise ValueError('Webcam attempt model identity does not match record')
    clean_coefficients = _json_tree(model['coefficients'], 'coefficients')
    anchors = _feature_matrix(model['support_anchors'], 'support anchors',
                              minimum_rows=len(TRAIN_TARGETS),
                              maximum_rows=len(TRAIN_TARGETS))
    model_train = _model_training_rows(model['training_rows'])
    if model_train != train_rows:
        raise ValueError('Webcam attempt model training data does not match session')
    validation_p90 = _number(model['validation_p90_px'], 'model validation p90', minimum=0)
    if validation_p90 != report['p90_px']:
        raise ValueError('Webcam attempt model metric does not match report')
    mapping = (WebcamMapping.from_dict(model['mapping'])
               if 'mapping' in model else None)
    profile = WebcamProfile(
        tuple(bounds), device_id, engine_version,
        tuple(tuple(_number(item, 'coefficient') for item in row)
              for row in clean_coefficients),
        tuple(tuple(row) for row in anchors), tuple(model_train),
        validation_p90, mapping)
    WebcamProfileStore._validate_model(profile)
    if profile.validate_rows(check_rows).as_dict() != report:
        raise ValueError('Webcam attempt report differs from numeric check data')
    return {
        **value, 'bounds': bounds, 'train_rows': train_rows,
        'check_rows': check_rows, 'report': report,
        'model': {**_json_tree(model, 'model'), 'bounds': bounds,
                  'coefficients': clean_coefficients,
                  'support_anchors': anchors, 'training_rows': model_train,
                  'validation_p90_px': validation_p90},
    }


class WebcamAttemptStore:
    """Persist bounded numeric receipts for complete webcam setup attempts."""

    schema_version = 1
    protocol = 'contextoperator.webcam-attempt.v1'
    archive_limit = 10
    max_attempt_bytes = 2 * 1024 * 1024

    def __init__(self, folder):
        self.root = Path(folder) / 'webcam-attempts'
        self.latest_path = self.root / 'latest.json'
        self.archive = self.root / 'archive'

    @staticmethod
    def _canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')

    @classmethod
    def _snapshot(cls, session: WebcamSession):
        if (not isinstance(session, WebcamSession) or not session.complete or
                session.phase != 'result' or session.index != 0 or
                not isinstance(session.candidate, WebcamProfile) or
                not isinstance(session.report, WebcamValidationReport)):
            raise ValueError('Only a complete webcam calibration attempt can be saved')
        profile, report = session.candidate, session.report
        if (profile.bounds != tuple(session.bounds) or
                profile.device_id != session.device_id or
                profile.engine_version != WEBCAM_ENGINE_VERSION):
            raise ValueError('Webcam attempt model identity does not match session')
        if len(session.train) != len(TRAIN_TARGETS) or len(session.check) != len(CHECK_TARGETS):
            raise ValueError('Webcam attempt is missing training or check targets')

        train_rows = _model_training_rows(session.train)
        check_rows = _check_rows(session.check, 'session check rows')
        clean_report = _validate_report(report.as_dict(), check_rows)
        # The report must describe this exact completed check set. Recalculation
        # checks record consistency, while preserving both usable and failed outcomes.
        recomputed = profile.validate_rows(session.check)
        if recomputed != report:
            raise ValueError('Webcam attempt report does not match its numeric check rows')
        profile_rows = _model_training_rows(profile.training_rows)
        WebcamProfileStore._validate_model(profile)
        if profile_rows != train_rows:
            raise ValueError('Webcam attempt profile does not match training rows')

        model = _json_tree(asdict(profile), 'model')
        payload = {
            'schema_version': cls.schema_version,
            'protocol': cls.protocol,
            'attempt_id': uuid.uuid4().hex,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'usable': clean_report['usable'],
            'passed': False,
            'bounds': list(profile.bounds),
            'device_id': profile.device_id,
            'engine_version': profile.engine_version,
            'train_rows': train_rows,
            'check_rows': check_rows,
            'model': model,
            'report': clean_report,
        }
        return _validate_payload(payload)

    @classmethod
    def _envelope(cls, payload):
        body = cls._canonical(payload)
        return {**payload, 'sha256': hashlib.sha256(body).hexdigest()}

    @staticmethod
    def _atomic_write(path: Path, content: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _prune(self):
        files = sorted(self.archive.glob('webcam-attempt-*.json'),
                       key=lambda path: (path.stat().st_mtime_ns, path.name),
                       reverse=True)
        for path in files[self.archive_limit:]:
            path.unlink()

    def save(self, session: WebcamSession) -> Path:
        """Save latest and one bounded archive entry; never adopts the profile."""
        payload = self._snapshot(session)
        encoded = self._canonical(self._envelope(payload))
        if len(encoded) > self.max_attempt_bytes:
            raise ValueError('Webcam attempt exceeds the bounded storage limit')
        filename = 'webcam-attempt-' + payload['created_at'].replace(':', '').replace('-', '') + '-' + payload['attempt_id'] + '.json'
        archive_path = self.archive / filename
        self._atomic_write(archive_path, encoded)
        self._atomic_write(self.latest_path, encoded)
        self._prune()
        return archive_path

    def load_latest(self):
        """Read and strictly validate the latest numeric attempt receipt."""
        size = self.latest_path.stat().st_size
        if size <= 0 or size > self.max_attempt_bytes:
            raise ValueError('Invalid webcam attempt file size')
        encoded = self.latest_path.read_bytes()
        if len(encoded) != size:
            raise ValueError('Webcam attempt changed while reading')
        raw = json.loads(encoded.decode('utf-8'), parse_constant=_reject_constant,
                         object_pairs_hook=_unique_pairs)
        if not isinstance(raw, dict) or set(raw) != {
                'schema_version', 'protocol', 'attempt_id', 'created_at', 'usable',
                'passed', 'bounds', 'device_id', 'engine_version', 'train_rows',
                'check_rows', 'model', 'report', 'sha256'}:
            raise ValueError('Malformed webcam attempt envelope')
        digest = raw.pop('sha256')
        actual = hashlib.sha256(self._canonical(raw)).hexdigest()
        if not isinstance(digest, str) or digest != actual:
            raise ValueError('Webcam attempt checksum mismatch')
        return _validate_payload(raw)
