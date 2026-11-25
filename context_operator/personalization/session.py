"""Raw numeric calibration; training and prospective checks stay separate."""

import json
import hashlib
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ..foundations.gaze.arkit import ENGINE_VERSION, IRIS_NUMBERS, IRIS_PROFILE_NUMBERS


STARTER_PROTOCOL = 'personal-arkit-starter-v2'
SUPERSEDED_PROTOCOLS = ('personal-arkit-starter-v1',)
MIN_FRAMES = 12
MIN_DURATION = .65
SETTLE_SECONDS = .85
DISTANCE_TOLERANCE_M = .07
# What the phone measured on the iris itself, kept with a fixation when every frame has it.
IRIS_RECORDS = (('iris', IRIS_NUMBERS), ('iris_profile', IRIS_PROFILE_NUMBERS))


def _bounds(bounds):
    b = np.asarray(bounds.as_list() if hasattr(bounds, 'as_list') else bounds, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= 0):
        raise ValueError('Invalid display bounds')
    return b


def _target(bounds, fraction):
    return (bounds[:2] + bounds[2:] * fraction).tolist()


class StarterSession:
    """One comfortable seat, thirteen fit targets and five fresh checks."""

    protocol = STARTER_PROTOCOL
    # Rows from the top; the edges are included because a screen plane fitted
    # only to the middle has to guess toward the sides.
    train_targets = ((.10, .12), (.50, .12), (.90, .12), (.30, .31), (.70, .31),
                     (.10, .50), (.50, .50), (.90, .50), (.30, .69), (.70, .69),
                     (.10, .88), (.50, .88), (.90, .88))
    check_targets = ((.20, .21), (.50, .31), (.80, .40), (.60, .79), (.20, .60))
    result_scope = 'starter-independent-targets'

    def __init__(self, bounds, archive=None, pixel_mm=None):
        self.bounds = _bounds(bounds)
        self.archive = Path(archive) if archive else None
        self.pixel_mm = pixel_mm
        self.train = []
        self.check = []
        self.phase = 'train'
        self.index = 0
        self.frames = []
        self.last_seq = -1
        self.settle_until = 0.
        self.candidate = None
        self.report = None
        self.hold_reason = ''
        self.ready_to_confirm = False
        self.candidate_key = None

    @property
    def target_fraction(self):
        if self.phase not in ('train', 'check'):
            return None
        return (self.check_targets if self.phase == 'check' else self.train_targets)[self.index]

    @property
    def target(self):
        fraction = self.target_fraction
        return None if fraction is None else _target(self.bounds, fraction)

    @property
    def complete(self):
        return self.phase == 'result'

    def _save(self):
        if self.archive is None:
            return
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        payload = {'protocol': self.protocol, 'display': self.bounds.tolist(),
                   'train': self.train, 'check': self.check}
        if self.pixel_mm is not None:
            payload['pixel_mm'] = self.pixel_mm
        if self.report is not None:
            payload['report'] = self.report
        if self.candidate_key:
            payload['candidate_sha256'] = self.candidate_key
        temporary = self.archive.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        os.replace(temporary, self.archive)

    @classmethod
    def resume(cls, bounds, archive, pixel_mm=None):
        session = cls(bounds, archive, pixel_mm)
        if not session.archive.is_file():
            return session
        data = json.loads(session.archive.read_text(encoding='utf-8'))
        if isinstance(data, dict) and data.get('protocol') in SUPERSEDED_PROTOCOLS:
            # An earlier layout cannot continue; keep what it collected in the history.
            from .history import preserve_attempt
            preserve_attempt(session.archive)
            return session
        if not isinstance(data, dict) or data.get('protocol') != session.protocol or data.get('display') != session.bounds.tolist():
            raise ValueError('Saved setup belongs to another screen or protocol')
        for phase, positions in (('train', session.train_targets), ('check', session.check_targets)):
            records = data.get(phase, [])
            if not isinstance(records, list) or len(records) > len(positions):
                raise ValueError('Invalid setup archive')
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError('Invalid setup archive')
                expected = np.asarray(_target(session.bounds, positions[index]))
                features = np.asarray(record.get('features'), dtype=float)
                origins = np.asarray(record.get('eye_origins'), dtype=float)
                if (record.get('phase') != phase or
                        record.get('index') != index or
                        not np.allclose(record.get('target'), expected, atol=.01) or
                        features.ndim != 2 or features.shape[1:] != (22,) or
                        len(features) < MIN_FRAMES or origins.shape != (len(features), 2, 3) or
                        not np.isfinite(features).all() or not np.isfinite(origins).all()):
                    raise ValueError('Invalid setup archive')
            kept = [{key: record[key] for key in ('phase', 'index', 'target', 'features', 'eye_origins')}
                    for record in records]
            for record, saved in zip(records, kept):
                for key, size in IRIS_RECORDS:
                    try:
                        numbers = np.asarray(record.get(key), dtype=float)
                    except (TypeError, ValueError):
                        continue
                    if numbers.shape == (len(saved['features']), size) and np.isfinite(numbers).all():
                        saved[key] = record[key]
            setattr(session, phase, kept)
        if session.check and len(session.train) != len(session.train_targets):
            raise ValueError('Check data without complete training')
        if len(session.train) == len(session.train_targets):
            session.phase = 'fit'
            if len(session.check) == len(session.check_targets) and isinstance(data.get('report'), dict):
                report = data['report']
                keys = ('median_px', 'p90_px', 'jitter_px', 'coverage')
                if (type(report.get('passed')) is bool and
                        all(isinstance(report.get(k), (int, float)) and np.isfinite(report[k]) for k in keys) and
                        np.asarray(report.get('bias_px')).shape == (2,)):
                    session.report = report
                    session.phase = 'result'
                    key = data.get('candidate_sha256')
                    if isinstance(key, str) and len(key) == 64 and set(key) <= set('0123456789abcdef'):
                        path = session.archive.parent/'starter-candidates'/f'{key}.npz'
                        try:
                            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                                raise ValueError('Candidate differs')
                            with np.load(path, allow_pickle=False) as saved:
                                metadata = json.loads(str(saved['metadata']))
                            if metadata.get('display') != session.bounds.tolist() or metadata.get('protocol') != session.protocol:
                                raise ValueError('Candidate belongs to another setup')
                            from .model import StarterProfile
                            session.candidate = StarterProfile.load(path, metadata)
                            session.candidate_key = key
                        except (OSError, ValueError, KeyError):
                            session.candidate = None
        else:
            session.index = len(session.train)
        return session

    def persist_result(self):
        """Retain the exact checked model so closing the window loses no result."""
        if self.archive is None or self.report is None or self.candidate is None:
            return
        folder = self.archive.parent/'starter-candidates'
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder/'candidate.staged.npz'
        self.candidate.save(temporary)
        key = hashlib.sha256(temporary.read_bytes()).hexdigest()
        destination = folder/f'{key}.npz'
        os.replace(temporary, destination)
        previous_key = self.candidate_key
        self.candidate_key = key
        try:
            self._save()
        except OSError:
            self.candidate_key = previous_key
            raise

    def distance_hint(self, sample):
        if not self.train:
            return ''
        baseline = float(np.median([np.median(np.asarray(r['features'])[:, 6]) for r in self.train]))
        depth = float(np.asarray(sample.features)[6])
        if abs(depth-baseline) <= DISTANCE_TOLERANCE_M:
            return ''
        return f'Move back to your usual position. You’re at {-depth*100:.0f} cm · aim for about {-baseline*100:.0f} cm.'

    def begin_check(self, candidate, now):
        if self.phase != 'fit':
            raise ValueError('Training is incomplete')
        previous = self.check, self.report, self.candidate_key
        self.check = []
        self.report = None
        self.candidate_key = None
        try:
            self._save()
        except OSError:
            self.check, self.report, self.candidate_key = previous
            raise
        self.candidate = candidate
        self.phase = 'check'
        self.index = 0
        self.frames = []
        self.settle_until = now + SETTLE_SECONDS

    def feed(self, sample, now):
        """Collect a steady fixation for explicit confirmation."""
        if self.phase not in ('train', 'check'):
            return
        try:
            x = np.asarray(sample.features, dtype=float)
            origins = np.asarray(sample.eye_origins, dtype=float)
            captured = float(sample.captured)
            seq = int(sample.seq)
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.frames.clear()
            self.ready_to_confirm = False
            return
        fresh = (not sample.error and not sample.blink and seq > self.last_seq and
                 0 <= now-captured < .20 and captured >= self.settle_until and
                 x.shape == (22,) and origins.shape == (2, 3) and
                 np.isfinite(x).all() and np.isfinite(origins).all())
        if not fresh:
            if self.frames and (seq > self.last_seq or not 0 <= now-captured < .20):
                self.frames.clear()
                self.ready_to_confirm = False
            return
        self.last_seq = seq
        self.ready_to_confirm = False
        if self.frames and captured - self.frames[-1].captured > .20:
            self.frames.clear()
        self.frames.append(sample)
        # Keep a temporal interval, not merely the last 12 packets: at 60 Hz
        # twelve packets span only 183 ms and could never meet the fixation gate.
        self.frames = [s for s in self.frames if captured-s.captured <= 1.2][-180:]
        if len(self.frames) < MIN_FRAMES or captured-self.frames[0].captured < MIN_DURATION:
            return
        x_window = np.asarray([s.features for s in self.frames], dtype=float)
        origins_window = np.asarray([s.eye_origins for s in self.frames], dtype=float)
        # Head translation and eye-direction spread reject motion/saccades.
        head_motion = np.linalg.norm(x_window[:, 4:7]-np.median(x_window[:, 4:7], axis=0), axis=1)
        origin_motion = np.linalg.norm(origins_window-origins_window.mean(axis=0), axis=2).max(axis=1)
        direction_spread = np.linalg.norm(x_window[:, 13:19]-np.median(x_window[:, 13:19], axis=0), axis=1)
        if (np.percentile(head_motion, 90) > .025 or
                np.percentile(origin_motion, 90) > .025 or
                np.percentile(direction_spread, 90) > .12):
            self.hold_reason = 'Keep your head and gaze steady.'
            return
        if self.phase == 'check':
            hint = self.distance_hint(sample)
            if hint:
                self.hold_reason = hint
                return
        self.ready_to_confirm = True
        self.hold_reason = 'Ready · keep looking at the center and press Space'

    def confirm(self, now):
        if (not self.ready_to_confirm or not self.frames or
                not 0 <= now-self.frames[-1].captured < .20):
            return False
        # Persist a small, evenly spaced numeric burst across the qualified
        # interval. Full frame rate is used for the fixation test only.
        chosen = np.rint(np.linspace(0, len(self.frames)-1, MIN_FRAMES)).astype(int)
        targets = self.check_targets if self.phase == 'check' else self.train_targets
        frames = [self.frames[int(i)] for i in chosen]
        record = {'phase': self.phase, 'index': self.index, 'target': _target(self.bounds, targets[self.index]),
                  'features': [np.asarray(s.features, dtype=float).tolist() for s in frames],
                  'eye_origins': [np.asarray(s.eye_origins, dtype=float).tolist() for s in frames]}
        # The phone's own iris measurement rides along when every frame has one,
        # so the two eye signals can be compared on the same fixations.
        for key, size in IRIS_RECORDS:
            readings = [getattr(s, key, ()) for s in frames]
            if all(len(reading) == size for reading in readings):
                record[key] = [list(reading) for reading in readings]
        self.hold_reason = ''
        records = self.check if self.phase == 'check' else self.train
        old_state = self.phase, self.index
        records.append(record)
        self.index += 1
        if self.index == len(targets):
            self.index = 0
            self.phase = 'result' if self.phase == 'check' else 'fit'
        try:
            self._save()
        except OSError:
            records.pop()
            self.phase, self.index = old_state
            raise
        self.settle_until = now + SETTLE_SECONDS
        self.frames = []
        self.ready_to_confirm = False
        return True

    def fit(self):
        if self.phase != 'fit' or len(self.train) != len(self.train_targets):
            raise ValueError('Training is incomplete')
        from .model import StarterProfile
        bursts = [np.asarray(row['features'], dtype=float) for row in self.train]
        origins = [np.asarray(row['eye_origins'], dtype=float) for row in self.train]
        targets = [row['target'] for row in self.train]
        metadata = {'display': self.bounds.tolist(), 'engine': ENGINE_VERSION,
                    'features': 22, 'protocol': self.protocol}
        return StarterProfile.fit(bursts, origins, targets, metadata, self.pixel_mm)

    def evaluate(self):
        if not self.complete or self.candidate is None or len(self.check) != len(self.check_targets):
            raise ValueError('Independent check is incomplete')
        rows = []
        frame_error_batches = []
        b = self.bounds
        for record in self.check:
            x = np.asarray(record['features'], dtype=float)
            target = np.asarray(record['target'], dtype=float)
            predictions = np.asarray(self.candidate.predict(x), dtype=float)
            if predictions.shape != (len(x), 2):
                raise ValueError('Invalid candidate predictions')
            finite = np.isfinite(predictions).all(axis=1)
            reasons = [self.candidate.support_reason(feature) for feature in x]
            supported = np.asarray([not reason for reason in reasons], dtype=bool)
            rejections = Counter(reasons)
            rejections.pop('', None)
            onscreen = finite & np.all((predictions >= b[:2]) & (predictions < b[:2]+b[2:]), axis=1)
            covered = finite & supported & onscreen
            if not finite.any():
                raise ValueError('Candidate has no numeric predictions')
            center = np.median(predictions[finite], axis=0)
            frame_errors = np.linalg.norm(predictions[finite]-target, axis=1)
            frame_error_batches.append(frame_errors)
            rows.append({'target': record['target'],
                         'median_px': float(np.median(frame_errors)),
                         'p90_px': float(np.percentile(frame_errors, 90)),
                         'bias': (center-target).tolist(),
                         'jitter_px': float(np.median(np.linalg.norm(predictions[finite]-center, axis=1))),
                         'coverage': float(np.mean(covered)),
                         'supported_fraction': float(np.mean(supported)),
                         'onscreen_fraction': float(np.mean(onscreen)),
                         'support_rejections': dict(rejections)})
        all_errors = np.concatenate(frame_error_batches)
        coverage = np.asarray([r['coverage'] for r in rows])
        jitter = np.asarray([r['jitter_px'] for r in rows])
        limit_median = min(400., b[3]*.28)
        limit_p90 = min(700., b[3]*.50)
        limit_jitter = min(150., b[3]*.11)
        report = {'protocol': self.protocol, 'scope': self.result_scope,
                  'train_targets': len(self.train_targets), 'check_targets': len(rows),
                  'median_px': float(np.median(all_errors)), 'p90_px': float(np.percentile(all_errors, 90)),
                  'bias_px': np.median([r['bias'] for r in rows], axis=0).tolist(),
                  'jitter_px': float(np.median(jitter)), 'coverage': float(np.mean(coverage)),
                  'targets': rows, 'limits': {'median_px': limit_median, 'p90_px': limit_p90,
                                              'jitter_px': limit_jitter, 'coverage': .80},
                  'quality': 'starter', 'passed': False}
        report['usable'] = bool(report['median_px'] <= limit_median and
                                report['p90_px'] <= limit_p90 and
                                report['jitter_px'] <= limit_jitter and
                                all(row['coverage'] >= .80 for row in report['targets']) and
                                all(row['p90_px'] <= limit_p90 for row in report['targets']) and
                                all(row['jitter_px'] <= limit_jitter for row in report['targets']))
        problems = []
        if report['median_px'] > limit_median:
            problems.append(f"Typical error {report['median_px']:.0f} px · limit {limit_median:.0f} px.")
        bad_error = sum(row['p90_px'] > limit_p90 for row in report['targets'])
        if bad_error:
            problems.append(f'High error at {bad_error} of {len(report["targets"])} points · 90th percentile above {limit_p90:.0f} px.')
        elif report['p90_px'] > limit_p90:
            problems.append(f"90th percentile error {report['p90_px']:.0f} px · limit {limit_p90:.0f} px.")
        bad_jitter = sum(row['jitter_px'] > limit_jitter for row in report['targets'])
        if bad_jitter:
            problems.append(f'Unsteady gaze at {bad_jitter} points · movement above {limit_jitter:.0f} px.')
        unsupported = sum(row['supported_fraction'] < .80 for row in report['targets'])
        if unsupported:
            problems.append(f'At {unsupported} of {len(report["targets"])} points, eye and head position fell outside the setup range.')
        outside = sum(row['onscreen_fraction'] < .80 for row in report['targets'])
        if outside:
            problems.append(f'At {outside} of {len(report["targets"])} points, gaze often fell outside the screen.')
        if not report['usable'] and not problems:
            problems.append('At some points, fewer than 80% of readings were both in the setup range and on the screen.')
        report['problems'] = problems
        self.report = report
        return report
