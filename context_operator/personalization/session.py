"""Raw numeric calibration; training and prospective checks stay separate."""

import json
import hashlib
import os
from collections import Counter
from pathlib import Path

import numpy as np

from ..foundations.gaze.arkit import ENGINE_VERSION
from ..foundations.gaze.profile import Profile


# Different screen locations: no check target enters the fit or model search.
TRAIN_TARGETS = tuple((x, y) for y in (.12, .50, .88) for x in (.10, .36, .64, .90))
CHECK_TARGETS = tuple((x, y) for y in (.22, .50, .78) for x in (.23, .50, .77))
POSTURES = ('Usual position', 'Slightly closer', 'Slightly farther away')
PROTOCOL = 'personal-arkit-direct-v1'
STARTER_PROTOCOL = 'personal-arkit-starter-v1'
MIN_FRAMES = 12
MIN_DURATION = .65
SETTLE_SECONDS = .85
POSTURE_DEPTH_DELTA_M = .07
POSTURE_DEPTH_MAX_M = .14


def _bounds(bounds):
    b = np.asarray(bounds.as_list() if hasattr(bounds, 'as_list') else bounds, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= 0):
        raise ValueError('Invalid display bounds')
    return b


def _target(bounds, fraction):
    return (bounds[:2] + bounds[2:] * fraction).tolist()


def _record(phase, posture, index, bounds, frames, train_targets=TRAIN_TARGETS,
            check_targets=CHECK_TARGETS):
    target = train_targets[index] if phase != 'check' else check_targets[index]
    return {'phase': phase, 'posture': posture, 'index': index,
            'target': _target(bounds, target),
            'features': [np.asarray(s.features, dtype=float).tolist() for s in frames],
            'eye_origins': [np.asarray(s.eye_origins, dtype=float).tolist() for s in frames]}


class PersonalSession:
    """Three prompted postures; 12 fit targets and nine unseen checks each.

    A target advances by fresh, steady fixation alone. The only explicit pause
    is between postures. This object stores no screen or camera images.
    """

    protocol = PROTOCOL
    train_targets = TRAIN_TARGETS
    check_targets = CHECK_TARGETS
    postures = POSTURES
    posture_count = 3
    result_scope = 'same-session-independent-targets'

    def __init__(self, bounds, archive=None, *, require_confirmation=False):
        self.bounds = _bounds(bounds)
        self.archive = Path(archive) if archive else None
        self.train = []
        self.check = []
        self.phase = 'train'
        self.posture = 0
        self.index = 0
        self.paused = False
        self.frames = []
        self.last_seq = -1
        self.settle_until = 0.
        self.candidate = None
        self.report = None
        self.hold_reason = ''
        self.require_confirmation = require_confirmation
        self.ready_to_confirm = False
        self.repair_postures = []
        self.candidate_key = None

    @property
    def target_fraction(self):
        if self.phase not in ('train', 'repair', 'check') or self.paused:
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
        if self.report is not None:
            payload['report'] = self.report
        if self.phase == 'repair':
            payload['repair_postures'] = self.repair_postures
        if self.candidate_key:
            payload['candidate_sha256'] = self.candidate_key
        temporary = self.archive.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        os.replace(temporary, self.archive)

    @classmethod
    def resume(cls, bounds, archive, *, require_confirmation=False):
        session = cls(bounds, archive, require_confirmation=require_confirmation)
        if not session.archive.is_file():
            return session
        data = json.loads(session.archive.read_text(encoding='utf-8'))
        if data.get('protocol') != session.protocol or data.get('display') != session.bounds.tolist():
            raise ValueError('Saved setup belongs to another screen or protocol')
        for phase, positions in (('train', session.train_targets), ('check', session.check_targets)):
            records = data.get(phase, [])
            if not isinstance(records, list) or len(records) > session.posture_count * len(positions):
                raise ValueError('Invalid setup archive')
            for number, record in enumerate(records):
                posture, index = divmod(number, len(positions))
                expected = np.asarray(_target(session.bounds, positions[index]))
                features = np.asarray(record.get('features'), dtype=float)
                origins = np.asarray(record.get('eye_origins'), dtype=float)
                if (record.get('phase') != phase or record.get('posture') != posture or
                        record.get('index') != index or
                        not np.allclose(record.get('target'), expected, atol=.01) or
                        features.ndim != 2 or features.shape[1:] != (22,) or
                        len(features) < MIN_FRAMES or origins.shape != (len(features), 2, 3) or
                        not np.isfinite(features).all() or not np.isfinite(origins).all()):
                    raise ValueError('Invalid setup archive')
            setattr(session, phase, records)
        if session.check and len(session.train) != session.posture_count * len(session.train_targets):
            raise ValueError('Check data without complete training')
        if len(session.train) == session.posture_count * len(session.train_targets):
            session.phase = 'fit'
            session.posture = 0
            session.index = 0
            repairs = data.get('repair_postures', [])
            if repairs:
                if not isinstance(repairs, list) or any(type(p) is not int or p not in (0, 1, 2) for p in repairs):
                    raise ValueError('Invalid repair state')
                session.repair_postures = sorted(set(repairs))
                session.phase = 'repair'
                session.posture = session.repair_postures[0]
                session.paused = True
            elif len(session.check) == session.posture_count * len(session.check_targets) and isinstance(data.get('report'), dict):
                report = data['report']
                keys = ('median_px', 'p90_px', 'jitter_px', 'coverage')
                if (type(report.get('passed')) is bool and
                        all(isinstance(report.get(k), (int, float)) and np.isfinite(report[k]) for k in keys) and
                        np.asarray(report.get('bias_px')).shape == (2,)):
                    session.report = report
                    session.phase = 'result'
                    key = data.get('candidate_sha256')
                    if isinstance(key, str) and len(key) == 64 and set(key) <= set('0123456789abcdef'):
                        candidate_folder = ('starter-candidates' if session.protocol == STARTER_PROTOCOL
                                            else 'personal-candidates')
                        path = session.archive.parent/candidate_folder/f'{key}.npz'
                        try:
                            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                                raise ValueError('Candidate differs')
                            with np.load(path, allow_pickle=False) as saved:
                                metadata = json.loads(str(saved['metadata']))
                                kind = json.loads(str(saved['settings'])).get('model_kind')
                            if metadata.get('display') != session.bounds.tolist() or metadata.get('protocol') != session.protocol:
                                raise ValueError('Candidate belongs to another setup')
                            from .model import PersonalProfile, StarterProfile
                            if kind == PersonalProfile.MODEL_KIND:
                                session.candidate = PersonalProfile.load(path, metadata)
                            elif kind == StarterProfile.MODEL_KIND:
                                session.candidate = StarterProfile.load(path, metadata)
                            elif kind is None:
                                session.candidate = Profile.load(path, metadata)
                            else:
                                raise ValueError('Unknown candidate model')
                            session.candidate_key = key
                        except (OSError, ValueError, KeyError):
                            session.candidate = None
        else:
            session.posture, session.index = divmod(len(session.train), len(session.train_targets))
            session.paused = session.posture > 0 and session.index == 0
        return session

    def next_posture(self, now):
        if not self.paused:
            raise ValueError('No posture change requested')
        self.paused = False
        self.settle_until = now + SETTLE_SECONDS

    def begin_repair(self, postures):
        if len(self.train) != self.posture_count * len(self.train_targets):
            raise ValueError('Training is incomplete')
        selected = sorted(set(postures))
        if not selected or any(type(p) is not int or p not in (0, 1, 2) for p in selected):
            raise ValueError('Invalid repair postures')
        previous = {key: getattr(self, key) for key in ('repair_postures','phase','posture','index',
                    'paused','frames','ready_to_confirm','report','candidate_key')}
        self.repair_postures = selected
        self.phase = 'repair'
        self.posture, self.index = selected[0], 0
        self.paused = True
        self.frames = []
        self.ready_to_confirm = False
        self.report = None
        self.candidate_key = None
        try:
            self._save()
        except OSError:
            for key, value in previous.items():
                setattr(self, key, value)
            raise

    def persist_result(self):
        """Retain the exact checked model so closing the window loses no result."""
        if self.archive is None or self.report is None or self.candidate is None:
            return
        folder_name = 'starter-candidates' if self.protocol == STARTER_PROTOCOL else 'personal-candidates'
        folder = self.archive.parent/folder_name
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

    def posture_hint(self, sample):
        """Bound the requested pose instead of accepting any greater distance."""
        if not self.train:
            return ''
        baseline = float(np.median([np.median(np.asarray(r['features'])[:, 6])
                                    for r in self.train if r['posture'] == 0]))
        depth = float(np.asarray(sample.features)[6])
        delta = depth-baseline
        if self.posture == 0:
            valid = abs(delta) <= POSTURE_DEPTH_DELTA_M
        else:
            offset = delta if self.posture == 1 else -delta
            valid = POSTURE_DEPTH_DELTA_M <= offset <= POSTURE_DEPTH_MAX_M
        if valid:
            return ''
        desired = -baseline + (0 if self.posture == 0 else -.095 if self.posture == 1 else .095)
        direction = 'back to your usual position' if self.posture == 0 else 'slightly closer' if self.posture == 1 else 'slightly farther away'
        return f'Move {direction}. You’re at {-depth*100:.0f} cm · aim for about {desired*100:.0f} cm.'

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
        self.posture = self.index = 0
        self.paused = False
        self.frames = []
        self.settle_until = now + SETTLE_SECONDS

    def feed(self, sample, now):
        """Return true when a target completes; never predicts with an old model."""
        if self.phase not in ('train', 'repair', 'check') or self.paused:
            return False
        try:
            x = np.asarray(sample.features, dtype=float)
            origins = np.asarray(sample.eye_origins, dtype=float)
            captured = float(sample.captured)
            seq = int(sample.seq)
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.frames.clear()
            self.ready_to_confirm = False
            return False
        fresh = (not sample.error and not sample.blink and seq > self.last_seq and
                 0 <= now-captured < .20 and captured >= self.settle_until and
                 x.shape == (22,) and origins.shape == (2, 3) and
                 np.isfinite(x).all() and np.isfinite(origins).all())
        if not fresh:
            if self.frames and (seq > self.last_seq or not 0 <= now-captured < .20):
                self.frames.clear()
                self.ready_to_confirm = False
            return False
        self.last_seq = seq
        self.ready_to_confirm = False
        if self.frames and captured - self.frames[-1].captured > .20:
            self.frames.clear()
        self.frames.append(sample)
        # Keep a temporal interval, not merely the last 12 packets: at 60 Hz
        # twelve packets span only 183 ms and could never meet the fixation gate.
        self.frames = [s for s in self.frames if captured-s.captured <= 1.2][-180:]
        if len(self.frames) < MIN_FRAMES or captured-self.frames[0].captured < MIN_DURATION:
            return False
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
            return False
        if self.phase in ('check', 'repair') or self.posture:
            hint = self.posture_hint(sample)
            if hint:
                self.hold_reason = hint
                return False
        self.ready_to_confirm = True
        if self.require_confirmation:
            self.hold_reason = 'Ready · keep looking at the center and press Space'
            return False
        return self.confirm(now)

    def confirm(self, now):
        if (not self.ready_to_confirm or not self.frames or
                not 0 <= now-self.frames[-1].captured < .20):
            return False
        # Persist a small, evenly spaced numeric burst across the qualified
        # interval. Full frame rate is used for the fixation test only.
        chosen = np.rint(np.linspace(0, len(self.frames)-1, MIN_FRAMES)).astype(int)
        record = _record('check' if self.phase == 'check' else 'train', self.posture, self.index, self.bounds,
                         [self.frames[int(i)] for i in chosen], self.train_targets, self.check_targets)
        record['confirmed'] = self.require_confirmation
        self.hold_reason = ''
        records = self.check if self.phase == 'check' else self.train
        repair = self.phase == 'repair'
        old_state = (self.phase, self.posture, self.index, self.paused)
        slot = self.posture*len(self.train_targets)+self.index
        old_record = records[slot] if repair else None
        if repair:
            records[slot] = record
        else:
            records.append(record)
        self.index += 1
        count = len(self.check_targets if self.phase == 'check' else self.train_targets)
        if self.index == count:
            self.index = 0
            self.posture = (next((p for p in self.repair_postures if p > self.posture), self.posture_count)
                            if repair else self.posture+1)
            if self.posture == self.posture_count:
                self.phase = 'result' if self.phase == 'check' else 'fit'
                self.posture = 0
            else:
                self.paused = True
        try:
            self._save()
        except OSError:
            if repair:
                records[slot] = old_record
            else:
                records.pop()
            self.phase, self.posture, self.index, self.paused = old_state
            raise
        self.settle_until = now + SETTLE_SECONDS
        self.frames = []
        self.ready_to_confirm = False
        return True

    def fit(self):
        if self.phase != 'fit' or len(self.train) != self.posture_count * len(self.train_targets):
            raise ValueError('Training is incomplete')
        bursts, y, groups, postures = [], [], [], []
        for row in self.train:
            features = np.asarray(row['features'], dtype=float)
            bursts.append(features)
            y.append(row['target'])
            groups.append(row['index'])
            postures.append(row['posture'])
        metadata = {'display': self.bounds.tolist(), 'engine': ENGINE_VERSION,
                    'features': 22, 'protocol': self.protocol}
        return self.fit_model(bursts, y, metadata, groups, postures)

    @staticmethod
    def fit_model(bursts, targets, metadata, groups, postures):
        from .model import PersonalProfile
        return PersonalProfile.fit(bursts, targets, metadata, groups, postures)

    def evaluate(self):
        if not self.complete or self.candidate is None or len(self.check) != self.posture_count * len(self.check_targets):
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
            supported = np.asarray([self.candidate.supports(feature) for feature in x], dtype=bool)
            support_reason = getattr(self.candidate, 'support_reason', None)
            rejections = Counter(support_reason(feature) for feature in x) if support_reason else Counter()
            rejections.pop('', None)
            onscreen = finite & np.all((predictions >= b[:2]) & (predictions < b[:2]+b[2:]), axis=1)
            covered = finite & supported & onscreen
            if not finite.any():
                raise ValueError('Candidate has no numeric predictions')
            center = np.median(predictions[finite], axis=0)
            frame_errors = np.linalg.norm(predictions[finite]-target, axis=1)
            frame_error_batches.append(frame_errors)
            rows.append({'posture': record['posture'], 'target': record['target'],
                         'median_px': float(np.median(frame_errors)),
                         'p90_px': float(np.percentile(frame_errors, 90)),
                         'bias': (center-target).tolist(),
                         'jitter_px': float(np.median(np.linalg.norm(predictions[finite]-center, axis=1))),
                         'coverage': float(np.mean(covered)),
                         'supported_fraction': float(np.mean(supported)),
                         'onscreen_fraction': float(np.mean(onscreen)),
                         'support_rejections': dict(rejections)})
        errors = np.asarray([r['median_px'] for r in rows])
        all_errors = np.concatenate(frame_error_batches)
        target_p90 = np.asarray([r['p90_px'] for r in rows])
        coverage = np.asarray([r['coverage'] for r in rows])
        jitter = np.asarray([r['jitter_px'] for r in rows])
        posture_medians = [float(np.median(errors[i*len(self.check_targets):(i+1)*len(self.check_targets)])) for i in range(self.posture_count)]
        limit_median = min(280., b[3]*.20)
        limit_p90 = min(460., b[3]*.32)
        limit_jitter = min(90., b[3]*.065)
        report = {'protocol': self.protocol, 'scope': self.result_scope,
                  'train_targets': len(self.train_targets), 'check_targets': len(rows),
                  'median_px': float(np.median(all_errors)), 'p90_px': float(np.percentile(all_errors, 90)),
                  'bias_px': np.median([r['bias'] for r in rows], axis=0).tolist(),
                  'jitter_px': float(np.median(jitter)), 'coverage': float(np.mean(coverage)),
                  'posture_median_px': posture_medians, 'targets': rows,
                  'limits': {'median_px': limit_median, 'p90_px': limit_p90,
                             'jitter_px': limit_jitter, 'coverage': .90},
                  'passed': bool(np.median(all_errors) <= limit_median and
                                 np.percentile(all_errors, 90) <= limit_p90 and
                                 np.max(target_p90) <= limit_p90 and
                                 np.median(jitter) <= limit_jitter and
                                 np.all(coverage >= .90) and
                                 all(value <= limit_median for value in posture_medians))}
        problems = []
        for index, value in enumerate(posture_medians):
            if value > limit_median:
                problems.append(f'{self.postures[index]}: error {value:.0f} px (target under {limit_median:.0f}).')
        if np.max(target_p90) > limit_p90:
            count = int(np.sum(target_p90 > limit_p90))
            problems.append(f'High error at {count} of {len(rows)} check points.')
        if np.median(jitter) > limit_jitter:
            problems.append(f'Unsteady point · movement {np.median(jitter):.0f} px.')
        if np.any(coverage < .90):
            problems.append('Some positions are outside the setup range or off the screen.')
        report['problems'] = problems
        report['failed_postures'] = [p for p in range(self.posture_count) if (
            posture_medians[p] > limit_median or
            max(r['p90_px'] for r in rows if r['posture'] == p) > limit_p90 or
            np.median([r['jitter_px'] for r in rows if r['posture'] == p]) > limit_jitter or
            min(r['coverage'] for r in rows if r['posture'] == p) < .90)]
        if report['failed_postures']:
            report['passed'] = False
        self.report = report
        return report


class StarterSession(PersonalSession):
    """One comfortable head position, nine fit targets and five fresh checks."""

    protocol = STARTER_PROTOCOL
    train_targets = tuple((x, y) for y in (.14, .45, .86) for x in (.14, .45, .86))
    check_targets = ((.08, .10), (.92, .10), (.08, .90), (.92, .90), (.61, .39))
    postures = ('One comfortable position',)
    posture_count = 1
    result_scope = 'starter-independent-targets'

    @staticmethod
    def fit_model(bursts, targets, metadata, groups, postures):
        from .model import StarterProfile
        return StarterProfile.fit(bursts, targets, metadata)

    def evaluate(self):
        report = super().evaluate()
        b = self.bounds
        # Five held-out points support coarse usability only; they never satisfy
        # the stricter three-posture acceptance contract.
        limit_median = min(400., b[3]*.28)
        limit_p90 = min(700., b[3]*.50)
        limit_jitter = min(150., b[3]*.11)
        report['quality'] = 'starter'
        report['strict_passed'] = False
        report['passed'] = False
        report['limits'] = {'median_px': limit_median, 'p90_px': limit_p90,
                            'jitter_px': limit_jitter, 'coverage': .80}
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
        report['failed_postures'] = [] if report['usable'] else [0]
        return report
