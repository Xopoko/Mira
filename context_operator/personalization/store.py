"""Accepted personal profile with a verifiable commit receipt and rollback."""

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..foundations.gaze.profile import Profile
from ..settings import LOCAL
from .session import PROTOCOL, STARTER_PROTOCOL
from .model import PersonalProfile, StarterProfile


def _load_model(path, metadata):
    with np.load(path, allow_pickle=False) as saved:
        kind = json.loads(str(saved['settings'])).get('model_kind')
    if kind is None:
        return Profile.load(path, metadata)
    if kind == PersonalProfile.MODEL_KIND:
        return PersonalProfile.load(path, metadata)
    if kind == StarterProfile.MODEL_KIND:
        return StarterProfile.load(path, metadata)
    raise ValueError('Unknown personal model format')


def _bounds(bounds):
    value = bounds.as_list() if hasattr(bounds, 'as_list') else bounds
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError('Invalid display')
    return result.tolist()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_report(protocol, report):
    if not isinstance(report, dict):
        return False
    if protocol == PROTOCOL:
        return (report.get('passed') is True and
                report.get('scope') == 'same-session-independent-targets' and
                report.get('train_targets') == 12 and report.get('check_targets') == 27)
    if protocol == STARTER_PROTOCOL:
        return (report.get('quality') == 'starter' and report.get('usable') is True and
                report.get('passed') is False and
                report.get('scope') == 'starter-independent-targets' and
                report.get('train_targets') == 9 and report.get('check_targets') == 5)
    return False


class ProfileStore:
    def __init__(self, folder=LOCAL):
        self.folder = Path(folder)
        self.profile_path = self.folder/'gaze-personal.npz'
        self.receipt_path = self.folder/'personal-profile.json'
        self.profile_backup = self.folder/'gaze-personal.bak.npz'
        self.receipt_backup = self.folder/'personal-profile.bak.json'

    @staticmethod
    def _verified(profile_path, receipt_path, display):
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        protocol = receipt.get('protocol')
        report = receipt.get('report')
        if (receipt.get('version') != 1 or not _accepted_report(protocol, report) or
                receipt.get('display') != display or
                not isinstance(receipt.get('sha256'), str) or
                len(receipt['sha256']) != 64 or
                _sha(profile_path) != receipt['sha256']):
            raise ValueError('Personal profile receipt does not match')
        try:
            p90 = float(receipt['report']['p90_px'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Personal validation metric is missing') from exc
        if not np.isfinite(p90) or p90 < 0:
            raise ValueError('Personal validation metric is invalid')
        with np.load(profile_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
        if metadata.get('display') != display or metadata.get('protocol') != protocol:
            raise ValueError('Personal profile belongs to another display')
        profile = _load_model(profile_path, metadata)
        if protocol == STARTER_PROTOCOL and not isinstance(profile, StarterProfile):
            raise ValueError('Starter profile has the wrong model type')
        if (profile.training_x.shape[1] != 22 or
                receipt.get('engine') != metadata.get('engine') or
                receipt.get('model') != profile.settings.get('model') or
                receipt.get('train_samples') != len(profile.training_x)):
            raise ValueError('Personal profile feature count differs')
        return profile, receipt

    def _read(self, bounds):
        display = _bounds(bounds)
        for profile_path, receipt_path in ((self.profile_path, self.receipt_path),
                                           (self.profile_backup, self.receipt_backup)):
            try:
                profile, receipt = self._verified(profile_path, receipt_path, display)
                return profile, {'available': True, 'reason': '', 'path': profile_path,
                                 'profile_key': receipt['sha256'], 'method': 'personal',
                                 'quality': receipt['report'].get('quality', 'full'),
                                 'report': receipt['report']}
            except (OSError, ValueError, KeyError, TypeError, AttributeError,
                    EOFError, zipfile.BadZipFile, UnicodeError):
                continue
        reason = ('Your gaze profile is damaged or uses a different screen' if
                  self.profile_path.exists() or self.receipt_path.exists() else
                  'Set up your gaze first')
        return None, {'available': False, 'reason': reason, 'path': None,
                      'profile_key': None, 'method': 'personal', 'quality': None, 'report': None}

    def status(self, bounds):
        _, status = self._read(bounds)
        return status

    def load(self, bounds):
        profile, status = self._read(bounds)
        if profile is None:
            raise RuntimeError(status['reason'])
        return profile, status

    @staticmethod
    def _atomic_bytes(path, content):
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def adopt(self, profile, report, bounds, collection_path=None, activate=None):
        """Commit a full pass or usable starter; recover incumbent on error."""
        display = _bounds(bounds)
        protocol = profile.metadata.get('protocol')
        if (not _accepted_report(protocol, report) or
                profile.metadata.get('display') != display or
                (protocol == STARTER_PROTOCOL and not isinstance(profile, StarterProfile)) or
                profile.training_x.shape[1] != 22):
            raise ValueError('Candidate has no matching independent acceptance')
        self.folder.mkdir(parents=True, exist_ok=True)
        staged = self.folder/'gaze-personal.staged.npz'
        receipt_stage = self.folder/'personal-profile.staged.json'
        old_profile = self.profile_path.read_bytes() if self.profile_path.exists() else None
        old_receipt = self.receipt_path.read_bytes() if self.receipt_path.exists() else None
        old_backup_profile = self.profile_backup.read_bytes() if self.profile_backup.exists() else None
        old_backup_receipt = self.receipt_backup.read_bytes() if self.receipt_backup.exists() else None
        incumbent = self.status(bounds)
        try:
            profile.save(staged)
            # Verify the exact bytes that will become active, without pickle.
            _load_model(staged, profile.metadata)
            key = _sha(staged)
            receipt = {'version': 1, 'protocol': protocol, 'display': display,
                       'sha256': key, 'created_at': datetime.now(timezone.utc).isoformat(),
                       'engine': profile.metadata.get('engine'),
                       'model': profile.settings.get('model'),
                       'train_samples': int(len(profile.training_x)),
                       'training_cv_by_target': profile.settings.get('cv_by_target', []),
                       'collection_sha256': _sha(Path(collection_path)) if collection_path else None,
                       'report': report}
            receipt_stage.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
            if incumbent['available'] and incumbent['path'] == self.profile_path:
                self._atomic_bytes(self.profile_backup, old_profile)
                self._atomic_bytes(self.receipt_backup, old_receipt)
            os.replace(staged, self.profile_path)
            os.replace(receipt_stage, self.receipt_path)
            self._verified(self.profile_path, self.receipt_path, display)
            if activate is not None:
                activate()
            return key
        except Exception:
            if old_profile is None:
                self.profile_path.unlink(missing_ok=True)
            else:
                self._atomic_bytes(self.profile_path, old_profile)
            if old_receipt is None:
                self.receipt_path.unlink(missing_ok=True)
            else:
                self._atomic_bytes(self.receipt_path, old_receipt)
            if old_backup_profile is None:
                self.profile_backup.unlink(missing_ok=True)
            else:
                self._atomic_bytes(self.profile_backup, old_backup_profile)
            if old_backup_receipt is None:
                self.receipt_backup.unlink(missing_ok=True)
            else:
                self._atomic_bytes(self.receipt_backup, old_backup_receipt)
            raise
        finally:
            staged.unlink(missing_ok=True)
            receipt_stage.unlink(missing_ok=True)


def personal_status(bounds, folder=LOCAL):
    return ProfileStore(folder).status(bounds)
