"""Accepted personal profile with a verifiable commit receipt and rollback."""

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..settings import LOCAL
from .session import STARTER_PROTOCOL, SUPERSEDED_PROTOCOLS, StarterSession, _bounds
from .model import StarterProfile


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_report(protocol, report):
    return (protocol == STARTER_PROTOCOL and isinstance(report, dict) and
            report.get('quality') == 'starter' and report.get('usable') is True and
            report.get('passed') is False and
            report.get('scope') == StarterSession.result_scope and
            report.get('train_targets') == len(StarterSession.train_targets) and
            report.get('check_targets') == len(StarterSession.check_targets))


class ProfileUnavailable(RuntimeError):
    pass


class ProfileStore:
    def __init__(self, folder=LOCAL):
        self.folder = Path(folder)
        self.profile_path = self.folder/'gaze-personal.npz'
        self.receipt_path = self.folder/'personal-profile.json'
        self.profile_backup = self.folder/'gaze-personal.bak.npz'
        self.receipt_backup = self.folder/'personal-profile.bak.json'
        self.profile_superseded = self.folder/'gaze-personal.superseded.npz'
        self.receipt_superseded = self.folder/'personal-profile.superseded.json'

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
        profile = StarterProfile.load(profile_path, metadata)
        if (receipt.get('engine') != metadata.get('engine') or
                receipt.get('train_samples') != len(profile.training_x)):
            raise ValueError('Personal profile receipt does not match')
        return profile, receipt

    def _read(self, bounds):
        display = _bounds(bounds).tolist()
        for profile_path, receipt_path in ((self.profile_path, self.receipt_path),
                                           (self.profile_backup, self.receipt_backup)):
            try:
                profile, receipt = self._verified(profile_path, receipt_path, display)
                return profile, {'available': True, 'reason': '', 'path': profile_path,
                                 'profile_key': receipt['sha256'],
                                 'quality': receipt['report']['quality'],
                                 'report': receipt['report']}
            except (OSError, ValueError, KeyError, TypeError, AttributeError,
                    EOFError, zipfile.BadZipFile, UnicodeError):
                continue
        reason = ('Gaze setup is more accurate now. Run it again.' if self._superseded() else
                  'Your gaze profile is damaged or uses a different screen' if
                  self.profile_path.exists() or self.receipt_path.exists() else
                  'Set up your gaze first')
        return None, {'available': False, 'reason': reason, 'path': None,
                      'profile_key': None, 'quality': None, 'report': None}

    def _superseded(self):
        try:
            receipt = json.loads(self.receipt_path.read_text(encoding='utf-8'))
            return receipt.get('protocol') in SUPERSEDED_PROTOCOLS
        except (OSError, ValueError, AttributeError):
            return False

    def status(self, bounds):
        _, status = self._read(bounds)
        return status

    def load(self, bounds):
        profile, status = self._read(bounds)
        if profile is None:
            raise ProfileUnavailable(status['reason'])
        return profile, status

    @staticmethod
    def _atomic_bytes(path, content):
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def adopt(self, profile, report, bounds, collection_path=None, activate=None):
        """Commit a usable starter; recover the accepted pair on error."""
        display = _bounds(bounds).tolist()
        protocol = profile.metadata.get('protocol')
        if not _accepted_report(protocol, report) or profile.metadata.get('display') != display:
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
            StarterProfile.load(staged, profile.metadata)
            key = _sha(staged)
            receipt = {'version': 1, 'protocol': protocol, 'display': display,
                       'sha256': key, 'created_at': datetime.now(timezone.utc).isoformat(),
                       'engine': profile.metadata.get('engine'),
                       'train_samples': int(len(profile.training_x)),
                       'collection_sha256': _sha(Path(collection_path)) if collection_path else None,
                       'report': report}
            receipt_stage.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
            if incumbent['available'] and incumbent['path'] == self.profile_path:
                self._atomic_bytes(self.profile_backup, old_profile)
                self._atomic_bytes(self.receipt_backup, old_receipt)
            elif old_profile is not None and self._superseded():
                # This version cannot load the earlier pair; keep it beside the new one.
                self._atomic_bytes(self.profile_superseded, old_profile)
                self._atomic_bytes(self.receipt_superseded, old_receipt)
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
