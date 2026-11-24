"""Keep numerical setup attempts, including failures, available for diagnosis."""
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from .session import STARTER_PROTOCOL


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def preserve_attempt(archive, report=None):
    archive = Path(archive)
    if not archive.is_file():
        return None
    raw = archive.read_bytes()
    try:
        collection = json.loads(raw)
        if (not isinstance(collection, dict) or collection.get('protocol') != STARTER_PROTOCOL or
                not isinstance(collection.get('train'), list) or
                not isinstance(collection.get('check'), list) or
                not isinstance(collection.get('display'), list) or len(collection['display']) != 4):
            raise ValueError('Invalid collection')
    except (ValueError, UnicodeError):
        path = archive.parent/'starter-attempts'/('corrupt-'+hashlib.sha256(raw).hexdigest()+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
        return path
    if not collection.get('train'):
        return None
    digest = hashlib.sha256(raw).hexdigest()
    path = archive.parent/'starter-attempts'/f'{digest}.json'
    receipt = {'version': 1, 'saved_at': datetime.now(timezone.utc).isoformat(),
               'collection_sha256': digest, 'display': collection['display'],
               'report': report or collection.get('report'), 'collection': collection}
    if not path.exists() or report is not None:
        _write(path, receipt)
    if receipt['report'] is not None:
        _write(archive.parent/'starter-last-result.json',
               {key: value for key, value in receipt.items() if key != 'collection'})
    return path


def latest_result(folder, bounds):
    expected = bounds.as_list() if hasattr(bounds, 'as_list') else list(bounds)
    try:
        path = Path(folder)/'starter-last-result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        report = data['report']
        if data['version'] != 1 or data['display'] != expected or not isinstance(report, dict):
            return None
        if not all(isinstance(report.get(key), (int, float)) and math.isfinite(report[key])
                   for key in ('median_px', 'p90_px', 'coverage')):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return None
