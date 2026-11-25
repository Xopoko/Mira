"""Replay saved gaze setup attempts through the current model.

Each attempt's setup points are fitted again and its check points are scored,
so a model change can be judged on recordings before anyone sits through a new
setup. Prints numbers only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_operator.foundations.gaze.arkit import ENGINE_VERSION
from context_operator.personalization.model import StarterProfile
from context_operator.personalization.session import STARTER_PROTOCOL
from context_operator.settings import LOCAL


def replay(path, pixel_mm=None):
    """Recorded and replayed error of one attempt, or None when it has no check."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    collection = data.get('collection', data)
    train, check = collection.get('train') or [], collection.get('check') or []
    if len(train) < StarterProfile.MIN_TARGETS or not check:
        return None
    metadata = {'display': collection['display'], 'engine': ENGINE_VERSION,
                'features': 22, 'protocol': STARTER_PROTOCOL}
    if pixel_mm is None:
        pixel_mm = collection.get('pixel_mm')
    profile = StarterProfile.fit([row['features'] for row in train], [row['eye_origins'] for row in train],
                                 [row['target'] for row in train], metadata, pixel_mm)
    errors = np.concatenate([np.linalg.norm(profile.predict(np.asarray(row['features'], dtype=float))-row['target'], axis=1)
                             for row in check])
    # A ray that misses the screen counts as a miss by the whole screen.
    errors = np.where(np.isfinite(errors), errors, np.hypot(*collection['display'][2:]))
    return {'saved_at': data.get('saved_at', ''), 'train': len(train), 'check': len(check),
            'recorded_median_px': (data.get('report') or {}).get('median_px'),
            'median_px': float(np.median(errors)), 'p90_px': float(np.percentile(errors, 90)),
            'viewing_cm': profile.settings['viewing_m']*100}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('folder', nargs='?', type=Path, default=LOCAL/'starter-attempts',
                        help='folder with saved attempts (default: this installation)')
    parser.add_argument('--pixel-mm', type=float,
                        help='size of one screen pixel in mm, for attempts that did not record it')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.folder.glob('*.json')):
        try:
            row = replay(path, args.pixel_mm)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'{path.stem[:10]}  skipped: {exc}')
            continue
        if row:
            rows.append((row['saved_at'], path.stem[:10], row))
    if not rows:
        print('No attempt with check points in', args.folder)
        return
    print('attempt     saved              points  recorded  now   p90  viewing')
    for saved, name, row in sorted(rows):
        recorded = '    --' if row['recorded_median_px'] is None else f"{row['recorded_median_px']:6.0f}"
        print(f"{name}  {saved[:16]:16s}  {row['train']:2d}+{row['check']:<2d}   {recorded}  {row['median_px']:4.0f}  {row['p90_px']:4.0f}  {row['viewing_cm']:4.0f} cm")
    print(f"median error now, mean over {len(rows)} attempts: {np.mean([row['median_px'] for _, _, row in rows]):.0f} px")


if __name__ == '__main__':
    main()
