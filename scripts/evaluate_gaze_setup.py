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


def fit(train, display, pixel_mm):
    metadata = {'display': list(display), 'engine': ENGINE_VERSION, 'features': 22, 'protocol': STARTER_PROTOCOL}
    return StarterProfile.fit([row['features'] for row in train], [row['eye_origins'] for row in train],
                              [row['target'] for row in train], metadata, pixel_mm)


def errors(profile, rows, display):
    """Error of every frame, and how far each fixation as a whole landed from its target."""
    frame, point = [], []
    # A ray that misses the screen counts as a miss by the whole screen.
    miss = np.hypot(*display[2:])
    for row in rows:
        pixels = profile.predict(np.asarray(row['features'], dtype=float))
        error = np.linalg.norm(pixels-row['target'], axis=1)
        frame.append(np.where(np.isfinite(error), error, miss))
        # What is left once jitter is smoothed away.
        point.append(np.linalg.norm(np.nanmedian(pixels, axis=0)-row['target']) if np.isfinite(pixels).any() else miss)
    return np.concatenate(frame), np.asarray(point)


def replay(path, pixel_mm=None, leave_one_out=False):
    """Recorded error of one attempt and its error now, or None when it has no check."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    collection = data.get('collection', data)
    train, check = collection.get('train') or [], collection.get('check') or []
    if len(train) < StarterProfile.MIN_TARGETS or not check:
        return None
    if pixel_mm is None:
        pixel_mm = collection.get('pixel_mm')
    display = collection['display']
    profile = fit(train, display, pixel_mm)
    frame, point = errors(profile, check, display)
    result = {'saved_at': data.get('saved_at', ''), 'train': len(train), 'check': len(check),
              'recorded_median_px': (data.get('report') or {}).get('median_px'),
              'median_px': float(np.median(frame)), 'p90_px': float(np.percentile(frame, 90)),
              'point_px': float(np.median(point)), 'viewing_cm': profile.settings['viewing_m']*100}
    if leave_one_out:
        # Five check points are few. Holding out every fixation in turn uses them all.
        rows = train+check
        held = [errors(fit(rows[:index]+rows[index+1:], display, pixel_mm), rows[index:index+1], display)[1]
                for index in range(len(rows))]
        result['held_out_px'] = float(np.median(np.concatenate(held)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('folder', nargs='?', type=Path, default=LOCAL/'starter-attempts',
                        help='folder with saved attempts (default: this installation)')
    parser.add_argument('--pixel-mm', type=float,
                        help='size of one screen pixel in mm, for attempts that did not record it')
    parser.add_argument('--leave-one-out', action='store_true',
                        help='also hold out every fixation in turn (slow, steadier than five check points)')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.folder.glob('*.json')):
        try:
            row = replay(path, args.pixel_mm, args.leave_one_out)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'{path.stem[:10]}  skipped: {exc}')
            continue
        if row:
            rows.append((row['saved_at'], path.stem[:10], row))
    if not rows:
        print('No attempt with check points in', args.folder)
        return
    rows.sort(key=lambda row: row[:2])
    # now and p90 are over frames; point is where whole fixations landed, held the same with each fixation held out.
    print('attempt     saved              points  recorded    now    p90  point'+('   held' if args.leave_one_out else '')+'  viewing')
    for saved, name, row in rows:
        recorded = '    --' if row['recorded_median_px'] is None else f"{row['recorded_median_px']:6.0f}"
        held = f"  {row['held_out_px']:5.0f}" if args.leave_one_out else ''
        print(f"{name}  {saved[:16]:16s}  {row['train']:2d}+{row['check']:<2d}   {recorded}  {row['median_px']:5.0f}  "
              f"{row['p90_px']:5.0f}  {row['point_px']:5.0f}{held}  {row['viewing_cm']:4.0f} cm")
    print(f"median error now, mean over {len(rows)} attempts: {np.mean([row['median_px'] for _, _, row in rows]):.0f} px")
    if args.leave_one_out:
        print(f"held-out fixations, mean over {len(rows)} attempts: "
              f"{np.mean([row['held_out_px'] for _, _, row in rows]):.0f} px")


if __name__ == '__main__':
    main()
