"""Replay saved gaze setup attempts through the current model.

Each attempt's setup points are fitted again and its check points are scored,
so a model change can be judged on recordings before anyone sits through a new
setup. An attempt recorded with an iPhone that measures the iris itself is
scored twice, once per eye signal, on the same fixations. Prints numbers only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_operator.foundations.gaze.arkit import ENGINE_VERSION
from context_operator.personalization import geometry, iris
from context_operator.personalization.model import StarterProfile, add_field, leftover_field
from context_operator.personalization.session import STARTER_PROTOCOL
from context_operator.settings import LOCAL


def frames(row, key):
    return np.asarray(row[key], dtype=float)


def arkit(train, display, pixel_mm):
    """ARKit's eye rotation through the model Mira uses."""
    metadata = {'display': list(display), 'engine': ENGINE_VERSION, 'features': 22, 'protocol': STARTER_PROTOCOL}
    profile = StarterProfile.fit([row['features'] for row in train], [row['eye_origins'] for row in train],
                                 [row['target'] for row in train], metadata, pixel_mm)
    return (lambda row: (profile.predict(frames(row, 'features')), 1.)), profile.settings['viewing_m']


def measured_iris(train, display, pixel_mm):
    """The iris position the phone measures, through the same screen fit."""
    def eyes(rows):
        return iris.IrisEyes(*(np.concatenate([frames(row, key)[keep] for row, keep in rows])
                               for key in ('features', 'eye_origins', 'iris')))

    kept = [(row, iris.found(frames(row, 'iris'))) for row in train]
    kept = [(row, keep) for row, keep in kept if keep.sum() >= 2]
    if len(kept) < StarterProfile.MIN_TARGETS:
        raise ValueError('both irises were found at too few setup points')
    display = np.asarray(display, dtype=float)
    targets = np.asarray([row['target'] for row, _ in kept], dtype=float)
    screen = geometry.fit_screen(eyes(kept), [int(keep.sum()) for _, keep in kept], targets, display, pixel_mm)

    def plane(rows):
        return geometry.project_rays(*eyes(rows).rays(screen['eye']), display,
                                     screen['frame'], screen['center'], screen['pixel_m'])

    # The same smooth field through the leftover error that the profile adds.
    landed = np.asarray([np.nanmedian(plane([fixation]), axis=0) for fixation in kept])
    reach = StarterProfile.CORRECTION_REACH*display[3]
    weights = leftover_field(landed, targets, reach)

    def predict(row):
        keep = iris.found(frames(row, 'iris'))
        if not keep.any():
            return np.empty((0, 2)), 0.
        return add_field(plane([(row, keep)]), landed, weights, reach), float(keep.mean())

    return predict, screen['viewing_m']


def errors(predict, rows, display):
    """Per frame and per fixation error, and the share of frames the signal could use."""
    frame, point, seen = [], [], []
    for row in rows:
        pixels, share = predict(row)
        seen.append(share)
        if not len(pixels):
            continue
        # A ray that misses the screen counts as a miss by the whole screen.
        miss = np.hypot(*display[2:])
        error = np.linalg.norm(pixels-row['target'], axis=1)
        frame.append(np.where(np.isfinite(error), error, miss))
        # Where the fixation as a whole landed: what is left once jitter is smoothed away.
        landed = np.linalg.norm(np.nanmedian(pixels, axis=0)-row['target']) if np.isfinite(pixels).any() else miss
        point.append(landed)
    return np.concatenate(frame or [np.empty(0)]), np.asarray(point), float(np.mean(seen))


def score(model, train, check, display, pixel_mm, leave_one_out):
    predict, viewing = model(train, display, pixel_mm)
    frame, point, seen = errors(predict, check, display)
    if not len(frame):
        raise ValueError('no check frame could be used')
    result = {'median_px': float(np.median(frame)), 'p90_px': float(np.percentile(frame, 90)),
              'point_px': float(np.median(point)), 'seen': seen, 'viewing_cm': viewing*100}
    if leave_one_out:
        # Five check points are few. Holding out every fixation in turn uses them all.
        rows = train+check
        held = [errors(model(rows[:index]+rows[index+1:], display, pixel_mm)[0], rows[index:index+1], display)[1]
                for index in range(len(rows))]
        result['held_out_px'] = float(np.median(np.concatenate(held)))
    return result


def replay(path, pixel_mm=None, leave_one_out=False):
    """Recorded error of one attempt and its error now per eye signal, or None when it has no check."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    collection = data.get('collection', data)
    train, check = collection.get('train') or [], collection.get('check') or []
    if len(train) < StarterProfile.MIN_TARGETS or not check:
        return None
    if pixel_mm is None:
        pixel_mm = collection.get('pixel_mm')
    result = {'saved_at': data.get('saved_at', ''), 'train': len(train), 'check': len(check),
              'recorded_median_px': (data.get('report') or {}).get('median_px'),
              'arkit': score(arkit, train, check, collection['display'], pixel_mm, leave_one_out)}
    if all('iris' in row for row in train+check):
        try:
            result['iris'] = score(measured_iris, train, check, collection['display'], pixel_mm, leave_one_out)
        except ValueError as exc:
            result['iris_skipped'] = str(exc)
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
    held = '  held' if args.leave_one_out else ''

    def cells(result):
        if result is None:
            return '   --    --    --'+('    --' if args.leave_one_out else '')
        return f"{result['median_px']:5.0f} {result['p90_px']:5.0f} {result['point_px']:5.0f}"+(
            f"  {result['held_out_px']:4.0f}" if args.leave_one_out else '')

    # median and p90 are over frames; point is where whole fixations landed, held the same with each fixation held out.
    print(f'attempt     saved              points  recorded  ARKit   p90 point{held}   iris   p90 point{held}  seen  viewing')
    for saved, name, row in rows:
        recorded = '    --' if row['recorded_median_px'] is None else f"{row['recorded_median_px']:6.0f}"
        seen = f"{row['iris']['seen']:4.0%}" if 'iris' in row else '  --'
        print(f"{name}  {saved[:16]:16s}  {row['train']:2d}+{row['check']:<2d}   {recorded}  {cells(row['arkit'])}  "
              f"{cells(row.get('iris'))}  {seen}  {row['arkit']['viewing_cm']:4.0f} cm")
        if 'iris_skipped' in row:
            print(f"{name}  iris skipped: {row['iris_skipped']}")
    print(f"ARKit, median error now, mean over {len(rows)} attempts: "
          f"{np.mean([row['arkit']['median_px'] for _, _, row in rows]):.0f} px")
    both = [row for _, _, row in rows if 'iris' in row]
    if both:
        key = 'held_out_px' if args.leave_one_out else 'point_px'
        print(f"Same {len(both)} attempts with both signals, where fixations landed"
              f"{' when held out' if args.leave_one_out else ''}: "
              f"ARKit {np.mean([row['arkit'][key] for row in both]):.0f} px, "
              f"iris {np.mean([row['iris'][key] for row in both]):.0f} px")


if __name__ == '__main__':
    main()
