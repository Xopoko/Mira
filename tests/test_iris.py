"""The iris position the iPhone measures: decoding, recording and the gaze model."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from context_operator.core.contracts import Rect
from context_operator.foundations.gaze.arkit import PacketDecoder
from context_operator.personalization import StarterSession, geometry, iris
from gaze_fixture import FRAME_MAP, FRAME_SHIFT, FRAME_SIZE, PIXEL_MM, iris_burst
from test_phone_wifi import packet

BOUNDS = Rect(0, 0, 1920, 1080)
ROOT = Path(__file__).resolve().parents[1]


def pixels(fractions):
    return np.asarray([[BOUNDS.w*x, BOUNDS.h*y] for x, y in fractions])


def eyes(targets, **seat):
    bursts = [iris_burst(target, BOUNDS, **seat) for target in targets]
    return iris.IrisEyes(*(np.concatenate(part) for part in zip(*bursts))), [len(burst[0]) for burst in bursts]


def measured(score=.9):
    return {'image': list(FRAME_SIZE), 'camera': [*FRAME_MAP.ravel(), *FRAME_SHIFT],
            'left': [700., 500., 11., score], 'right': [700., 620., 11., score]}


class IrisTests(unittest.TestCase):
    def test_decoder_keeps_a_complete_reading_and_drops_a_broken_one(self):
        decoder = PacketDecoder()
        self.assertEqual(decoder.decode(packet(1)).iris, ())
        one_eye = decoder.decode(dict(packet(2), iris=dict(measured(), right=None)))
        self.assertEqual(len(one_eye.iris), 16)
        self.assertFalse(iris.found([one_eye.iris])[0])
        self.assertTrue(iris.found([decoder.decode(dict(packet(3), iris=measured())).iris])[0])
        self.assertFalse(iris.found([decoder.decode(dict(packet(4), iris=measured(score=.1))).iris])[0])
        for seq, broken in enumerate(('yes', {'image': [0, 0]}, dict(measured(), camera=[1, 2]),
                                      dict(measured(), left=[1., 2., float('nan'), 1.])), 5):
            frame = decoder.decode(dict(packet(seq), iris=broken))
            self.assertEqual(frame.iris, ())
            self.assertEqual(frame.features.shape, (22,))

    def test_setup_lands_on_the_check_points_from_the_iris_whatever_the_eye_size(self):
        train, check = pixels(StarterSession.train_targets), pixels(StarterSession.check_targets)
        on_its_side = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        # One seat cannot tell a small eye from a far screen, and it does not need to.
        for seat in ({}, {'limbus_m': .0095, 'turn': .5}, {'limbus_m': .012, 'head': (.08, .03, -.6)},
                     {'camera': on_its_side, 'turn': .3}):
            fitted, counts = eyes(train, **seat)
            screen = geometry.fit_screen(fitted, counts, train, BOUNDS.as_list(), PIXEL_MM)
            looked, _ = eyes(check, **seat)
            landed = geometry.project_rays(*looked.rays(screen['eye']), np.asarray(BOUNDS.as_list(), dtype=float),
                                           screen['frame'], screen['center'], screen['pixel_m'])
            self.assertLess(np.linalg.norm(landed-np.repeat(check, 12, axis=0), axis=1).max(), 16., seat)

    def test_setup_records_the_iris_with_each_fixation_and_replays_both_signals(self):
        spec = importlib.util.spec_from_file_location('evaluate_gaze_setup', ROOT/'scripts'/'evaluate_gaze_setup.py')
        evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'starter.json'
            session = StarterSession(BOUNDS, archive, PIXEL_MM)
            seq, stamp = 0, 100.

            def collect(with_iris=True):
                nonlocal seq, stamp
                features, origins, readings = iris_burst(session.target, BOUNDS)
                for frame in range(36):
                    seq += 1
                    captured = stamp+1.+frame*.05
                    session.feed(SimpleNamespace(seq=seq, captured=captured, features=features[0], eye_origins=origins[0],
                                                 iris=tuple(readings[0]) if with_iris else (), error='', blink=False),
                                 captured+.005)
                    if session.ready_to_confirm and session.confirm(captured+.01):
                        stamp = captured+.5
                        return
                self.fail(session.hold_reason)

            collect(with_iris=False)
            self.assertNotIn('iris', session.train[0])
            session = StarterSession(BOUNDS, archive, PIXEL_MM)
            for _ in session.train_targets:
                collect()
            session = StarterSession.resume(BOUNDS, archive, PIXEL_MM)
            self.assertEqual(np.shape(session.train[-1]['iris']), (12, 16))
            session.begin_check(session.fit(), stamp)
            for _ in session.check_targets:
                collect()
            result = evaluate.replay(archive)
            self.assertLess(result['arkit']['median_px'], 12.)
            self.assertLess(result['iris']['median_px'], 12.)
            self.assertEqual(result['iris']['seen'], 1.)

            saved = json.loads(archive.read_text(encoding='utf-8'))
            for row in saved['train']:
                row.pop('iris')
            archive.write_text(json.dumps(saved), encoding='utf-8')
            self.assertNotIn('iris', evaluate.replay(archive))


if __name__ == '__main__':
    unittest.main()
