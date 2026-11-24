import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QWidget

from context_operator.core.contracts import Rect
from context_operator.personalization import StarterSession, ProfileStore
from context_operator.foundations.gaze.arkit import Sample
from context_operator.providers.gaze import PhoneGaze
from context_operator.personalization.session import MIN_DURATION, MIN_FRAMES
from context_operator.ui.setup import SetupCanvas, SetupWindow, open_setup
from context_operator.ui.webcam_setup import open_webcam_setup
from test_starter import feature_for, starter_profile


APP = QApplication.instance() or QApplication([])
BOUNDS = Rect(0, 0, 1920, 1080)


def sample(seq, stamp, target, depth=-.7):
    features = np.zeros(22)
    features[:2] = target
    features[6] = depth
    features[13:19] = [0, 0, 1, 0, 0, 1]
    origins = ((0., 0., depth), (.06, 0., depth))
    return Sample(seq=seq, captured=stamp, features=features, eye_origins=origins)


class DirectModel:
    def predict(self, x):
        return np.asarray(x)[..., :2]

    def support_reason(self, x):
        return ''


def accepted_report():
    return {'passed': False, 'usable': True, 'quality': 'starter', 'scope': 'starter-independent-targets',
            'train_targets': 9, 'check_targets': 5,
            'median_px': 42., 'p90_px': 93., 'coverage': 1.}


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_starter_result_guide_fits_short_window_after_result_growth(self):
        controller = SimpleNamespace(win=SimpleNamespace(bounds=lambda: BOUNDS),
                                     settings={'phone_transport': 'usb'})
        window = SetupWindow(controller, archive=self.folder/'starter.json',
                             store=ProfileStore(self.folder))
        try:
            window.resize(960, 680)
            window.session.phase = 'result'
            window.session.report = {'quality': 'starter', 'usable': True, 'passed': False,
                                     'median_px': 330., 'p90_px': 690., 'jitter_px': 30.,
                                     'coverage': .95, 'bias_px': [2., -4.], 'problems': []}
            window.refresh()
            window.show()
            APP.processEvents()
            top = window.heading.mapTo(window, QPoint(0, 0)).y()
            self.assertGreaterEqual(top, 0)
            for button in (window.connect_button, window.begin_button, window.retry_button,
                           window.apply_button, window.cancel_button):
                if button.isVisible():
                    origin = button.mapTo(window, QPoint(0, 0))
                    self.assertGreaterEqual(origin.x(), 0)
                    self.assertGreaterEqual(origin.y(), 0)
                    self.assertLessEqual(origin.x()+button.width(), window.width())
                    self.assertLessEqual(origin.y()+button.height(), window.height())
            self.assertIn('Gaze setup ready', window.guide_heading.text())
        finally:
            window.close()

    def test_stale_blink_and_duplicate_frames_do_not_complete_target(self):
        session = StarterSession(BOUNDS)
        target = session.target
        for seq in range(1, 12):
            stamp = 100+seq*.07
            session.feed(sample(seq, stamp, target), stamp+.01)
            self.assertFalse(session.ready_to_confirm)
        session.feed(sample(12, 100.84, target), 101.5)
        self.assertFalse(session.ready_to_confirm)
        self.assertEqual(session.index, 0)
        session.feed(sample(12, 101.6, target), 101.61)
        self.assertFalse(session.ready_to_confirm)
        session.feed(sample(12, 101.6, target), 101.62)
        self.assertFalse(session.ready_to_confirm)
        self.assertEqual(len(session.frames), 1)
        blink = sample(13, 101.67, target)
        blink.blink = True
        session.feed(blink, 101.68)
        self.assertFalse(session.ready_to_confirm)
        self.assertEqual(session.index, 0)
        self.assertEqual(session.frames, [])

    def test_corrupt_archive_fails_closed(self):
        path = self.folder/'session.json'
        path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            StarterSession.resume(BOUNDS, path)
        self.assertEqual(path.read_text(encoding='utf-8'), '{broken')

    def test_check_uses_unseen_targets_and_measures_raw_errors(self):
        session = StarterSession(BOUNDS)
        self.assertFalse(set(session.train_targets) & set(session.check_targets))
        session.phase = 'fit'
        session.begin_check(DirectModel(), 100.)
        for index, fraction in enumerate(session.check_targets):
            target = [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h]
            x = np.zeros((12, 22))
            x[:, :2] = target
            session.check.append({'phase': 'check', 'index': index,
                                  'target': target, 'features': x.tolist(),
                                  'eye_origins': np.zeros((12, 2, 3)).tolist()})
        session.phase = 'result'
        report = session.evaluate()
        self.assertTrue(report['usable'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['median_px'], 0)
        self.assertEqual(report['p90_px'], 0)
        self.assertEqual(report['coverage'], 1)
        self.assertEqual(report['check_targets'], 5)
        for frame in session.check[0]['features'][:5]:
            frame[0] += 1000
        failed = session.evaluate()
        self.assertFalse(failed['usable'])
        # Global P90 must describe the individual held-out predictions; a bad
        # target still fails its separate local gate even below 10% globally.
        self.assertEqual(failed['p90_px'], 0.)
        self.assertGreater(failed['targets'][0]['p90_px'], 900.)

    def test_overall_p90_has_its_own_problem_when_all_targets_meet_the_limit(self):
        session = StarterSession(BOUNDS)
        session.phase = 'result'
        session.candidate = DirectModel()
        for fraction in session.check_targets:
            target = [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h]
            x = np.zeros((12, 22))
            x[:, :2] = target
            x[-2:, 0] += 580. if fraction[0] < .5 else -580.
            session.check.append({'target': target, 'features': x.tolist()})
        report = session.evaluate()
        limit = report['limits']['p90_px']
        self.assertGreater(report['p90_px'], limit)
        self.assertTrue(all(row['p90_px'] <= limit for row in report['targets']))
        self.assertEqual(report['coverage'], 1.)
        self.assertFalse(report['usable'])
        self.assertEqual(report['problems'], ['90th percentile error 580 px · limit 540 px.'])

    def test_store_checks_receipt_and_rolls_back_failed_adoption(self):
        store = ProfileStore(self.folder)
        old = starter_profile()
        key = store.adopt(old, accepted_report(), BOUNDS)
        self.assertEqual(store.status(BOUNDS)['profile_key'], key)
        before_profile = store.profile_path.read_bytes()
        before_receipt = store.receipt_path.read_bytes()
        new = starter_profile(offset=90)
        original_replace = os.replace
        triggered = False

        def fail_receipt(source, destination):
            nonlocal triggered
            if Path(destination) == store.receipt_path and not triggered:
                triggered = True
                raise OSError('simulated failed commit')
            return original_replace(source, destination)

        with patch('context_operator.personalization.store.os.replace', side_effect=fail_receipt):
            with self.assertRaises(OSError):
                store.adopt(new, accepted_report(), BOUNDS)
        self.assertEqual(store.profile_path.read_bytes(), before_profile)
        self.assertEqual(store.receipt_path.read_bytes(), before_receipt)
        self.assertEqual(store.status(BOUNDS)['profile_key'], key)
        store.adopt(new, accepted_report(), BOUNDS)
        store.profile_path.write_bytes(b'broken')
        self.assertTrue(store.status(BOUNDS)['available'])  # Valid backup remains usable.
        self.assertEqual(store.status(BOUNDS)['quality'], 'starter')
        loaded, status = store.load(BOUNDS)
        self.assertEqual(status['profile_key'], key)
        np.testing.assert_array_equal(loaded.training_y, old.training_y)
        self.assertEqual(loaded.metadata['protocol'], old.metadata['protocol'])
        self.assertFalse(status['report']['passed'])
        store.profile_backup.write_bytes(b'broken too')
        self.assertFalse(store.status(BOUNDS)['available'])

    def test_activation_failure_rolls_back_profile(self):
        store = ProfileStore(self.folder)
        key = store.adopt(starter_profile(), accepted_report(), BOUNDS)
        def fail_activation():
            raise RuntimeError('profile activation failed')
        with self.assertRaises(RuntimeError):
            store.adopt(starter_profile(offset=30), accepted_report(), BOUNDS,
                        activate=fail_activation)
        self.assertEqual(store.status(BOUNDS)['profile_key'], key)

    def test_rejects_unqualified_and_screen_mismatch(self):
        store = ProfileStore(self.folder)
        for invalid in (dict(accepted_report(), usable=False), dict(accepted_report(), passed=True)):
            with self.assertRaises(ValueError):
                store.adopt(starter_profile(), invalid, BOUNDS)
        self.assertFalse(store.status(BOUNDS)['available'])
        store.adopt(starter_profile(), accepted_report(), BOUNDS)
        self.assertFalse(store.status(Rect(0, 0, 1280, 720))['available'])

    def test_phone_provider_loads_accepted_starter_profile(self):
        store = ProfileStore(self.folder)
        key = store.adopt(starter_profile(), accepted_report(), BOUNDS)
        with patch('context_operator.providers.gaze.LOCAL', self.folder):
            phone = PhoneGaze(BOUNDS)
        self.assertEqual(phone.profile_key, key)
        self.assertEqual(phone.feature_scale.shape, (22,))
        self.assertEqual(phone.validation_radius, 140.)

    def test_offscreen_window_starts_without_old_profile_and_cancel_stops_worker(self):
        controller = SimpleNamespace(win=SimpleNamespace(bounds=lambda: BOUNDS),
                                     settings={'phone_transport': 'usb'})
        worker = Mock()
        worker.latest.return_value = Sample()
        worker.is_alive.return_value = False
        window = SetupWindow(controller, archive=self.folder/'session.json',
                             store=ProfileStore(self.folder), worker_factory=lambda *a, **k: worker)
        try:
            self.assertEqual(window.session.phase, 'train')
            self.assertFalse(window.begin_button.isEnabled())
            window.connect_phone()
            self.assertFalse(window.begin_button.isEnabled())
        finally:
            window.close()
        worker.stop_event.set.assert_called_once()
        worker.join.assert_called_once()

    def test_result_can_be_applied_only_after_the_checked_candidate_is_saved(self):
        controller = SimpleNamespace(win=SimpleNamespace(bounds=lambda: BOUNDS), settings={})
        store = Mock()
        window = SetupWindow(controller, archive=self.folder/'starter.json', store=store)
        self.addCleanup(window.close)
        candidate = Mock()
        candidate.save.side_effect = lambda path: path.write_bytes(b'candidate')
        window.session.phase = 'result'
        window.candidate = window.session.candidate = candidate
        report = dict(accepted_report(), quality='starter', usable=True, passed=False,
                      jitter_px=12., bias_px=[0., 0.])

        with patch.object(window.session, 'evaluate', return_value=report):
            with patch.object(window.session, '_save', side_effect=OSError('attempt write failed')):
                window._target_completed()
            self.assertTrue(window.apply_button.isHidden())
            self.assertEqual(window.begin_button.text(), 'Save result')
            window.adopt()
            store.adopt.assert_not_called()

            with patch('context_operator.ui.setup.preserve_attempt', side_effect=OSError('archive write failed')):
                window.begin()
            self.assertIn('Result saved', window.error)
            self.assertFalse(window.apply_button.isHidden())

    def test_screen_gap_rejected_before_pausing_controller(self):
        first = SimpleNamespace(geometry=lambda: QRect(0, 0, 800, 1080),
                                devicePixelRatio=lambda: 1.)
        second = SimpleNamespace(geometry=lambda: QRect(1120, 0, 800, 1080),
                                 devicePixelRatio=lambda: 1.)
        controller = SimpleNamespace(busy=False, output_pending=False,
                                     settings={'webcam_device_id': 'fixture'},
                                     win=SimpleNamespace(bounds=lambda: BOUNDS), pause=Mock())
        with patch('context_operator.ui.setup.QApplication.screens', return_value=[first, second]):
            for open_window in (open_setup, open_webcam_setup):
                with self.subTest(open_window=open_window.__name__):
                    with self.assertRaisesRegex(RuntimeError, 'falls between screens'):
                        open_window(controller)
        with patch('context_operator.ui.setup.QApplication.screens', return_value=[first]):
            for open_window in (open_setup, open_webcam_setup):
                with self.subTest(layout_mismatch=open_window.__name__):
                    with self.assertRaisesRegex(RuntimeError, 'Check Windows scaling and run setup again'):
                        open_window(controller)
        controller.pause.assert_not_called()

    def test_setup_ring_requires_enough_frames_and_steady_fixation(self):
        owner = QWidget()
        owner.running = True
        owner.session = StarterSession(BOUNDS)
        owner._ready = lambda: True
        canvas = SetupCanvas(owner)
        canvas.resize(640, 480)
        with patch('context_operator.ui.setup.QPainter') as painter:
            frames = [Sample(captured=t) for t in np.linspace(0., MIN_DURATION+.01, MIN_FRAMES)]
            owner.session.frames = [frames[0], frames[-1]]
            canvas.paintEvent(None)
            arc = painter.return_value.drawArc
            self.assertEqual(arc.call_args.args[-1], -int(2/MIN_FRAMES*360*16))
            owner.session.frames = frames
            canvas.paintEvent(None)
            self.assertGreater(arc.call_args.args[-1], -360*16)
            owner.session.ready_to_confirm = True
            canvas.paintEvent(None)
            self.assertEqual(arc.call_args.args[-1], -360*16)

    def test_disconnect_exposes_reconnect_and_resumes_current_target(self):
        controller = SimpleNamespace(win=SimpleNamespace(bounds=lambda: BOUNDS),
                                     settings={'phone_transport': 'usb'})
        worker = Mock()
        worker.is_alive.return_value = True
        worker.latest.return_value = sample(25, 100., [192, 130])
        window = SetupWindow(controller, archive=self.folder/'session.json',
                             worker_factory=lambda *a, **k: worker)
        window.timer.stop()
        try:
            with patch('context_operator.ui.setup.time.monotonic', return_value=100.01):
                window.connect_phone()
                window.begin()
                self.assertTrue(window.running)
                worker.latest.return_value = Sample(error='Connection closed')
                worker.is_alive.return_value = False
                window.tick()
                self.assertFalse(window.running)
                self.assertFalse(window.connect_button.isHidden())
                self.assertEqual(window.session.index, 0)
                window.session.last_seq = 25
                window.connect_phone()
                self.assertEqual(window.session.last_seq, -1)
                worker.is_alive.return_value = True
                worker.latest.return_value = sample(1, 100., [192, 130])
                window.begin()
                self.assertTrue(window.running)
                self.assertEqual(window.error, '')
        finally:
            worker.is_alive.return_value = False
            window.close()

    def test_failed_point_save_preserves_target_and_prior_archive(self):
        path = self.folder/'session.json'
        session = StarterSession(BOUNDS, path)
        session._save()
        before = path.read_bytes()
        for seq in range(1, 20):
            stamp = 100+seq*.07
            session.feed(sample(seq, stamp, session.target), stamp+.005)
        self.assertTrue(session.ready_to_confirm)
        with patch.object(session, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                session.confirm(stamp+.01)
        self.assertEqual(session.index, 0)
        self.assertEqual(session.train, [])
        self.assertEqual(path.read_bytes(), before)

    def test_explicit_confirmation_is_required_and_cannot_accept_stale_fixation(self):
        session = StarterSession(BOUNDS)
        for seq in range(1, 30):
            stamp = 100+seq*.07
            self.assertIsNone(session.feed(sample(seq, stamp, session.target), stamp+.005))
        self.assertTrue(session.ready_to_confirm)
        self.assertEqual(session.index, 0)
        self.assertFalse(session.confirm(stamp+.25))
        self.assertTrue(session.confirm(stamp+.01))
        self.assertEqual(session.index, 1)
        self.assertEqual(len(session.train), 1)
        self.assertEqual(session.train[0]['index'], 0)

    def test_corrupt_archive_can_be_replaced_without_losing_bad_bytes(self):
        path=self.folder/'session.json';path.write_bytes(b'{broken')
        controller=SimpleNamespace(win=SimpleNamespace(bounds=lambda:BOUNDS),settings={'phone_transport':'usb'})
        window=SetupWindow(controller,archive=path,store=ProfileStore(self.folder))
        try:
            window.restart()
            self.assertEqual(window.error,'')
            self.assertEqual(window.session.phase,'train')
            saved=list((self.folder/'starter-attempts').glob('corrupt-*.json'))
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0].read_bytes(),b'{broken')
        finally:window.close()

    def test_first_use_60hz_training_30hz_check_and_adoption(self):
        """No old profile: raw frames -> fit -> unseen check -> active provider."""
        session = StarterSession(BOUNDS, self.folder/'session.json')
        seq, stamp = 0, 100.

        def collect_target(hz, depth):
            nonlocal seq, stamp
            target = session.target
            advanced = False
            for frame in range(int(hz*1.2)):
                seq += 1
                current = stamp + 1. + frame/hz
                observation = sample(seq, current, target, depth)
                observation.features = feature_for(target)
                session.feed(observation, current+.002)
                if session.ready_to_confirm:
                    advanced = session.confirm(current+.003)
                if advanced:
                    break
            self.assertTrue(advanced, session.hold_reason)
            stamp = current+2.

        for _ in session.train_targets:
            collect_target(60, -.7)
        self.assertEqual(session.phase, 'fit')
        self.assertEqual(len(session.train[0]['features']), 12)
        candidate = session.fit()
        session.begin_check(candidate, stamp)
        for frame in range(30):
            current = stamp+1.+frame/30
            seq += 1
            session.feed(sample(seq, current, session.target, -.78), current+.002)
        self.assertFalse(session.ready_to_confirm)
        self.assertEqual(session.hold_reason,
                         'Move back to your usual position. You’re at 78 cm · aim for about 70 cm.')
        self.assertEqual(session.index, 0)
        stamp = current+1.
        for _ in session.check_targets:
            collect_target(30, -.7)
        self.assertTrue(session.complete)
        report = session.evaluate()
        self.assertTrue(report['usable'], report)
        self.assertFalse(report['passed'])
        store = ProfileStore(self.folder)
        selected = []
        key = store.adopt(candidate, report, BOUNDS, session.archive,
                          activate=lambda: selected.append('personal'))
        self.assertEqual(selected, ['personal'])
        with patch('context_operator.providers.gaze.LOCAL', self.folder):
            phone = PhoneGaze(BOUNDS)
        self.assertEqual(phone.profile_key, key)


if __name__ == '__main__':
    unittest.main()
