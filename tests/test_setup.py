import os
import json
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

from context_operator.core.contracts import Rect
from context_operator.foundations.gaze.profile import Profile
from context_operator.personalization import PersonalSession, ProfileStore
from context_operator.personalization.session import CHECK_TARGETS, TRAIN_TARGETS, PROTOCOL
from context_operator.providers.gaze import PhoneGaze, Sample, learning_path
from context_operator.ui.setup import SetupWindow, open_setup


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

    def supports(self, x):
        return True


def fitted_profile(offset=0):
    x = np.zeros((12, 22), dtype=float)
    x[:, 0] = np.linspace(.1, .9, 12)
    x[:, 1] = np.linspace(.9, .1, 12)
    y = np.c_[x[:, 0]*1900+offset, x[:, 1]*1000]
    arrays = Profile._fit(x, y, ('linear', 10., 0.))
    metadata = {'display': BOUNDS.as_list(), 'engine': 'arkit-truedepth-v1',
                'features': 22, 'protocol': PROTOCOL}
    settings = {'version': 2, 'model': ['linear', 10., 0.], 'support_limit': 2.}
    return Profile(arrays, metadata, settings)


def accepted_report():
    return {'passed': True, 'scope': 'same-session-independent-targets',
            'train_targets': 12, 'check_targets': 27,
            'median_px': 42., 'p90_px': 93., 'coverage': 1.}


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_raw_collection_progresses_without_incumbent_and_resumes(self):
        path = self.folder/'session.json'
        session = PersonalSession(BOUNDS, path)
        seq = 0
        stamp = 100.
        for posture in range(3):
            if posture:
                self.assertTrue(session.paused)
                session.next_posture(stamp)
            for index in range(len(TRAIN_TARGETS)):
                target = session.target
                self.assertIsNotNone(target)
                for frame in range(12):
                    seq += 1
                    current = stamp + 1. + frame*.07
                    advanced = session.feed(sample(seq, current, target, (-.7, -.6, -.8)[posture]), current+.005)
                self.assertTrue(advanced)
                stamp += 2.
            if posture == 0:
                resumed = PersonalSession.resume(BOUNDS, path)
                self.assertEqual(len(resumed.train), len(TRAIN_TARGETS))
                self.assertTrue(resumed.paused)
        self.assertEqual(session.phase, 'fit')
        self.assertEqual(len(session.train), 3*len(TRAIN_TARGETS))
        self.assertNotIn('video', path.read_text(encoding='utf-8'))
        self.assertEqual(learning_path('personal', 'abcdef1234567890', self.folder).name,
                         'gaze-adaptation-personal-abcdef1234567890.json')

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
        session = PersonalSession(BOUNDS)
        target = session.target
        for seq in range(1, 12):
            stamp = 100+seq*.07
            self.assertFalse(session.feed(sample(seq, stamp, target), stamp+.01))
        self.assertFalse(session.feed(sample(12, 100.84, target), 101.5))
        self.assertEqual(session.index, 0)
        self.assertFalse(session.feed(sample(12, 101.6, target), 101.61))
        self.assertFalse(session.feed(sample(12, 101.6, target), 101.62))
        self.assertEqual(len(session.frames), 1)
        blink = sample(13, 101.67, target)
        blink.blink = True
        self.assertFalse(session.feed(blink, 101.68))
        self.assertEqual(session.index, 0)
        self.assertEqual(session.frames, [])

    def test_second_posture_requires_measured_distance_change(self):
        session = PersonalSession(BOUNDS)
        for index, fraction in enumerate(TRAIN_TARGETS):
            x = np.zeros((12, 22))
            x[:, 6] = -.7
            session.train.append({'phase': 'train', 'posture': 0, 'index': index,
                                  'target': [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h],
                                  'features': x.tolist(),
                                  'eye_origins': np.zeros((12, 2, 3)).tolist()})
        session.posture = 1
        target = session.target
        for seq in range(1, 73):
            stamp = 100+seq/60
            self.assertFalse(session.feed(sample(seq, stamp, target, -.7), stamp+.002))
        self.assertEqual(len(session.train), len(TRAIN_TARGETS))
        self.assertIn('slightly closer', session.hold_reason)
        advanced = False
        for seq in range(73, 145):
            stamp = 103+seq/60
            advanced = session.feed(sample(seq, stamp, target, -.6), stamp+.002)
            if advanced:
                break
        self.assertTrue(advanced)
        self.assertEqual(len(session.train), len(TRAIN_TARGETS)+1)

    def test_corrupt_archive_fails_closed(self):
        path = self.folder/'session.json'
        path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            PersonalSession.resume(BOUNDS, path)
        self.assertEqual(path.read_text(encoding='utf-8'), '{broken')

    def test_check_uses_unseen_targets_and_measures_raw_errors(self):
        self.assertFalse(set(TRAIN_TARGETS) & set(CHECK_TARGETS))
        session = PersonalSession(BOUNDS)
        session.phase = 'fit'
        session.begin_check(DirectModel(), 100.)
        for posture in range(3):
            for index, fraction in enumerate(CHECK_TARGETS):
                target = [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h]
                x = np.zeros((12, 22))
                x[:, :2] = target
                session.check.append({'phase': 'check', 'posture': posture, 'index': index,
                                      'target': target, 'features': x.tolist(),
                                      'eye_origins': np.zeros((12, 2, 3)).tolist()})
        session.phase = 'result'
        report = session.evaluate()
        self.assertTrue(report['passed'])
        self.assertEqual(report['median_px'], 0)
        self.assertEqual(report['p90_px'], 0)
        self.assertEqual(report['coverage'], 1)
        self.assertEqual(report['check_targets'], 27)
        for frame in session.check[0]['features'][:5]:
            frame[0] += 1000
        failed = session.evaluate()
        self.assertFalse(failed['passed'])
        # Global P90 must describe the individual held-out predictions; a bad
        # target still fails its separate local gate even below 10% globally.
        self.assertEqual(failed['p90_px'], 0.)
        self.assertGreater(failed['targets'][0]['p90_px'], 900.)

    def test_direct_profile_fit_has_22_features_and_target_group_validation(self):
        session = PersonalSession(BOUNDS)
        for posture in range(3):
            for index, fraction in enumerate(TRAIN_TARGETS):
                x = np.zeros((12, 22), dtype=float)
                x[:, 0] = fraction[0] + posture*.01
                x[:, 1] = fraction[1] - posture*.01
                x[:, 6] = -.7-posture*.1
                x[:, 13:19] = [0, 0, 1, 0, 0, 1]
                session.train.append({'phase': 'train', 'posture': posture, 'index': index,
                                      'target': [fraction[0]*BOUNDS.w, fraction[1]*BOUNDS.h],
                                      'features': x.tolist(),
                                      'eye_origins': np.zeros((12, 2, 3)).tolist()})
        session.phase = 'fit'
        profile = session.fit()
        self.assertEqual(profile.training_x.shape, (36, 22))
        self.assertEqual(profile.settings['model_kind'], 'personal_fixation_v1')
        self.assertEqual(profile.metadata['protocol'], PROTOCOL)

    def test_store_checks_receipt_and_rolls_back_failed_adoption(self):
        store = ProfileStore(self.folder)
        old = fitted_profile()
        key = store.adopt(old, accepted_report(), BOUNDS)
        self.assertEqual(store.status(BOUNDS)['profile_key'], key)
        before_profile = store.profile_path.read_bytes()
        before_receipt = store.receipt_path.read_bytes()
        new = fitted_profile(offset=90)
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
        loaded, status = store.load(BOUNDS)
        self.assertEqual(status['profile_key'], key)
        np.testing.assert_array_equal(loaded.training_y, old.training_y)
        store.profile_backup.write_bytes(b'broken too')
        self.assertFalse(store.status(BOUNDS)['available'])

    def test_activation_failure_rolls_back_profile(self):
        store = ProfileStore(self.folder)
        key = store.adopt(fitted_profile(), accepted_report(), BOUNDS)
        def fail_activation():
            raise RuntimeError('settings save failed')
        with self.assertRaises(RuntimeError):
            store.adopt(fitted_profile(offset=30), accepted_report(), BOUNDS,
                        activate=fail_activation)
        self.assertEqual(store.status(BOUNDS)['profile_key'], key)

    def test_rejects_unqualified_and_screen_mismatch(self):
        store = ProfileStore(self.folder)
        with self.assertRaises(ValueError):
            store.adopt(fitted_profile(), dict(accepted_report(), passed=False), BOUNDS)
        self.assertFalse(store.status(BOUNDS)['available'])
        store.adopt(fitted_profile(), accepted_report(), BOUNDS)
        self.assertFalse(store.status(Rect(0, 0, 1280, 720))['available'])

    def test_phone_provider_loads_only_explicit_personal_profile(self):
        store = ProfileStore(self.folder)
        key = store.adopt(fitted_profile(), accepted_report(), BOUNDS)
        with patch('context_operator.providers.gaze.LOCAL', self.folder):
            phone = PhoneGaze(BOUNDS, 'personal')
        self.assertEqual(phone.method, 'personal')
        self.assertEqual(phone.profile_key, key)
        self.assertEqual(phone.feature_scale.shape, (22,))
        self.assertEqual(phone.validation_radius, 116.25)

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
                                     win=SimpleNamespace(bounds=lambda: BOUNDS), pause=Mock())
        with patch('context_operator.ui.setup.QApplication.screens', return_value=[first, second]):
            with self.assertRaisesRegex(RuntimeError, 'falls between screens'):
                open_setup(controller)
        controller.pause.assert_not_called()

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
        session = PersonalSession(BOUNDS, path)
        session._save()
        before = path.read_bytes()
        with patch.object(session, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                for seq in range(1, 20):
                    stamp = 100+seq*.07
                    session.feed(sample(seq, stamp, session.target), stamp+.005)
        self.assertEqual(session.index, 0)
        self.assertEqual(session.train, [])
        self.assertEqual(path.read_bytes(), before)

    def test_explicit_confirmation_is_required_and_cannot_accept_stale_fixation(self):
        session = PersonalSession(BOUNDS, require_confirmation=True)
        for seq in range(1, 30):
            stamp = 100+seq*.07
            self.assertFalse(session.feed(sample(seq, stamp, session.target), stamp+.005))
        self.assertTrue(session.ready_to_confirm)
        self.assertEqual(session.index, 0)
        self.assertFalse(session.confirm(stamp+.25))
        self.assertTrue(session.confirm(stamp+.01))
        self.assertEqual(session.index, 1)
        self.assertTrue(session.train[0]['confirmed'])

    def test_far_posture_has_upper_bound_and_repair_preserves_other_positions(self):
        session = PersonalSession(BOUNDS, self.folder/'session.json')
        for posture, depth in enumerate((-.7, -.6, -.8)):
            for index, fraction in enumerate(TRAIN_TARGETS):
                x = np.zeros((12,22));x[:,6]=depth
                session.train.append({'phase':'train','posture':posture,'index':index,
                    'target':[fraction[0]*BOUNDS.w,fraction[1]*BOUNDS.h],
                    'features':x.tolist(),'eye_origins':np.zeros((12,2,3)).tolist()})
        ordinary = json.loads(json.dumps(session.train[:24]))
        session.phase='result'
        session.begin_repair([2]);session.next_posture(90.)
        self.assertIn('aim for about',session.posture_hint(sample(1,100.,session.target,-.96)))
        for seq in range(1,30):
            stamp=100+seq*.07
            self.assertFalse(session.feed(sample(seq,stamp,session.target,-.96),stamp+.001))
        self.assertEqual(session.index,0)
        stamp=110.;seq=100
        for _ in TRAIN_TARGETS:
            for frame in range(12):
                seq+=1;current=stamp+1+frame*.07
                done=session.feed(sample(seq,current,session.target,-.8),current+.001)
            self.assertTrue(done)
            stamp+=2.
        self.assertEqual(session.phase,'fit')
        self.assertEqual(session.train[:24],ordinary)
        self.assertEqual(len(session.train),36)
        resumed=PersonalSession.resume(BOUNDS,session.archive)
        self.assertEqual(resumed.phase,'fit')

    def test_completed_profile_reopens_with_exact_checked_candidate(self):
        session = PersonalSession(BOUNDS, self.folder/'session.json')
        for phase, positions in (('train',TRAIN_TARGETS),('check',CHECK_TARGETS)):
            rows=[]
            for posture in range(3):
                for index, fraction in enumerate(positions):
                    target=[fraction[0]*BOUNDS.w,fraction[1]*BOUNDS.h]
                    x=np.zeros((12,22));x[:,:2]=target
                    rows.append({'phase':phase,'posture':posture,'index':index,'target':target,
                                 'features':x.tolist(),'eye_origins':np.zeros((12,2,3)).tolist()})
            setattr(session,phase,rows)
        session.phase='result';session.candidate=fitted_profile()
        session.report=dict(accepted_report(),jitter_px=10.,bias_px=[0.,0.])
        session.persist_result()
        loaded=PersonalSession.resume(BOUNDS,session.archive,require_confirmation=True)
        self.assertEqual(loaded.phase,'result')
        self.assertIsNotNone(loaded.candidate)
        np.testing.assert_allclose(loaded.candidate.predict(np.zeros(22)),session.candidate.predict(np.zeros(22)))

    def test_one_jittery_posture_cannot_be_hidden_by_other_two(self):
        session=PersonalSession(BOUNDS)
        session.phase='result';session.candidate=DirectModel()
        for posture in range(3):
            for index,fraction in enumerate(CHECK_TARGETS):
                target=[fraction[0]*BOUNDS.w,fraction[1]*BOUNDS.h]
                x=np.zeros((12,22));x[:,:2]=target
                if posture==2:x[:,0]+=np.tile([-100.,100.],6)
                session.check.append({'phase':'check','posture':posture,'index':index,'target':target,
                                      'features':x.tolist(),'eye_origins':np.zeros((12,2,3)).tolist()})
        result=session.evaluate()
        self.assertLess(result['jitter_px'],result['limits']['jitter_px'])
        self.assertFalse(result['passed'])
        self.assertIn(2,result['failed_postures'])

    def test_corrupt_archive_can_be_replaced_without_losing_bad_bytes(self):
        path=self.folder/'session.json';path.write_bytes(b'{broken')
        controller=SimpleNamespace(win=SimpleNamespace(bounds=lambda:BOUNDS),settings={'phone_transport':'usb'})
        window=SetupWindow(controller,archive=path,store=ProfileStore(self.folder))
        try:
            window.restart()
            self.assertEqual(window.error,'')
            self.assertEqual(window.session.phase,'train')
            saved=list((self.folder/'personal-attempts').glob('corrupt-*.json'))
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0].read_bytes(),b'{broken')
        finally:window.close()

    def test_first_use_60hz_training_30hz_check_and_adoption(self):
        """No old profile: raw frames -> fit -> unseen check -> active provider."""
        session = PersonalSession(BOUNDS, self.folder/'session.json', require_confirmation=True)
        seq, stamp = 0, 100.

        def collect_target(hz, depth):
            nonlocal seq, stamp
            target = session.target
            advanced = False
            for frame in range(int(hz*1.2)):
                seq += 1
                current = stamp + 1. + frame/hz
                advanced = session.feed(sample(seq, current, target, depth), current+.002)
                if session.ready_to_confirm:
                    advanced = session.confirm(current+.003)
                if advanced:
                    break
            self.assertTrue(advanced, session.hold_reason)
            stamp = current+2.

        for posture, depth in enumerate((-.7, -.6, -.8)):
            if posture:
                session.next_posture(stamp)
            for _ in TRAIN_TARGETS:
                collect_target(60, depth)
        self.assertEqual(session.phase, 'fit')
        self.assertEqual(len(session.train[0]['features']), 12)
        candidate = session.fit()
        session.begin_check(candidate, stamp)
        for posture, depth in enumerate((-.7, -.6, -.8)):
            if posture:
                session.next_posture(stamp)
            for _ in CHECK_TARGETS:
                collect_target(30, depth)
        self.assertTrue(session.complete)
        report = session.evaluate()
        self.assertTrue(report['passed'], report)
        store = ProfileStore(self.folder)
        selected = []
        key = store.adopt(candidate, report, BOUNDS, session.archive,
                          activate=lambda: selected.append('personal'))
        self.assertEqual(selected, ['personal'])
        with patch('context_operator.providers.gaze.LOCAL', self.folder):
            phone = PhoneGaze(BOUNDS, 'personal')
        self.assertEqual(phone.profile_key, key)
        self.assertEqual(phone.method, 'personal')


if __name__ == '__main__':
    unittest.main()
