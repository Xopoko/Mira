import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase

from context_operator.controller import Controller
from context_operator.core.contracts import Element, Focus, Gaze, Intent, Rect, Scene
from context_operator.core.webcam import GazeArbitrator, WebcamObservation
from context_operator.personalization.webcam import TRAIN_TARGETS
from context_operator.providers.webcam import SCHEMA, WebcamGaze, decode_observation
from context_operator.settings import DEFAULTS
from context_operator.ui.webcam_setup import WebcamSetupWindow

APP = QApplication.instance() or QApplication([])
for _font in ('segoeui.ttf', 'segoeuib.ttf', 'seguisb.ttf', 'seguisym.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/' + _font)


class ArbitrationTests(unittest.TestCase):
    def test_primary_coordinates_remain_authoritative_and_disagreement_is_reported(self):
        arbitrator = GazeArbitrator()
        primary = Gaze(300., 200., 10.00, 55., 'iphone')
        secondary = Gaze(720., 600., 10.02, 40., 'webcam')
        for stamp in (10.00, 10.06, 10.13):
            choice = arbitrator.choose(primary, Gaze(720., 600., stamp, 40., 'webcam'), stamp + .01)
            self.assertIs(choice.gaze, primary)
            self.assertEqual((choice.gaze.x, choice.gaze.y, choice.gaze.source), (300., 200., 'iphone'))
        self.assertFalse(choice.corroborated)

    def test_fallback_requires_three_distinct_frames_and_minimum_span(self):
        arbitrator = GazeArbitrator()
        values = [Gaze(100., 90., stamp, 50., 'webcam') for stamp in (20., 20.05, 20.13)]
        self.assertEqual(arbitrator.choose(None, values[0], 20.01).mode, 'none')
        # Polling the same frame again is not independent evidence.
        self.assertEqual(arbitrator.choose(None, values[0], 20.02).mode, 'none')
        self.assertEqual(arbitrator.choose(None, values[1], 20.06).mode, 'none')
        choice = arbitrator.choose(None, values[2], 20.14)
        self.assertEqual(choice.mode, 'webcam')
        self.assertIs(choice.gaze, values[2])

    def test_stale_future_wrong_source_and_no_face_reset_fallback(self):
        arbitrator = GazeArbitrator()
        first = Gaze(100., 90., 30., 50., 'webcam')
        second = Gaze(100., 90., 30.07, 50., 'webcam')
        self.assertEqual(arbitrator.choose(None, first, 30.01).mode, 'none')
        self.assertEqual(arbitrator.choose(None, second, 30.08).mode, 'none')
        self.assertEqual(arbitrator.choose(None, None, 30.09).mode, 'none')
        self.assertEqual(arbitrator.count, 0)
        self.assertEqual(arbitrator.choose(None, Gaze(1, 1, 31., 50., 'webcam'), 30.9).mode, 'none')
        self.assertEqual(arbitrator.count, 0)
        self.assertEqual(arbitrator.choose(None, Gaze(1, 1, 30., 50., 'iphone'), 30.01).mode, 'none')
        self.assertEqual(arbitrator.count, 0)

    def test_provider_no_face_stale_and_future_clear_smoothed_gaze(self):
        class Worker:
            def __init__(self):
                self.observation = WebcamObservation(1, 40., (.1,) * 8)
                self.error = ''
                self.ready = False
            def stop(self): pass

        worker = Worker()
        provider = WebcamGaze(Rect(0, 0, 1000, 800), 'camera-fixture', calibrated=False,
                              worker_factory=lambda: worker)
        provider.active = True
        provider.worker = worker
        provider.smoother.update((100., 100.), 40.)
        worker.observation = WebcamObservation(2, 40.1, (), 'blink')
        self.assertEqual(provider.poll(40.11), (None, 'blink'))
        self.assertIsNone(provider.smoother.point)
        worker.observation = WebcamObservation(3, 39.7, (.1,) * 8)
        self.assertIsNone(provider.poll(40.)[0])
        worker.observation = WebcamObservation(4, 40.2, (.1,) * 8)
        self.assertIsNone(provider.poll(40.)[0])
        provider.stop()

    def test_invalid_frame_barrier_rejects_late_worker_result(self):
        class Worker:
            def __init__(self):
                self.observation = WebcamObservation(1, 40.1, (.1,) * 8)
                self.error = ''
                self.ready = True
            def stop(self): pass

        class InvalidVideoFrame:
            def isValid(self): return False

        worker = Worker()
        provider = WebcamGaze(Rect(0, 0, 1000, 800), 'fixture-camera', calibrated=False,
                              worker_factory=lambda: worker)
        provider.active = True
        provider.worker = worker
        with patch('context_operator.providers.webcam.time.monotonic', return_value=40.2):
            provider._frame(InvalidVideoFrame())
        # An inference already in flight completes after the invalid camera frame.
        worker.observation = WebcamObservation(1, 40.1, (.1,) * 8)
        latest = provider.latest()
        self.assertFalse(latest.features)
        self.assertEqual('Camera frame unavailable', latest.reason)
        self.assertIsNone(provider.poll(40.21)[0])
        provider.stop()

    def test_worker_response_contract_rejects_malformed_rows(self):
        valid = {'schema': SCHEMA, 'seq': 7, 'captured': 50.25,
                 'features': [.0] * 8, 'reason': ''}
        parsed = decode_observation(json.dumps(valid), 7, 50.25)
        self.assertEqual(parsed.seq, 7)
        self.assertEqual(parsed.reason, '')
        for change in (
            {'seq': 8}, {'captured': 50.26}, {'features': [.0] * 7},
            {'extra': True}, {'schema': 'untyped'},
        ):
            malformed = dict(valid, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode_observation(json.dumps(malformed), 7, 50.25)
        with self.assertRaises(ValueError):
            decode_observation('{"schema":"contextoperator.webcam.v1","seq":7,"captured":NaN,"features":[],"reason":""}', 7, 50.25)


class FakeWebcam:
    def __init__(self, bounds, device_id, gazes):
        self.bounds = bounds
        self.device_id = device_id
        self.gazes = iter(gazes)
        self.stopped = False

    def start(self):
        pass

    def poll(self, now):
        return next(self.gazes, None), 'fixture'

    def stop(self):
        self.stopped = True


class ControllerWebcamTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(dict(DEFAULTS, vision=False, ranker=False,
            click_learning=False, anticipation=False), output=Mock())
        self.controller.timer.stop()
        self.controller.win = Mock()
        self.controller.win.bounds.return_value = Rect(0, 0, 1000, 800)
        self.controller.win.emergency.return_value = False
        self.controller.win.shift_down.return_value = False

    def tearDown(self):
        self.controller.close()

    def test_source_transition_revokes_focus_dwell_latched_commands_and_pause_stops_camera(self):
        times = (60., 60.06, 60.13)
        webcam = FakeWebcam(Rect(0, 0, 1000, 800), 'fixture',
            [Gaze(100., 100., stamp, 45., 'webcam') for stamp in times])
        c = self.controller
        c.running = True
        c.webcam_factory = Mock(return_value=webcam)
        c.toggle_webcam()
        c.gaze_source_mode = 'iphone'
        c.gaze = Gaze(100., 100., 59., 50., 'iphone')
        c.focus = Focus(Element('old', 'Old button', 'Button', Rect(80, 80, 40, 40)))
        c.dwell_id = ('window', 'rect')
        c.activation_history = ((59.,),)
        for stamp in times:
            c._select_gaze_source(stamp + .01)
        self.assertEqual(c.gaze_source_mode, 'webcam')
        self.assertIsNone(c.focus.target)
        self.assertIsNone(c.dwell_id)
        self.assertEqual(c.activation_history, ())
        self.assertEqual(c.gaze.source, 'webcam')
        c.pause()
        self.assertTrue(webcam.stopped)
        self.assertIsNone(c.webcam)
        self.assertIsNone(c.gaze)
        webcam.gazes = iter([Gaze(900., 700., 61., 45., 'webcam')])
        c.tick()
        self.assertIsNone(c.gaze)

    def test_webcam_points_never_enter_phone_click_learning(self):
        c = self.controller
        c.phone = None
        c.gaze = Gaze(400., 300., time.monotonic(), 50., 'webcam')
        c.click_history.observe = Mock()
        c.learning.learn_click = Mock(return_value=True)
        events = [Mock(x=400, y=300, captured=time.monotonic(), released=time.monotonic())]
        self.assertFalse(c._learn_clicks(None, time.monotonic(), events))
        c.click_history.observe.assert_not_called()
        c.learning.learn_click.assert_not_called()

    def test_old_source_epoch_cannot_queue_an_action(self):
        c = self.controller
        c.running = True
        c.busy = True
        old_epoch = c.gaze_source_epoch
        c._revoke_gaze_target()
        c._decision(Intent('click'), Scene(1, time.monotonic(), Rect(0, 0, 1000, 800), ()),
                    c.generation, time.monotonic(), source_epoch=old_epoch)
        self.assertFalse(c.busy)
        self.assertFalse(c.output_pending)
        c.output.execute.assert_not_called()

    def test_source_change_after_queue_prevents_dispatch(self):
        c = self.controller
        c.running = True
        c.phone = Mock()
        c.phone.worker.is_alive.return_value = False
        c.phone.poll.return_value = (None, 'iPhone connection closed', 0)
        c.gaze_source_mode = 'iphone'
        scene = Scene(1, time.monotonic(), Rect(0, 0, 1000, 800), ())
        c.scene = scene
        with patch('context_operator.controller.threading.Thread') as thread:
            c._decision(Intent('scroll'), scene, c.generation, time.monotonic(),
                        source_epoch=c.gaze_source_epoch)
            dispatch = thread.call_args.kwargs['target']
            c.tick()
            dispatch()
        APP.processEvents()
        c.output.execute.assert_not_called()
        self.assertFalse(c.busy)
        self.assertFalse(c.output_pending)
        self.assertEqual(c.gaze_status, 'iPhone connection closed')

    def test_latched_speech_survives_missing_webcam_frame_but_not_camera_failure(self):
        c = self.controller
        c.running = c.microphone = True
        c.voice_generation = 5
        c.command = Mock()
        webcam = WebcamGaze(c.win.bounds(), 'fixture', calibrated=False)
        c.webcam_factory = Mock(return_value=webcam)
        with patch.object(webcam, 'start'):
            c.toggle_webcam()
        webcam.active = True
        c.gaze_source_mode = 'webcam'
        now = time.monotonic()
        c._speech({'type': 'speech_start', 'utteranceId': 'old'}, 5)
        c._select_gaze_source(now)
        self.assertIn('old', c.speech_focus)

        webcam._fail('Webcam disconnected')
        c._speech({'type': 'final', 'utteranceId': 'old', 'text': 'click'}, 5)
        c.command.assert_not_called()
        primary = Gaze(100., 100., now)
        c.gaze = primary
        c._select_gaze_source(now + .01)
        self.assertIs(c.gaze, primary)
        self.assertEqual(c.gaze_source_mode, 'iphone')
        self.assertEqual(c.webcam_status, 'Webcam disconnected')

    def test_reconnect_resets_camera_sequence_but_keeps_completed_targets(self):
        c = self.controller
        c.settings['webcam_device_id'] = 'fixture-camera'
        made = []

        class RestartedProvider:
            def __init__(self, bounds, device_id, *, calibrated=True):
                self.bounds, self.device_id, self.calibrated = bounds, device_id, calibrated
                self.observation = WebcamObservation(1, 100., (0.,) * 8)
                self.error = ''
                self.stopped = False
                made.append(self)
            def start(self): pass
            def latest(self): return self.observation
            def stop(self): self.stopped = True

        with tempfile.TemporaryDirectory() as folder:
            setup = WebcamSetupWindow(c, provider_factory=RestartedProvider, folder=folder)
            setup.timer.stop()
            self.addCleanup(setup.close)
            clock = {'now': 100.}
            with patch('context_operator.ui.webcam_setup.time.monotonic', side_effect=lambda: clock['now']):
                setup.connect_camera()
                setup.begin()
                target = TRAIN_TARGETS[0]
                features = (target[0] - .5, target[1] - .5, 0., 0., 0., .5, .5, .25)
                for seq in range(1, 12):
                    stamp = 100.51 + (seq - 1) * .07
                    clock['now'] = stamp
                    made[0].observation = WebcamObservation(seq, stamp, features)
                    setup.tick()
                self.assertEqual(setup.session.index, 1)
                self.assertEqual(len(setup.session.train), 1)

                setup.prepare_again()
                clock['now'] = 102.
                setup.connect_camera()
                made[1].observation = WebcamObservation(0, 102., features)
                setup.begin()
                self.assertTrue(made[0].stopped)
                self.assertEqual(setup.session.index, 1)
                target = TRAIN_TARGETS[1]
                features = (target[0] - .5, target[1] - .5, 0., 0., 0., .5, .5, .25)
                for seq in range(1, 12):
                    stamp = 102.51 + (seq - 1) * .07
                    clock['now'] = stamp
                    made[1].observation = WebcamObservation(seq, stamp, features)
                    setup.tick()
            self.assertEqual(setup.session.index, 2)
            self.assertEqual(len(setup.session.train), 2)


if __name__ == '__main__':
    unittest.main()
