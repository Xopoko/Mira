"""Offscreen Qt walkthrough of webcam settings and the full numeric setup flow.

Run with ``.venv/Scripts/python.exe scripts/check_webcam_ui.py``. All camera
observations are generated in process; no capture device, model runtime, network,
or native input is opened. PNG evidence and a compact manifest go to
``.local/evidence/webcam``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from context_operator.controller import Controller
from context_operator.core.contracts import Rect
from context_operator.core.webcam import WebcamObservation
from context_operator.personalization.webcam import CHECK_TARGETS, TRAIN_TARGETS, WebcamProfileStore
from context_operator.personalization.webcam_attempts import WebcamAttemptStore
from context_operator.settings import DEFAULTS
from context_operator.ui.window import Window
from context_operator.ui.webcam_setup import WebcamSetupWindow, REACTION_SECONDS

APP = QApplication.instance() or QApplication([])
for _font in ('segoeui.ttf', 'segoeuib.ttf', 'seguisb.ttf', 'seguisym.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/' + _font)
APP.setFont(QFont('Segoe UI', 10))
EVIDENCE = ROOT / '.local' / 'evidence' / 'webcam'
BOUNDS = Rect(0, 0, 1920, 1080)


def feature_for(target):
    x, y = target
    return (x - .5, y - .5, 0., 0., 0., .5, .5, .25)


class FakeCameraProvider:
    def __init__(self, bounds, device_id, *, calibrated=True):
        self.bounds = bounds
        self.device_id = device_id
        self.calibrated = calibrated
        self.error = ''
        self.started = False
        self.stopped = False
        self.observation = WebcamObservation(1, 100., feature_for(TRAIN_TARGETS[0]))

    def start(self):
        self.started = True

    def latest(self):
        return self.observation

    def stop(self):
        self.stopped = True
        self.started = False


class WebcamUiWalkthrough(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EVIDENCE.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.controller = Controller(dict(DEFAULTS, vision=False, ranker=False,
            click_learning=False, anticipation=False), output=Mock())
        self.controller.timer.stop()
        self.controller.win = Mock()
        self.controller.win.bounds.return_value = BOUNDS
        self.controller.win.emergency.return_value = False
        self.addCleanup(self.controller.close)
        self.window = Window(self.controller)
        self.window.resize(1280, 800)
        self.addCleanup(self.window.close)

    def test_settings_select_device_without_connecting_it(self):
        self.window.navigate(3)
        self.window.show()
        APP.processEvents()
        self.assertFalse(self.window.webcam_device.isVisible())
        with patch('context_operator.ui.window.Speech.available_microphones',return_value=[]):
            self.window.advanced.click()
            for _ in range(20):
                if not self.window._microphone_listing:
                    break
                QTest.qWait(10)
            self.assertFalse(self.window._microphone_listing)
        self.assertIsNone(self.controller.webcam)
        self.assertEqual(self.window.webcam_toggle.text(), 'Connect webcam')
        self.assertEqual(self.controller.settings['webcam_device_id'], '')
        with patch('context_operator.providers.webcam.available_cameras',
                   return_value=[('fixture-a', 'Fixture camera A'), ('fixture-b', 'Fixture camera B')]):
            self.window.webcam_find.click()
        self.assertEqual([self.window.webcam_device.itemData(i)
                          for i in range(self.window.webcam_device.count())], ['fixture-a', 'fixture-b'])
        self.window.webcam_device.setCurrentIndex(1)
        with patch('context_operator.settings.save') as save:
            self.window._select_webcam()
        save.assert_called_once()
        self.assertEqual(self.controller.settings['webcam_device_id'], 'fixture-b')
        self.assertIsNone(self.controller.webcam)

        area = self.window.stack.widget(3)
        content_y = self.window.webcam_device.mapTo(area.widget(), QPoint(0, 0)).y()
        bar = area.verticalScrollBar()
        bar.setValue(max(0, min(bar.maximum(), content_y - area.viewport().height() // 3)))
        APP.processEvents()
        self.assertTrue(self.window.webcam_device.isVisible())
        self.assertTrue(self.window.webcam_find.isVisible())
        self.assertTrue(self.window.webcam_toggle.isVisible())
        self.assertTrue(self.window.grab().save(str(EVIDENCE / 'settings_webcam.png')))

    def test_setup_intro_full_9_plus_5_stream_and_result(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.settings['webcam_device_id'] = 'fixture-camera'
            self.controller.settings['webcam_device_name'] = 'Fixture Camera'
            window = WebcamSetupWindow(self.controller, folder=folder,
                                       provider_factory=FakeCameraProvider)
            self.addCleanup(window.close)
            window.timer.stop()
            window.resize(1020, 690)
            window.show()
            APP.processEvents()
            self.assertIn('9 points', window.guide.text())
            self.assertFalse(window.collecting)
            self.assertIsNone(window.provider)
            self.assertGreaterEqual(window.guide.height(), window.guide.sizeHint().height())
            self.assertGreaterEqual(window.guide.geometry().top(), 0)
            self.assertLessEqual(window.guide.geometry().bottom() + 1, window.canvas.height())
            self.assertEqual((window.width(), window.height()), (1020, 690))
            self.assertTrue(window.grab().save(str(EVIDENCE / 'setup_intro.png')))

            clock = {'now': 100.}
            with patch('context_operator.ui.webcam_setup.time.monotonic', side_effect=lambda: clock['now']):
                window.connect_button.click()
                self.assertTrue(window.provider.started)
                self.assertFalse(window.provider.calibrated)
                window.begin_button.click()
                self.assertTrue(window.collecting)
                camera = window.provider
                seq = 1
                target_screenshot = False
                for target in (*TRAIN_TARGETS, *CHECK_TARGETS):
                    values = feature_for(target)
                    previous = (window.session.phase, window.session.index)
                    base = window.collect_after + .01
                    # Six distinct frames/second was previously an impossible gate.
                    for frame in range(20):
                        captured = base + frame / 6
                        clock['now'] = captured + .06
                        window.provider.observation = WebcamObservation(seq, captured, values)
                        window.tick()
                        # The Qt timer polls more often than the synthetic camera; a repeated fresh
                        # frame must not count twice or clear a ready fixation.
                        clock['now'] = captured + .10
                        window.tick()
                        # Inference delay creates a short stale interval before the
                        # next result. It revokes readiness without erasing history.
                        clock['now'] = captured + .21
                        window.tick()
                        seq += 1
                        if not target_screenshot and frame == 5:
                            APP.processEvents()
                            self.assertFalse(window.session.ready)
                            self.assertGreater(window.session.progress, 0.)
                            self.assertTrue(window.grab().save(str(EVIDENCE / 'setup_target.png')))
                            target_screenshot = True
                        if previous != (window.session.phase, window.session.index):
                            break
                    self.assertNotEqual(previous, (window.session.phase, window.session.index))

            self.assertEqual(window.session.phase, 'result')
            self.assertEqual(len(window.session.train), 9)
            self.assertEqual(len(window.session.check), 5)
            self.assertTrue(window.session.report.usable)
            self.assertFalse(window.session.report.passed)
            self.assertFalse(window.collecting)
            self.assertTrue(camera.stopped)
            self.assertTrue(window.apply_button.isVisible())
            self.assertIsNotNone(window.attempt_path)
            self.assertEqual(WebcamAttemptStore(folder).load_latest()['usable'], True)
            APP.processEvents()
            self.assertTrue(window.grab().save(str(EVIDENCE / 'setup_result.png')))
            window.apply_button.click()
            self.assertTrue(window.applied)
            saved = WebcamProfileStore(folder).load(BOUNDS, 'fixture-camera')
            self.assertEqual(saved.device_id, 'fixture-camera')
            manifest = {
                'mode': 'offscreen Qt with generated numeric observations',
                'hardware_opened': False, 'network_used': False, 'native_input_used': False,
                'cadence_fps': 6, 'inference_latency_ms': 60, 'automatic_advance': True,
                'train_targets': len(window.session.train), 'check_targets': len(window.session.check),
                'usable': window.session.report.usable, 'passed': window.session.report.passed,
                'tests': ['WebcamUiWalkthrough.test_settings_select_device_without_connecting_it',
                          'WebcamUiWalkthrough.test_setup_intro_full_9_plus_5_stream_and_result',
                          'WebcamUiWalkthrough.test_pause_preview_return_and_target_reaction_barrier',
                          'WebcamUiWalkthrough.test_compact_preparation_has_one_action_and_readable_content',
                          'WebcamUiWalkthrough.test_failed_result_preserves_numeric_attempt_and_does_not_blame_placement'],
                'screenshots': ['settings_webcam.png', 'setup_intro.png', 'setup_target.png', 'setup_result.png', 'setup_compact.png', 'setup_failed.png'],
            }
            (EVIDENCE / 'walkthrough.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def test_pause_preview_return_and_target_reaction_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.settings['webcam_device_id'] = 'fixture-camera'
            window = WebcamSetupWindow(self.controller, folder=folder, provider_factory=FakeCameraProvider)
            self.addCleanup(window.close)
            window.timer.stop()
            window.show()
            clock = {'now': 100.}
            with patch('context_operator.ui.webcam_setup.time.monotonic', side_effect=lambda: clock['now']):
                window.connect_camera()
                window.begin()
                camera = window.provider
                # Frames from the previous target / preparation cannot train this one.
                clock['now'] = 100. + REACTION_SECONDS + .01
                camera.observation = WebcamObservation(2, clock['now']-.05, feature_for(TRAIN_TARGETS[0]))
                window.tick()
                self.assertEqual(len(window.session.frames), 0)
                camera.observation = WebcamObservation(3, clock['now'], feature_for(TRAIN_TARGETS[0]))
                window.tick()
                self.assertEqual(len(window.session.frames), 1)
                window.pause_button.click()
                self.assertTrue(window.paused)
                self.assertEqual(len(window.session.frames), 0)
                clock['now'] += .1
                camera.observation = WebcamObservation(4, clock['now'], feature_for(TRAIN_TARGETS[0]))
                window.tick()
                self.assertEqual(len(window.session.frames), 0)
                self.assertIn('Paused', window.target_message())
                window.pause_button.click()
                self.assertFalse(window.paused)
                self.assertGreater(window.collect_after, clock['now'])
                window.camera_button.click()
                self.assertFalse(window.collecting)
                self.assertFalse(camera.stopped)
                self.assertTrue(window.preview_panel.isVisible())
                window.begin()
                camera.error = 'Can’t read the camera image'
                window.tick()
                self.assertFalse(window.collecting)
                self.assertTrue(camera.stopped)
                self.assertIn('Can’t read the camera image', window.status.text())
                self.assertTrue(window.connect_button.isVisible())
                window.restart()
                self.assertEqual(window.session.phase, 'train')
                self.assertIn('When the ring fills', window.guide.text())

    def test_compact_preparation_has_one_action_and_readable_content(self):
        self.controller.settings['webcam_device_id'] = 'fixture-camera'
        window = WebcamSetupWindow(self.controller, provider_factory=FakeCameraProvider)
        self.addCleanup(window.close)
        window.timer.stop()
        window.resize(820, 590)
        window.show()
        APP.processEvents()
        self.assertTrue(window.connect_button.isVisible())
        self.assertFalse(window.begin_button.isVisible())
        self.assertFalse(window.collection_bar.isVisible())
        self.assertGreaterEqual(window.guide.height(), window.guide.heightForWidth(window.guide.width()))
        for widget in (window.preview, window.guide, window.connect_button, window.close_button):
            position = widget.mapTo(window.canvas, QPoint(0, 0))
            self.assertGreaterEqual(position.x(), 0)
            self.assertGreaterEqual(position.y(), 0)
            self.assertLessEqual(position.x()+widget.width(), window.canvas.width())
            self.assertLessEqual(position.y()+widget.height(), window.canvas.height())
        self.assertTrue(window.grab().save(str(EVIDENCE / 'setup_compact.png')))

    def test_failed_result_preserves_numeric_attempt_and_does_not_blame_placement(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.settings['webcam_device_id'] = 'fixture-camera'
            window = WebcamSetupWindow(self.controller, provider_factory=FakeCameraProvider, folder=folder)
            self.addCleanup(window.close)
            window.timer.stop()
            window.resize(1020, 690)
            window.show()
            clock = {'now': 100.}
            seq = 0
            with patch('context_operator.ui.webcam_setup.time.monotonic', side_effect=lambda: clock['now']):
                window.connect_camera()
                window.begin()
                for index, target in enumerate((*TRAIN_TARGETS, *CHECK_TARGETS)):
                    values = list(feature_for(target))
                    if index >= 9:
                        values[0] += .18  # Deliberately incorrect unseen measurements.
                    before = (window.session.phase, window.session.index)
                    base = window.collect_after+.01
                    for frame in range(15):
                        seq += 1
                        clock['now'] = base+frame*.07
                        window.provider.observation = WebcamObservation(seq, clock['now'], tuple(values))
                        window.tick()
                        if before != (window.session.phase, window.session.index):
                            break
            APP.processEvents()
            self.assertEqual(window.session.phase, 'result')
            self.assertFalse(window.session.report.usable)
            self.assertFalse(window.apply_button.isVisible())
            self.assertFalse(window.store.path.exists())
            self.assertTrue(window.attempt_path.exists())
            attempt = WebcamAttemptStore(folder).load_latest()
            self.assertFalse(attempt['usable'])
            self.assertTrue(attempt['train_rows'][0]['samples'])
            self.assertIn('saved the measurements', window.record_status.text())
            for bad_copy in ('improve the angle', 'glare', 'eye level', 'Turn your'):
                self.assertNotIn(bad_copy, window.section.text()+window.guide.text())
            self.assertTrue(window.grab().save(str(EVIDENCE / 'setup_failed.png')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
