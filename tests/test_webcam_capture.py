"""Real Qt video frames in memory; no camera is enumerated or opened."""
import json
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import unittest
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor
from PySide6.QtMultimedia import QVideoFrame
from context_operator.core.contracts import Rect
from context_operator.core.webcam import WebcamObservation
from context_operator.providers.webcam import WebcamGaze,decode_observation

APP=QApplication.instance() or QApplication([])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.provider=WebcamGaze(Rect(0,0,1920,1080),'fixture',calibrated=False)
        self.provider.active=True
        self.worker=Mock(ready=True,error='')
        self.worker.observation=WebcamObservation(0,0.)
        self.provider.worker=self.worker
        self.addCleanup(self.provider.stop)

    def frame(self,stamp,width=641,height=481):
        image=QImage(width,height,QImage.Format_RGB888)
        image.fill(QColor(12,34,56))
        frame=QVideoFrame(image)
        frame.setStartTime(stamp)
        return frame

    def test_frame_resize_padding_and_camera_time_preserved(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(self.frame(1_000_000))
        seq,captured,width,height,rgb=self.worker.submit.call_args.args[0]
        self.assertEqual(seq,1)
        self.assertEqual(captured,10.)
        self.assertLessEqual(width,1280)
        self.assertLessEqual(height,720)
        self.assertEqual(len(rgb),width*height*3)
        self.assertEqual(rgb[:3],bytes((12,34,56)))
        self.worker.submit.reset_mock()
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.8):
            self.provider._frame(self.frame(1_033_333))
        self.worker.submit.assert_not_called()  # old buffered image is not restamped

    def test_capture_keeps_large_input_within_1280_by_720_envelope(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(self.frame(1_000_000,1601,901))
        seq,captured,width,height,rgb=self.worker.submit.call_args.args[0]
        self.assertGreater(width,1200)
        self.assertEqual(height,720)
        self.assertEqual(len(rgb),width*height*3)

    def test_small_input_is_not_upscaled_and_preview_is_display_only_mirror(self):
        image=QImage(2,1,QImage.Format_RGB888)
        image.setPixelColor(0,0,QColor(255,0,0))
        image.setPixelColor(1,0,QColor(0,0,255))
        frame=QVideoFrame(image)
        frame.setStartTime(1_000_000)
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(frame)
        _,_,width,height,rgb=self.worker.submit.call_args.args[0]
        self.assertEqual((width,height),(2,1))
        self.assertEqual(rgb[:6],bytes((255,0,0,0,0,255)))
        self.assertEqual(self.provider.preview_image.pixelColor(0,0),QColor(0,0,255))
        self.assertEqual(self.provider.preview_captured,10.)

    def test_preview_clears_on_invalid_frame_and_stop(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(self.frame(1_000_000))
        self.assertIsNotNone(self.provider.preview_image)
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.1):
            self.provider._frame(QVideoFrame())
        self.assertIsNone(self.provider.preview_image)
        self.assertEqual(self.provider.preview_captured,0.)
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.2):
            self.provider._frame(self.frame(2_000_000))
        self.assertIsNotNone(self.provider.preview_image)
        self.provider.stop()
        self.assertIsNone(self.provider.preview_image)

    def test_latest_preserves_runtime_error_after_invalid_epoch(self):
        self.provider.preview_image=QImage(2,2,QImage.Format_RGB888)
        self.worker.error='Face tracking unavailable · FileNotFoundError'
        self.worker.observation=WebcamObservation(0,0.,reason=self.worker.error)
        self.provider.invalid_after=10.
        self.assertEqual(self.provider.latest().reason,self.worker.error)
        self.assertIsNone(self.provider.preview_image)

    def test_calibrated_capture_never_exposes_preview(self):
        provider=WebcamGaze(Rect(0,0,1920,1080),'fixture',calibrated=True)
        provider.active=True
        provider.worker=self.worker
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            provider._frame(self.frame(1_000_000,320,240))
        self.assertIsNone(provider.preview_image)
        self.assertEqual(provider.preview_captured,0.)
        provider.stop()

    def test_duplicate_frame_does_not_make_new_observation(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(self.frame(100))
            self.provider._frame(self.frame(100))
        self.assertEqual(self.worker.submit.call_count,1)

    def test_camera_clock_rewind_stops_source(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(self.frame(100))
            self.provider._frame(self.frame(99))
        self.assertFalse(self.provider.active)
        self.worker.stop.assert_called_once()

    def test_late_inference_cannot_cross_invalid_camera_frame(self):
        with patch('context_operator.providers.webcam.time.monotonic',return_value=10.):
            self.provider._frame(QVideoFrame())
        self.worker.observation=WebcamObservation(1,9.99,(0.,)*8)
        late = self.provider.latest()
        self.assertFalse(late.features)
        self.assertEqual(late.reason, 'Camera frame unavailable')
        self.assertEqual(self.provider.poll(10.005), (None, 'Camera frame unavailable'))
        self.worker.observation = fresh = WebcamObservation(2,10.01,(0.,)*8)
        self.assertEqual(self.provider.latest(), fresh)

    def test_contradictory_status_and_duplicate_keys_rejected(self):
        packet={'schema':'contextoperator.webcam.v1','seq':1,'captured':10.,'features':[0.]*8,'reason':'blink'}
        with self.assertRaises(ValueError):decode_observation(json.dumps(packet),1,10.)
        packet['reason']=''
        raw=json.dumps(packet)[:-1]+',"seq":1}'
        with self.assertRaises(ValueError):decode_observation(raw,1,10.)


if __name__=='__main__':unittest.main()
