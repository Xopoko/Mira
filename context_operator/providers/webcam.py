"""Explicit Qt camera capture; latest RGB stays in memory in a private worker.

MediaPipe uses the existing isolated foundation runtime, under LocalModels'
start policy. Neither importing this module nor constructing it opens a camera.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
from collections import deque

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QImage

from ..core.contracts import GAZE_TTL, Gaze
from ..core.webcam import WebcamObservation
from ..foundations.gaze.smoothing import Smoother
from ..foundations.local_models import STANDALONE
from ..foundations.speech.runtime import VoiceError, check_policy
from ..settings import LOCAL, ROOT

_webcam_root = LOCAL if STANDALONE else ROOT.parent / 'MyEyeTracking'
_runtime = _webcam_root / ('webcam/.venv' if STANDALONE else '.venv') / 'Scripts/python.exe'
RUNTIME = Path(os.environ.get('MIRA_WEBCAM_PYTHON') or _runtime).expanduser()
MODEL = Path(os.environ.get('MIRA_WEBCAM_MODEL') or _webcam_root / 'models/face_landmarker.task').expanduser()
HEADER = struct.Struct('<4sQdII')
SCHEMA = 'contextoperator.webcam.v1'
REASONS = {'face_not_found':'Face out of view', 'ambiguous_faces':'More than one face in view',
           'blink':'Eyes closed', 'eyes_too_small':'Move closer so the camera can see your eyes',
           'unstable_iris':'The camera can’t see your eyes clearly', 'invalid_eye_line':'Can’t locate your eyes',
           'invalid_features':'Waiting for a usable face reading', 'invalid_landmarks':'Only part of your face is visible',
           'invalid_blendshapes':'Can’t tell if your eyes are open', 'invalid_frame_size':'Unsupported camera frame size'}


def available_cameras():
    from PySide6.QtMultimedia import QMediaDevices
    return [(bytes(item.id()).hex(), item.description()) for item in QMediaDevices.videoInputs()]


def decode_observation(line, seq, captured):
    def nonfinite(value):
        raise ValueError('Nonfinite webcam response')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate webcam response key')
            result[key]=value
        return result
    value = json.loads(line, parse_constant=nonfinite,object_pairs_hook=unique)
    if (not isinstance(value, dict) or set(value) != {'schema', 'seq', 'captured', 'features', 'reason'}
            or value['schema'] != SCHEMA or type(value['seq']) is not int or value['seq'] != seq
            or type(value['captured']) not in (float, int) or value['captured'] != captured
            or not math.isfinite(value['captured']) or not isinstance(value['reason'], str)
            or len(value['reason']) > 300 or not isinstance(value['features'], list)
            or len(value['features']) not in (0, 8)
            or bool(value['features']) == bool(value['reason'])
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value['features'])):
        raise ValueError('Invalid webcam observation contract')
    reason=REASONS.get(value['reason'],'Unusable camera frame') if value['reason'] else ''
    return WebcamObservation(seq, captured, tuple(value['features']), reason)


class InferenceWorker(threading.Thread):
    def __init__(self, *, runtime=RUNTIME, model=MODEL):
        super().__init__(daemon=True, name='webcam-inference')
        self.runtime, self.model = Path(runtime), Path(model)
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.ready = False
        self.child = None
        self.error = ''
        self.observation = WebcamObservation(0, 0., reason='Starting webcam tracking')
        self.child_lock = threading.Lock()
        self.inference_times = deque(maxlen=30)

    def submit(self, frame):
        if self.stop_event.is_set():
            return
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def run(self):
        child = None
        try:
            check_policy()
            if not self.runtime.is_file():
                raise FileNotFoundError('Set MIRA_WEBCAM_PYTHON to the MediaPipe interpreter')
            if not self.model.is_file():
                raise FileNotFoundError('Set MIRA_WEBCAM_MODEL to the Face Landmarker task file')
            with self.child_lock:
                if self.stop_event.is_set():
                    return
                child = subprocess.Popen(
                    [str(self.runtime), '-u', str(ROOT/'context_operator/providers/webcam_runtime.py'), '--model', str(self.model)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.child = child
            ready = json.loads(child.stdout.readline(4097))
            if ready != {'schema': SCHEMA, 'ready': True}:
                raise RuntimeError('Webcam tracking didn’t start')
            self.ready = True
            while not self.stop_event.is_set():
                check_policy()
                try:
                    seq, captured, width, height, pixels = self.frames.get(timeout=.2)
                except queue.Empty:
                    continue
                if not 0 <= time.monotonic() - captured < GAZE_TTL:
                    continue
                child.stdin.write(HEADER.pack(b'WCF1', seq, captured, width, height))
                child.stdin.write(pixels)
                child.stdin.flush()
                observation = decode_observation(child.stdout.readline(4097), seq, captured)
                completed = time.monotonic()
                if not self.stop_event.is_set():
                    self.observation = observation
                    self.inference_times.append(completed)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = (str(exc) if isinstance(exc, (VoiceError, FileNotFoundError))
                              else 'Webcam tracking unavailable · ' + type(exc).__name__)
                self.observation = WebcamObservation(0, time.monotonic(), reason=self.error)
        finally:
            self.ready = False
            if child is not None:
                with self.child_lock:
                    self._terminate(child)
                    for stream in (child.stdin,child.stdout):
                        if stream is not None:stream.close()

    @staticmethod
    def _terminate(child):
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=.5)
        except (OSError,subprocess.TimeoutExpired):
            pass

    def stop(self):
        self.stop_event.set()
        self.ready = False
        self.observation = WebcamObservation(0, 0., reason='Off')
        with self.child_lock:
            if self.child is not None:
                self._terminate(self.child)
        if self.is_alive():
            self.join(timeout=.7)
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break


class WebcamGaze(QObject):
    failed = Signal(object, str)

    def __init__(self, bounds, device_id, *, folder=LOCAL, calibrated=True):
        super().__init__()
        self.bounds, self.device_id = bounds, device_id
        self.folder, self.calibrated = folder, calibrated
        self.worker = None
        self.camera = self.capture = self.sink = None
        self.profile = None
        self.error = ''
        self.active = False
        self.seq = 0
        self.frame_clock = None
        self.last_media_time = None
        self.last_frame = 0.
        self.preview_image = None
        self.preview_captured = 0.
        self.preview_updated_at = 0.
        self.invalid_after = 0.
        self.camera_started = False
        self.camera_started_at = 0.
        self.last_seq = 0
        self.point = None
        self.smoother = Smoother(.08, scale=bounds.h)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._service)

    def start(self):
        if self.active:
            return
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError('The previous webcam session is still closing')
        from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
        from ..personalization.webcam import WebcamProfileStore
        device = next((d for d in QMediaDevices.videoInputs() if bytes(d.id()).hex() == self.device_id), None)
        if device is None:
            raise ValueError('Webcam not found. Choose a webcam in Settings.')
        if self.calibrated:
            self.profile = WebcamProfileStore(self.folder).load(self.bounds, self.device_id)
        self.camera = QCamera(device, self)
        formats = [f for f in device.videoFormats()
                   if f.resolution().width() <= 1280
                   and f.resolution().height() <= 720
                   and f.maxFrameRate() >= 20]
        if formats:
            self.camera.setCameraFormat(min(
                formats,
                key=lambda f: (abs(f.resolution().width() - 1280)
                               + abs(f.resolution().height() - 720),
                               -f.maxFrameRate())))
        self.capture = QMediaCaptureSession(self)
        self.sink = QVideoSink(self)
        self.capture.setCamera(self.camera)
        self.capture.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self._frame)
        self.camera.errorOccurred.connect(self._camera_error)
        self.worker = InferenceWorker()
        self.active = True
        self.error = ''
        self.started = time.monotonic()
        self.seq = self.last_seq = 0
        self.frame_clock = self.last_media_time = None
        self.last_frame = 0.
        self._clear_preview()
        self.preview_updated_at = 0.
        self.invalid_after = 0.
        self.worker.start()
        self.timer.start()

    def _service(self):
        if not self.active:
            return
        if self.worker.error:
            self._fail(self.worker.error)
        elif not self.worker.ready and time.monotonic() - self.started > 20:
            self._fail('Webcam tracking didn’t respond. Reconnect the webcam.')
        elif self.worker.ready and not self.camera.isActive():
            if self.last_frame:
                self._fail('Webcam disconnected')
            else:
                # start exactly once; an error never causes an automatic reopen.
                if not getattr(self, 'camera_started', False):
                    self.camera_started = True
                    self.camera_started_at = time.monotonic()
                    self.camera.start()
        if self.active and self.camera_started and time.monotonic() - (self.last_frame or self.camera_started_at) > 5:
            self._fail('Waiting for a fresh webcam frame')
        if self.active and self.camera_started and self.last_frame and time.monotonic()-max(self.worker.observation.captured,self.camera_started_at)>5:
            self._fail('Webcam tracking stopped responding. Reconnect the webcam.')

    def _camera_error(self, *args):
        if self.active:
            self._fail('Webcam unavailable. Check permissions and whether another app is using it.')

    def _fail(self, reason):
        self.error = reason
        self.failed.emit(self, reason)
        self.stop()

    def _clear_preview(self):
        self.preview_image = None
        self.preview_captured = 0.

    def diagnostics(self):
        worker = self.worker
        inference_times = getattr(worker, 'inference_times', ())
        processed_fps = 0.0
        if isinstance(inference_times, (tuple, list, deque)) and len(inference_times) >= 2:
            elapsed = inference_times[-1] - inference_times[0]
            if elapsed > 0:
                processed_fps = (len(inference_times) - 1) / elapsed
        return {'processed_fps': float(processed_fps)}

    def _frame(self, frame):
        if not self.active or not self.worker or not self.worker.ready:
            return
        now = time.monotonic()
        if not frame.isValid():
            self.invalid_after = now
            self._clear_preview()
            self.worker.observation = WebcamObservation(self.seq, now, reason='Camera frame unavailable')
            return
        media_time = frame.startTime() / 1_000_000
        if media_time < 0:
            self._fail('The camera isn’t reporting frame times')
            return
        if self.last_media_time is not None and media_time <= self.last_media_time:
            if media_time < self.last_media_time:
                self._fail('The camera clock reset. Reconnect the webcam.')
            return
        self.last_media_time = media_time
        self.frame_clock = min(self.frame_clock if self.frame_clock is not None else now-media_time, now-media_time)
        captured = min(now, self.frame_clock + media_time)
        if now - captured >= GAZE_TTL:
            return
        self.last_frame = now
        image = frame.toImage()
        if image.isNull():
            self.invalid_after = now
            self._clear_preview()
            return
        if not self.calibrated and now - self.preview_updated_at >= .10:
            self.preview_image = image.mirrored(True, False)
            self.preview_captured = captured
            self.preview_updated_at = now
        if image.width() > 1280 or image.height() > 720:
            image = image.scaled(1280, 720, Qt.KeepAspectRatio, Qt.FastTransformation)
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        raw = bytes(image.constBits())
        stride = image.bytesPerLine()
        pixels = b''.join(raw[row*stride:row*stride+width*3] for row in range(height))
        self.seq += 1
        self.worker.submit((self.seq, captured, width, height, pixels))

    def latest(self):
        if not self.active or not self.worker:
            return WebcamObservation(0, 0., reason=self.error or 'Webcam off')
        observation=self.worker.observation
        worker_error = getattr(self.worker, 'error', '')
        if worker_error:
            self._clear_preview()
            return WebcamObservation(observation.seq, observation.captured,
                                     reason=worker_error)
        if observation.captured<=self.invalid_after:
            return WebcamObservation(observation.seq, observation.captured,
                                     reason=observation.reason or 'Camera frame unavailable')
        return observation

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        observation = self.latest()
        if self.error:
            return None, self.error
        if observation.reason or not observation.features or not 0 <= now - observation.captured < GAZE_TTL:
            self.smoother.reset()
            return None, observation.reason or 'Waiting for fresh webcam gaze data'
        if self.profile is None:
            return None, 'Eyes visible · set up the webcam first'
        if not self.profile.supports(observation.features):
            self.smoother.reset()
            return None, 'Webcam · position outside setup range'
        point = self.profile.predict(observation.features)
        if not self.bounds.contains(*point):
            self.smoother.reset()
            return None, 'Webcam · gaze outside the screen'
        if observation.seq != self.last_seq or self.smoother.point is None:
            self.last_seq = observation.seq
            self.point = self.smoother.update(point, observation.captured)
        radius = self.profile.validation_radius
        return Gaze(*self.point, observation.captured, radius, 'webcam'), 'Webcam · ready as a backup'

    def stop(self):
        self.active = False
        self._clear_preview()
        self.timer.stop()
        if self.camera is not None:
            self.camera.stop()
        if self.worker is not None:
            self.worker.stop()
        for name in ('capture', 'sink', 'camera'):
            value = getattr(self, name)
            if value is not None:
                value.deleteLater()
                setattr(self, name, None)
        self.camera_started = False
        self.point = None
        self.smoother.reset()
