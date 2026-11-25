"""Headless MediaPipe child for numeric webcam gaze features.

The parent owns camera access and sends one RGB frame at a time over stdin.
This process never opens a camera and never writes frames or diagnostics to disk.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading

import numpy as np


SCHEMA = "contextoperator.webcam.v1"
MAGIC = b"WCF1"
HEADER = struct.Struct("<4sQdII")
MAX_WIDTH = 1280
MAX_HEIGHT = 720
FEATURE_COUNT = 8


@dataclass(frozen=True)
class FeatureResult:
    features: tuple[float, ...] | None
    reason: str


def _points_array(landmarks) -> np.ndarray | None:
    """Accept MediaPipe normalized landmarks or a numeric array."""
    try:
        if isinstance(landmarks, np.ndarray):
            points = np.asarray(landmarks, dtype=float)
        else:
            points = np.asarray([(point.x, point.y, point.z) for point in landmarks], dtype=float)
    except (TypeError, ValueError, AttributeError):
        return None
    if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] < 3:
        return None
    points = points[:, :3]
    if not np.isfinite(points).all():
        return None
    return points


def extract_features(landmarks, width: int, height: int,
                     blendshapes: dict[str, float] | None = None) -> FeatureResult:
    """Build the fixed eight-value webcam calibration vector.

    Order: mean horizontal and vertical iris offsets; normalized nose x/y
    offsets (yaw/pitch proxies); eye-line roll in radians; face bounding-box
    center x/y normalized to the frame; and inter-ocular width divided by frame
    width. Image x grows right and y grows down. No screen prediction is made.
    """
    if type(width) is not int or type(height) is not int or not (1 <= width <= MAX_WIDTH and 1 <= height <= MAX_HEIGHT):
        return FeatureResult(None, "invalid_frame_size")
    points = _points_array(landmarks)
    if points is None:
        return FeatureResult(None, "invalid_landmarks")

    pixels = points[:, :2] * np.array([width, height], dtype=float)
    eye_specs = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))
    eye_centers: list[np.ndarray] = []
    eye_axes: list[np.ndarray] = []
    iris_offsets: list[tuple[float, float]] = []
    eye_aspects: list[float] = []
    eye_spans: list[float] = []

    for corner_a, corner_b, upper, lower, iris_start in eye_specs:
        a, b = pixels[corner_a], pixels[corner_b]
        vector = b - a
        span = float(np.linalg.norm(vector))
        if not math.isfinite(span) or span < 8.0:
            return FeatureResult(None, "eyes_too_small")
        axis = vector / span
        vertical = np.array([-axis[1], axis[0]], dtype=float)
        center = (a + b) / 2.0
        iris = pixels[iris_start:iris_start + 5].mean(axis=0)
        offset = iris - center
        iris_offsets.append((float(offset @ axis / span), float(offset @ vertical / span)))
        eye_centers.append(center)
        eye_axes.append(axis)
        eye_spans.append(span)
        eye_aspects.append(float(abs((pixels[lower] - pixels[upper]) @ vertical) / span))

    if max(abs(value) for pair in iris_offsets for value in pair) > 0.75:
        return FeatureResult(None, "unstable_iris")

    shapes = blendshapes or {}
    blink = (max(float(shapes.get("eyeBlinkLeft", 0.0)),
                 float(shapes.get("eyeBlinkRight", 0.0))) > 0.6
             or min(eye_aspects) < 0.07)
    if blink:
        return FeatureResult(None, "blink")

    mean_span = float(np.mean(eye_spans))
    eye_line_center = np.mean(eye_centers, axis=0)
    nose = pixels[1]
    mean_axis = np.mean(eye_axes, axis=0)
    axis_norm = float(np.linalg.norm(mean_axis))
    if not math.isfinite(axis_norm) or axis_norm < 1e-6:
        return FeatureResult(None, "invalid_eye_line")
    mean_axis /= axis_norm
    face_width = float(np.linalg.norm(pixels[33] - pixels[263]) / width)
    face_min = pixels[:, :2].min(axis=0)
    face_max = pixels[:, :2].max(axis=0)
    face_center = (face_min + face_max) / 2.0

    values = (
        float(np.mean([pair[0] for pair in iris_offsets])),
        float(np.mean([pair[1] for pair in iris_offsets])),
        float((nose[0] - eye_line_center[0]) / mean_span),
        float((nose[1] - eye_line_center[1]) / mean_span),
        float(math.atan2(mean_axis[1], mean_axis[0])),
        float(face_center[0] / width),
        float(face_center[1] / height),
        face_width,
    )
    if len(values) != FEATURE_COUNT or not np.isfinite(values).all():
        return FeatureResult(None, "invalid_features")
    return FeatureResult(values, "")


def extract_single_face(face_landmarks, face_blendshapes, width: int, height: int) -> FeatureResult:
    """Reject missing or ambiguous detections, then extract one face."""
    if not face_landmarks:
        return FeatureResult(None, "face_not_found")
    if len(face_landmarks) != 1:
        return FeatureResult(None, "ambiguous_faces")
    blendshape_scores = {}
    if face_blendshapes:
        try:
            blendshape_scores = {item.category_name: float(item.score)
                                 for item in face_blendshapes[0]}
        except (TypeError, ValueError, AttributeError):
            return FeatureResult(None, "invalid_blendshapes")
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0
               for score in blendshape_scores.values()):
            return FeatureResult(None, "invalid_blendshapes")
    return extract_features(face_landmarks[0], width, height, blendshape_scores)


def _emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read_exact(stream, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            if allow_clean_eof and not chunks:
                return None
            raise EOFError
        chunks.extend(part)
    return bytes(chunks)


def _watch_parent() -> None:
    """Exit promptly if the application process disappears on Windows."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, os.getppid())  # SYNCHRONIZE
    if not handle:
        raise RuntimeError("parent_watch_unavailable")

    def supervise() -> None:
        try:
            while True:
                status = kernel32.WaitForSingleObject(handle, 200)
                if status != 0x00000102:  # WAIT_TIMEOUT is the only live result.
                    os._exit(0)
        finally:
            kernel32.CloseHandle(handle)

    threading.Thread(target=supervise, name="webcam-parent-watch", daemon=True).start()


def _run(model_path: Path) -> int:
    try:
        repository = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repository))
        _watch_parent()
        from context_operator.foundations.speech.runtime import check_policy
        check_policy()
        if not model_path.is_file():
            raise FileNotFoundError

        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception:
        _emit({"schema": SCHEMA, "error": "webcam runtime unavailable or not permitted"})
        return 2

    _emit({"schema": SCHEMA, "ready": True})
    last_seq = 0
    last_captured = -1.0
    last_timestamp_ms = 0
    try:
        while True:
            raw_header = _read_exact(sys.stdin.buffer, HEADER.size, allow_clean_eof=True)
            if raw_header is None:
                return 0
            magic, seq, captured, width, height = HEADER.unpack(raw_header)
            if (magic != MAGIC or seq <= last_seq or not math.isfinite(captured)
                    or captured < 0 or captured <= last_captured
                    or width < 1 or width > MAX_WIDTH or height < 1 or height > MAX_HEIGHT):
                _emit({"schema": SCHEMA, "error": "invalid webcam frame header"})
                return 2
            byte_count = width * height * 3
            raw_frame = _read_exact(sys.stdin.buffer, byte_count)
            check_policy()
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            last_timestamp_ms = max(last_timestamp_ms + 1, int(captured * 1000.0))
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame))
            result = landmarker.detect_for_video(image, last_timestamp_ms)
            feature_result = extract_single_face(
                result.face_landmarks, result.face_blendshapes, width, height)
            values = list(feature_result.features) if feature_result.features is not None else []
            _emit({"schema": SCHEMA, "seq": int(seq), "captured": float(captured),
                   "features": values, "reason": feature_result.reason})
            last_seq, last_captured = int(seq), float(captured)
    except EOFError:
        _emit({"schema": SCHEMA, "error": "truncated webcam frame"})
        return 2
    except Exception:
        _emit({"schema": SCHEMA, "error": "webcam inference failed"})
        return 2
    finally:
        landmarker.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        _emit({"schema": SCHEMA, "error": "invalid runtime arguments"})
        return 2
    return _run(Path(args.model))


if __name__ == "__main__":
    raise SystemExit(main())
