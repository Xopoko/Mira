"""Opt-in, numeric-only shared-memory gaze export for local game integrations.

Importing this module has no Windows or hardware side effects. The publisher
acquires a process-wide writer lease and publishes fixed-size packets guarded
by a short named mutex.
"""
from __future__ import annotations

import math
import struct
import time

from ..core.contracts import GAZE_TTL, Gaze, GazeMeasurement, Rect

MAPPING_NAME = r"Local\ContextOperator.Gaze.v1"
LOCK_NAME = r"Local\ContextOperator.Gaze.v1.Lock"
WRITER_NAME = r"Local\ContextOperator.Gaze.v1.Writer"
PACKET = struct.Struct("<8sIIQ9d32s")
PACKET_SIZE = 128
MAGIC = b"COGAZE1\0"
VERSION = 1
FLAG_ACTIVE = 1


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _desktop_values(desktop):
    if not isinstance(desktop, Rect):
        return None
    try:
        values = tuple(float(v) for v in (desktop.x, desktop.y, desktop.w, desktop.h))
    except (TypeError, ValueError, OverflowError):
        return None
    return values if all(math.isfinite(v) for v in values) else None


def pack_packet(sequence, captured_s, published_s, gaze_x=0.0, gaze_y=0.0,
                desktop=None, radius=0.0, *, active=False):
    """Build one validated 128-byte packet; inactive packets carry zero gaze."""
    if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < 2**64:
        raise ValueError("sequence must be an unsigned 64-bit integer")
    if not _finite(captured_s) or not _finite(published_s):
        raise ValueError("timestamps must be finite")
    captured_s, published_s = float(captured_s), float(published_s)
    if active:
        desktop_values = _desktop_values(desktop)
        if desktop_values is None or not all(_finite(v) for v in (gaze_x, gaze_y, radius)):
            raise ValueError("active gaze values and desktop bounds must be finite")
        gaze_x, gaze_y, radius = float(gaze_x), float(gaze_y), float(radius)
        age = float(published_s) - float(captured_s)
        if age < 0 or age >= GAZE_TTL - 1e-12:
            raise ValueError("active gaze timestamp is future-dated or stale")
        desktop_x, desktop_y, desktop_w, desktop_h = desktop_values
        if desktop_w <= 0 or desktop_h <= 0 or radius <= 0:
            raise ValueError("desktop dimensions and gaze radius must be positive")
        if not (desktop_x <= float(gaze_x) < desktop_x + desktop_w
                and desktop_y <= float(gaze_y) < desktop_y + desktop_h):
            raise ValueError("gaze point is outside the physical desktop")
        values = (float(captured_s), float(published_s), float(gaze_x), float(gaze_y),
                  desktop_x, desktop_y, desktop_w, desktop_h,
                  float(radius))
        flags = FLAG_ACTIVE
    else:
        values = (float(captured_s), float(published_s), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        flags = 0
    return PACKET.pack(MAGIC, VERSION, flags, sequence, *values, bytes(32))


def eligible_gaze(gaze, measurement, desktop, now, *, phone_present,
                  pointer_mode=False, mouse_control=False,
                  setup=False, running=True):
    """Return whether this frame is eligible for export from the real gaze source."""
    if (not running or not phone_present or pointer_mode or mouse_control or setup):
        return False
    if not isinstance(gaze, Gaze) or not isinstance(measurement, GazeMeasurement):
        return False
    desktop_values = _desktop_values(desktop)
    if desktop_values is None:
        return False
    if gaze.source != "iphone" or not measurement.supported:
        return False
    try:
        if len(measurement.features) != 22 or len(measurement.point) != 2:
            return False
        if not all(_finite(value) for value in tuple(measurement.features) + tuple(measurement.point)):
            return False
    except (TypeError, ValueError):
        return False
    if not _finite(now) or not _finite(gaze.captured) or not _finite(measurement.captured):
        return False
    age = float(now) - float(gaze.captured)
    if age < 0 or age >= GAZE_TTL - 1e-12:
        return False
    # Corrected gaze keeps the source frame timestamp. A mismatch means that
    # the point and the supported TrueDepth measurement do not describe one frame.
    if abs(float(gaze.captured) - float(measurement.captured)) > 1e-6:
        return False
    if not all(_finite(v) for v in (gaze.x, gaze.y, gaze.radius)):
        return False
    desktop_x, desktop_y, desktop_w, desktop_h = desktop_values
    gaze_x, gaze_y, radius = float(gaze.x), float(gaze.y), float(gaze.radius)
    if desktop_w <= 0 or desktop_h <= 0 or radius <= 0:
        return False
    return desktop_x <= gaze_x < desktop_x + desktop_w and desktop_y <= gaze_y < desktop_y + desktop_h


class GameGazePublisher:
    """Windows named mapping publisher. Construction explicitly claims writer ownership."""

    def __init__(self):
        import ctypes
        import mmap
        import os
        if os.name != "nt":
            raise OSError("game gaze shared memory is available only on Windows")
        # ABI v1 timestamps explicitly use Windows uptime, as PhoneGaze's Python
        # 3.11 monotonic clock does. Never silently switch epochs after a Python upgrade.
        if time.get_clock_info("monotonic").implementation != "GetTickCount64()":
            raise RuntimeError("game gaze ABI v1 requires the GetTickCount64 monotonic clock (Python 3.11)")
        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        ctypes.set_last_error(0)
        self._writer = self._kernel32.CreateMutexW(None, False, WRITER_NAME)
        if not self._writer:
            raise self._last_error("CreateMutexW(writer)")
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS; mutex waits are recursive per thread.
            self._kernel32.CloseHandle(self._writer)
            self._writer = None
            raise RuntimeError("another ContextOperator gaze publisher owns the mapping")
        wait = self._kernel32.WaitForSingleObject(self._writer, 0)
        if wait not in (0, 0x80):
            self._kernel32.CloseHandle(self._writer)
            self._writer = None
            raise RuntimeError("another ContextOperator gaze publisher owns the mapping")
        self._writer_owned = True
        self._lock = None
        self._mapping = None
        self._sequence = 0
        try:
            self._lock = self._kernel32.CreateMutexW(None, False, LOCK_NAME)
            if not self._lock:
                raise self._last_error("CreateMutexW(lock)")
            self._mapping = mmap.mmap(-1, PACKET_SIZE, tagname=MAPPING_NAME, access=mmap.ACCESS_WRITE)
            # Make stale active state from a prior owner impossible to observe.
            self.publish_inactive(time.monotonic())
        except Exception:
            self.close()
            raise

    def _configure_api(self):
        c = self._ctypes
        k = self._kernel32
        k.CreateMutexW.argtypes = (c.c_void_p, c.c_int, c.c_wchar_p)
        k.CreateMutexW.restype = c.c_void_p
        k.WaitForSingleObject.argtypes = (c.c_void_p, c.c_uint32)
        k.WaitForSingleObject.restype = c.c_uint32
        k.ReleaseMutex.argtypes = (c.c_void_p,)
        k.ReleaseMutex.restype = c.c_int
        k.CloseHandle.argtypes = (c.c_void_p,)
        k.CloseHandle.restype = c.c_int

    def _last_error(self, operation):
        return OSError(self._ctypes.get_last_error(), operation)

    def _publish(self, packet, timeout=0):
        if self._mapping is None or self._lock is None:
            raise RuntimeError("game gaze mapping is closed")
        wait = self._kernel32.WaitForSingleObject(self._lock, timeout)
        if wait == 0x102:  # WAIT_TIMEOUT
            return False
        if wait not in (0, 0x80):
            raise self._last_error("WaitForSingleObject(lock)")
        try:
            self._mapping.seek(0)
            self._mapping.write(packet)
            self._sequence += 1
            return True
        finally:
            self._kernel32.ReleaseMutex(self._lock)

    def publish(self, gaze, measurement, desktop, now, *, phone_present,
                pointer_mode=False, mouse_control=False,
                setup=False, running=True):
        active = eligible_gaze(gaze, measurement, desktop, now,
                               phone_present=phone_present,
                               pointer_mode=pointer_mode,
                               mouse_control=mouse_control,
                               setup=setup, running=running)
        if active:
            packet = pack_packet(self._sequence + 1, gaze.captured, now,
                                 gaze.x, gaze.y, desktop, gaze.radius, active=True)
        else:
            packet = pack_packet(self._sequence + 1, 0.0, now, active=False)
        return active, self._publish(packet)

    def publish_inactive(self, now=None):
        now = time.monotonic() if now is None else now
        return self._publish(pack_packet(self._sequence + 1, 0.0, now, active=False), timeout=100)

    def close(self):
        """Revoke the last value, close handles, and release the writer lease."""
        try:
            if getattr(self, "_mapping", None) is not None and getattr(self, "_lock", None):
                self.publish_inactive()
        finally:
            mapping, self._mapping = getattr(self, "_mapping", None), None
            if mapping is not None:
                mapping.close()
            for name in ("_lock", "_writer"):
                handle = getattr(self, name, None)
                if handle:
                    if name == "_writer" and getattr(self, "_writer_owned", False):
                        self._kernel32.ReleaseMutex(handle)
                        self._writer_owned = False
                    self._kernel32.CloseHandle(handle)
                    setattr(self, name, None)
