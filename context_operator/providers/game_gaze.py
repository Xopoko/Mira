"""Opt-in, numeric-only shared-memory gaze export for local game integrations.

Importing this module has no Windows or hardware side effects. The publisher
acquires a process-wide writer lease and publishes fixed-size packets guarded
by a short named mutex.
"""
from __future__ import annotations

import math
import struct
import time

from ..core.contracts import GAZE_TTL

MAPPING_NAME = r"Local\ContextOperator.Gaze.v1"
LOCK_NAME = r"Local\ContextOperator.Gaze.v1.Lock"
WRITER_NAME = r"Local\ContextOperator.Gaze.v1.Writer"
PACKET = struct.Struct("<8sIIQ9d32s")
PACKET_SIZE = 128
MAGIC = b"COGAZE1\0"
VERSION = 1
FLAG_ACTIVE = 1


def eligible_gaze(gaze, measurement, desktop, now, *, phone_present,
                  pointer_mode=False, mouse_control=False,
                  setup=False, running=True):
    """Whether this frame is the fresh, supported, on-screen gaze of the iPhone."""
    if not running or not phone_present or pointer_mode or mouse_control or setup:
        return False
    if gaze is None or measurement is None or desktop is None:
        return False
    if gaze.source != "iphone" or not measurement.supported:
        return False
    # Corrected gaze keeps the source frame timestamp. A mismatch means that
    # the point and the supported TrueDepth measurement do not describe one frame.
    if abs(gaze.captured - measurement.captured) > 1e-6:
        return False
    values = (gaze.x, gaze.y, gaze.radius, gaze.captured, now, desktop.x, desktop.y, desktop.w, desktop.h)
    if not all(math.isfinite(value) for value in values) or desktop.w <= 0 or desktop.h <= 0 or gaze.radius <= 0:
        return False
    # A reader compares against 200 ms too; rounding must not hand it a packet on the limit.
    if not 0 <= now - gaze.captured < GAZE_TTL - 1e-12:
        return False
    return desktop.x <= gaze.x < desktop.x + desktop.w and desktop.y <= gaze.y < desktop.y + desktop.h


def gaze_packet(sequence, gaze, measurement, desktop, now, **gates):
    """Whether this frame is active, and its 128-byte packet. An inactive packet carries zero gaze and geometry."""
    active = eligible_gaze(gaze, measurement, desktop, now, **gates)
    values = ((gaze.captured, now, gaze.x, gaze.y, desktop.x, desktop.y, desktop.w, desktop.h, gaze.radius)
              if active else (0., now, 0., 0., 0., 0., 0., 0., 0.))
    return active, PACKET.pack(MAGIC, VERSION, FLAG_ACTIVE if active else 0, sequence, *values, bytes(32))


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
            # A prior owner's last packet must not stay readable as active.
            if not self.publish_inactive(time.monotonic()):
                raise OSError("a reader is holding the game gaze mapping")
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
        active, packet = gaze_packet(self._sequence + 1, gaze, measurement, desktop, now,
                                     phone_present=phone_present, pointer_mode=pointer_mode,
                                     mouse_control=mouse_control, setup=setup, running=running)
        return active, self._publish(packet)

    def publish_inactive(self, now=None):
        now = time.monotonic() if now is None else now
        packet = gaze_packet(self._sequence + 1, None, None, None, now, phone_present=False)[1]
        return self._publish(packet, timeout=100)

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
