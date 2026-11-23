"""Passive per-device keyboard observation through Windows Raw Input."""

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import hashlib
import threading
import time


WM_INPUT = 0xFF
WM_INPUT_DEVICE_CHANGE = 0xFE
WM_QUIT = 0x12
GIDC_REMOVAL = 2
RID_INPUT = 0x10000003
RIDI_DEVICENAME = 0x20000007
RIM_TYPEKEYBOARD = 1
RI_KEY_BREAK = 1
RI_KEY_E0 = 2
RI_KEY_E1 = 4
RIDEV_REMOVE = 1
RIDEV_INPUTSINK = 0x100
RIDEV_DEVNOTIFY = 0x2000
RAW_FLAGS = RIDEV_INPUTSINK | RIDEV_DEVNOTIFY
UINT_ERROR = 0xFFFFFFFF


class RAWINPUTDEVICE(C.Structure):
    _fields_ = [('usUsagePage', W.WORD), ('usUsage', W.WORD),
                ('dwFlags', W.DWORD), ('hwndTarget', W.HWND)]


class RAWINPUTHEADER(C.Structure):
    _fields_ = [('dwType', W.DWORD), ('dwSize', W.DWORD),
                ('hDevice', W.HANDLE), ('wParam', W.WPARAM)]


class RAWKEYBOARD(C.Structure):
    _fields_ = [('MakeCode', W.WORD), ('Flags', W.WORD),
                ('Reserved', W.WORD), ('VKey', W.WORD),
                ('Message', W.UINT), ('ExtraInformation', W.ULONG)]


class RAWINPUT(C.Structure):
    _fields_ = [('header', RAWINPUTHEADER), ('keyboard', RAWKEYBOARD)]


@dataclass(frozen=True)
class DeviceKeyEvent:
    device_id: str
    vk: int
    scan_code: int
    extended: int
    down: bool
    captured: float
    removed: bool = False


def device_id_for_name(name):
    """Return an opaque ID; never expose the raw device path to clients."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Unknown input device')
    return hashlib.sha256(name.strip().casefold().encode('utf-8')).hexdigest()


def valid_device_id(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in '0123456789abcdef' for c in value)


def validate_vk(vk):
    if isinstance(vk, bool) or not isinstance(vk, int):
        raise ValueError('Choose a single Windows key')
    allowed = (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A
               or vk in (0x08, 0x09, 0x0D, 0x20, 0x2D, 0x2E,
                         0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
                         0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
                         0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
                         0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0,
                         0xDB, 0xDC, 0xDD, 0xDE, 0xE2)
               or 0x70 <= vk <= 0x87)
    if not allowed or vk in (0x1B, 0x7B, 0x10, 0x11, 0x12, 0x5B, 0x5C):
        raise ValueError('Choose a regular key. Escape, F12, and modifier keys are reserved.')
    return vk


def valid_button_vk(vk):
    try:
        validate_vk(vk)
        return True
    except (TypeError, ValueError):
        return False


def _bind(user, kernel):
    user.CreateWindowExW.argtypes = [W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD,
                                     C.c_int, C.c_int, C.c_int, C.c_int,
                                     W.HWND, W.HMENU, W.HINSTANCE, W.LPVOID]
    user.CreateWindowExW.restype = W.HWND
    user.DestroyWindow.argtypes = [W.HWND]
    user.DestroyWindow.restype = W.BOOL
    user.RegisterRawInputDevices.argtypes = [C.POINTER(RAWINPUTDEVICE), W.UINT, W.UINT]
    user.RegisterRawInputDevices.restype = W.BOOL
    user.GetMessageW.argtypes = [C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT]
    user.GetMessageW.restype = C.c_int
    user.DispatchMessageW.argtypes = [C.POINTER(W.MSG)]
    user.DispatchMessageW.restype = W.LPARAM
    user.GetRawInputData.argtypes = [W.HANDLE, W.UINT, W.LPVOID, C.POINTER(W.UINT), W.UINT]
    user.GetRawInputData.restype = W.UINT
    user.GetRawInputDeviceInfoW.argtypes = [W.HANDLE, W.UINT, W.LPVOID, C.POINTER(W.UINT)]
    user.GetRawInputDeviceInfoW.restype = W.UINT
    user.GetMessageTime.argtypes = []
    user.GetMessageTime.restype = W.LONG
    user.PostThreadMessageW.argtypes = [W.DWORD, W.UINT, W.WPARAM, W.LPARAM]
    user.PostThreadMessageW.restype = W.BOOL
    kernel.GetCurrentThreadId.argtypes = []
    kernel.GetCurrentThreadId.restype = W.DWORD
    kernel.GetTickCount64.argtypes = []
    kernel.GetTickCount64.restype = C.c_ulonglong


def _device_id(user, handle):
    count = W.UINT()
    if user.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, None,
                                   C.byref(count)) == UINT_ERROR or not 1 < count.value <= 65536:
        return None
    buffer = C.create_unicode_buffer(count.value)
    if user.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, buffer,
                                   C.byref(count)) == UINT_ERROR or not buffer.value:
        return None
    return device_id_for_name(buffer.value)


def _read_key(user, handle):
    size = W.UINT()
    header_size = C.sizeof(RAWINPUTHEADER)
    if user.GetRawInputData(handle, RID_INPUT, None, C.byref(size), header_size) == UINT_ERROR:
        return None
    if not C.sizeof(RAWINPUT) <= size.value <= 65536:
        return None
    buffer = C.create_string_buffer(size.value)
    read = user.GetRawInputData(handle, RID_INPUT, buffer, C.byref(size), header_size)
    if read == UINT_ERROR or read < C.sizeof(RAWINPUT):
        return None
    raw = C.cast(buffer, C.POINTER(RAWINPUT)).contents
    if raw.header.dwType != RIM_TYPEKEYBOARD or not raw.header.hDevice:
        return None
    return raw.header.hDevice, raw.keyboard.VKey, raw.keyboard.MakeCode, raw.keyboard.Flags


class _RawKeyboardHub:
    """One registration and hidden message-only window per process."""

    def __init__(self):
        self.lock = threading.RLock()
        self.subscribers = {}
        self.worker = None
        self.ready = threading.Event()
        self.startup_error = ''
        self.runtime_error = ''
        self.thread_id = 0
        self.stopping = False

    def subscribe(self, owner, callback):
        with self.lock:
            if owner in self.subscribers and self.worker is not None and not self.stopping:
                self.subscribers[owner] = callback
            elif self.stopping:
                raise RuntimeError('The keyboard reader is still stopping')
            else:
                self.subscribers[owner] = callback
            if self.worker is None:
                self.ready.clear()
                self.startup_error = ''
                self.runtime_error = ''
                self.worker = threading.Thread(target=self._run, daemon=True,
                                               name='raw-keyboard-events')
                self.worker.start()
        if not self.ready.wait(2):
            self.unsubscribe(owner)
            raise RuntimeError('The keyboard reader didn’t start in time')
        if self.startup_error:
            self.unsubscribe(owner)
            raise RuntimeError('Keyboard reader: ' + self.startup_error)
        with self.lock:
            running = self.worker is not None and self.worker.is_alive() and not self.stopping
            failure = self.runtime_error
        if not running or failure:
            self.unsubscribe(owner)
            raise RuntimeError('Keyboard reader: ' + (failure or 'stopped'))

    @property
    def running(self):
        with self.lock:
            return self.worker is not None and self.worker.is_alive() and not self.stopping

    @property
    def error(self):
        with self.lock:
            return self.runtime_error or self.startup_error

    def unsubscribe(self, owner):
        with self.lock:
            self.subscribers.pop(owner, None)
            worker = self.worker
            if self.subscribers or worker is None:
                return
            self.stopping = True
            thread_id = self.thread_id
        if thread_id:
            user = C.WinDLL('user32', use_last_error=True)
            user.PostThreadMessageW.argtypes = [W.DWORD, W.UINT, W.WPARAM, W.LPARAM]
            user.PostThreadMessageW.restype = W.BOOL
            user.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)
        if worker is not threading.current_thread():
            worker.join(2)
            if worker.is_alive():
                raise RuntimeError('The keyboard reader is still shutting down')

    def _publish(self, event):
        with self.lock:
            callbacks = tuple(self.subscribers.values())
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                pass  # One client cannot break the native message loop.

    def _run(self):
        window = None
        registered = False
        failure = ''
        try:
            user = C.WinDLL('user32', use_last_error=True)
            kernel = C.WinDLL('kernel32', use_last_error=True)
            _bind(user, kernel)
            self.thread_id = kernel.GetCurrentThreadId()
            # STATIC is a system class; HWND_MESSAGE (-3) is never visible.
            window = user.CreateWindowExW(0, 'STATIC', '', 0, 0, 0, 0, 0,
                                          W.HWND(-3), None, None, None)
            if not window:
                raise C.WinError(C.get_last_error())
            device = RAWINPUTDEVICE(1, 6, RAW_FLAGS, window)
            if not user.RegisterRawInputDevices(C.byref(device), 1, C.sizeof(device)):
                raise C.WinError(C.get_last_error())
            registered = True
            self.ready.set()
            devices = {}
            msg = W.MSG()
            while True:
                with self.lock:
                    if self.stopping:
                        break
                result = user.GetMessageW(C.byref(msg), None, 0, 0)
                if result <= 0:
                    if result < 0:
                        raise C.WinError(C.get_last_error())
                    with self.lock:
                        if not self.stopping:
                            failure = 'The keyboard reader stopped'
                    break
                try:
                    if msg.message == WM_INPUT:
                        decoded = _read_key(user, msg.lParam)
                        if decoded:
                            handle, vk, scan_code, flags = decoded
                            identity = devices.get(handle)
                            if identity is None:
                                identity = _device_id(user, handle)
                                if identity:
                                    devices[handle] = identity
                            if identity:
                                tick = user.GetMessageTime() & UINT_ERROR
                                age = ((kernel.GetTickCount64() & UINT_ERROR) - tick) & UINT_ERROR
                                if age <= 500:
                                    self._publish(DeviceKeyEvent(
                                        identity, vk, scan_code, flags & (RI_KEY_E0 | RI_KEY_E1),
                                        not bool(flags & RI_KEY_BREAK), time.monotonic() - age / 1000))
                    elif msg.message == WM_INPUT_DEVICE_CHANGE and msg.wParam == GIDC_REMOVAL:
                        identity = devices.pop(C.c_size_t(msg.lParam).value, None)
                        if identity:
                            self._publish(DeviceKeyEvent(identity, 0, 0, 0, False,
                                                         time.monotonic(), removed=True))
                finally:
                    # Dispatch to the system window procedure for WM_INPUT cleanup.
                    user.DispatchMessageW(C.byref(msg))
        except Exception as exc:
            failure = type(exc).__name__
            if not self.ready.is_set():
                self.startup_error = failure
                self.ready.set()
        finally:
            if registered:
                device = RAWINPUTDEVICE(1, 6, RIDEV_REMOVE, None)
                user.RegisterRawInputDevices(C.byref(device), 1, C.sizeof(device))
            if window:
                user.DestroyWindow(window)
            with self.lock:
                if failure:
                    self.runtime_error = failure
                self.thread_id = 0
                self.worker = None
                self.stopping = False
            self.ready.set()


_HUB = _RawKeyboardHub()


class RawKeyboardEvents:
    """Subscribe to ephemeral events from the shared passive observer."""

    def __init__(self):
        self._started = False

    @property
    def running(self):
        return self._started and _HUB.running

    @property
    def error(self):
        return _HUB.error if self._started else ''

    def start(self, callback):
        if not callable(callback):
            raise TypeError('A keyboard handler is required')
        _HUB.subscribe(self, callback)
        self._started = True

    def stop(self):
        if self._started:
            self._started = False
            _HUB.unsubscribe(self)


class KeyboardButtonFilter:
    """The bound device must release before a new press can activate."""

    def __init__(self, vk, device_id, scan_code=None, extended=None, *, initially_down=False):
        self.vk = validate_vk(vk)
        if not valid_device_id(device_id):
            raise ValueError('No button device is bound')
        if scan_code is not None and (isinstance(scan_code, bool) or
                                      not isinstance(scan_code, int) or not 0 <= scan_code <= 0xffff):
            raise ValueError('Invalid button scan code')
        if extended is not None and extended not in (0, RI_KEY_E0, RI_KEY_E1):
            raise ValueError('Invalid key flag')
        self.device_id = device_id
        self.scan_code = scan_code
        self.extended = extended
        self.armed = not initially_down
        self.held = False

    def feed(self, event):
        if event.device_id != self.device_id:
            return False
        if event.removed:
            self.armed = False
            self.held = False
            return False
        if (event.vk != self.vk or
                self.scan_code is not None and event.scan_code != self.scan_code or
                self.extended is not None and event.extended != self.extended):
            return False
        if not event.down:
            self.armed = True
            self.held = False
            return False
        if not self.armed or self.held:
            return False
        self.held = True
        return True


class KeyboardButton:
    """Emit one activation for the bound key, leaving native input untouched."""

    def __init__(self, vk, device_id, scan_code=None, extended=None):
        self.gate = KeyboardButtonFilter(vk, device_id, scan_code, extended)
        self.vk = self.gate.vk
        self.device_id = self.gate.device_id
        self.scan_code = self.gate.scan_code
        self.extended = self.gate.extended
        self.observer = RawKeyboardEvents()
        self.emit = None
        self._error = ''
        self._accept_after = float('inf')
        self._lock = threading.Lock()

    @property
    def error(self):
        return self._error or self.observer.error

    @property
    def running(self):
        return self.emit is not None and self.observer.running

    def start(self, emit):
        if not callable(emit):
            raise TypeError('A button handler is required')
        with self._lock:
            if self.emit is not None and self.observer.running:
                return
            self.emit = None
            self._error = ''
            self._accept_after = float('inf')
            self.gate.armed = False
            self.gate.held = False
        try:
            self.observer.start(self._on_event)
            user = C.WinDLL('user32', use_last_error=True)
            user.GetAsyncKeyState.argtypes = [C.c_int]
            user.GetAsyncKeyState.restype = C.c_short
            down = bool(user.GetAsyncKeyState(self.vk) & 0x8000)
            with self._lock:
                self.gate.armed = not down
                self.gate.held = False
                self._accept_after = time.monotonic()
                self.emit = emit
            if not self.observer.running:
                raise RuntimeError('The keyboard reader stopped')
        except Exception:
            with self._lock:
                self.emit = None
            self.observer.stop()
            raise

    def stop(self):
        with self._lock:
            self.emit = None
            self._accept_after = float('inf')
        self.observer.stop()

    def _on_event(self, event):
        with self._lock:
            emit = self.emit
            age = time.monotonic() - event.captured
            activate = (emit is not None and event.captured >= self._accept_after
                        and 0 <= age < .5 and self.gate.feed(event))
        if activate:
            try:
                from ..core.contracts import InputEvent
                emit(InputEvent('activate', event.captured))
            except Exception as exc:
                self._error = type(exc).__name__
