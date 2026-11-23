import ctypes
import threading
import unittest
from unittest.mock import Mock, PropertyMock, patch

from context_operator.providers import button


DEVICE = button.device_id_for_name('\\\\?\\HID#VID_1234&PID_5678#A')
OTHER = button.device_id_for_name('\\\\?\\HID#VID_1234&PID_5678#B')


def key(device=DEVICE, vk=0x0D, scan=0x1C, extended=0, down=True,
        removed=False, captured=12.5):
    return button.DeviceKeyEvent(device, vk, scan, extended, down, captured, removed)


class KeyboardButtonTests(unittest.TestCase):
    def test_device_identity_is_opaque_and_normalized(self):
        self.assertEqual(DEVICE, button.device_id_for_name(' \\\\?\\hid#vid_1234&pid_5678#a '))
        self.assertNotEqual(DEVICE, OTHER)
        self.assertTrue(button.valid_device_id(DEVICE))
        self.assertFalse(button.valid_device_id('not-a-device'))

    def test_other_keyboard_enter_never_activates_or_changes_gate(self):
        gate = button.KeyboardButtonFilter(0x0D, DEVICE, 0x1C, 0)
        self.assertFalse(gate.feed(key(OTHER)))
        self.assertFalse(gate.feed(key(OTHER, down=False)))
        self.assertTrue(gate.feed(key()))
        self.assertFalse(gate.feed(key()))
        self.assertFalse(gate.feed(key(OTHER, down=False)))
        self.assertFalse(gate.feed(key()))
        self.assertFalse(gate.feed(key(down=False)))
        self.assertTrue(gate.feed(key()))

    def test_scan_and_extended_are_part_of_binding(self):
        gate = button.KeyboardButtonFilter(0x0D, DEVICE, 0x1C, 0)
        self.assertFalse(gate.feed(key(scan=0x5A)))
        self.assertFalse(gate.feed(key(extended=button.RI_KEY_E0)))
        self.assertTrue(gate.feed(key()))

    def test_held_start_and_removal_require_bound_release(self):
        gate = button.KeyboardButtonFilter(0x0D, DEVICE, initially_down=True)
        self.assertFalse(gate.feed(key()))
        self.assertFalse(gate.feed(key(down=False)))
        self.assertTrue(gate.feed(key()))
        self.assertFalse(gate.feed(key(removed=True)))
        self.assertFalse(gate.feed(key()))
        self.assertFalse(gate.feed(key(OTHER, down=False)))
        self.assertFalse(gate.feed(key(down=False)))
        self.assertTrue(gate.feed(key()))

    def test_button_start_stop_and_callback_failure(self):
        source = button.KeyboardButton(0x0D, DEVICE, 0x1C, 0)
        self.assertEqual((source.vk, source.device_id, source.scan_code, source.extended),
                         (0x0D, DEVICE, 0x1C, 0))
        user = Mock()
        user.GetAsyncKeyState.return_value = 0
        events = []
        with patch.object(button.C, 'WinDLL', return_value=user), patch.object(
                source.observer, 'start') as start, patch.object(source.observer, 'stop') as stop, \
                patch.object(button.RawKeyboardEvents, 'running', new_callable=PropertyMock,
                             return_value=True), patch.object(button.time, 'monotonic', return_value=12.5):
            source.start(events.append)
            start.assert_called_once()
            source._on_event(key(OTHER))
            source._on_event(key())
            source._on_event(key())
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].kind, 'activate')
            self.assertEqual(events[0].captured, 12.5)
            source.stop()
            stop.assert_called_once()
            source._on_event(key(down=False))
            self.assertEqual(len(events), 1)

    def test_observer_starts_disarmed_before_state_snapshot_and_stale_release_is_ignored(self):
        source = button.KeyboardButton(0x0D, DEVICE, 0x1C, 0)
        user = Mock()
        user.GetAsyncKeyState.return_value = 0
        events = []
        order = []

        def subscribe(callback):
            order.append('observer')
            callback(key())  # A press during registration must not activate.

        def get_state(_vk):
            order.append('state')
            return 0

        user.GetAsyncKeyState.side_effect = get_state
        with patch.object(button.C, 'WinDLL', return_value=user), patch.object(
                source.observer, 'start', side_effect=subscribe), patch.object(
                source.observer, 'stop'), patch.object(
                button.RawKeyboardEvents, 'running', new_callable=PropertyMock,
                return_value=True), patch.object(button.time, 'monotonic', return_value=12.5):
            source.start(events.append)
            self.assertEqual(order, ['observer', 'state'])
            self.assertFalse(events)
            source._on_event(key())
            self.assertEqual(len(events), 1)
            with patch.object(button.time, 'monotonic', return_value=13.1):
                source._on_event(key(down=False))  # Stale up cannot rearm.
                source._on_event(key(captured=13.1))
                self.assertEqual(len(events), 1)
                source._on_event(key(down=False, captured=13.1))
                source._on_event(key(captured=13.1))
                self.assertEqual(len(events), 2)
            source.stop()

    def test_validation_and_passive_registration_flags(self):
        self.assertEqual(button.RAW_FLAGS, 0x2100)
        self.assertEqual(button.RAW_FLAGS & 0x30, 0)  # never NOLEGACY
        self.assertEqual(button.RAW_FLAGS & 0x200, 0)  # never NOHOTKEYS
        for vk in (0x30, 0x41, 0x0D, 0x20, 0x70):
            self.assertTrue(button.valid_button_vk(vk))
        for vk in (0x10, 0x1B, 0x7B, True, 'A'):
            self.assertFalse(button.valid_button_vk(vk))
        with self.assertRaises(ValueError):
            button.KeyboardButton(0x0D, 'old-global-key-only-binding')

    def test_typed_native_keyboard_decode_and_hashed_device_name(self):
        native = button.RAWINPUT()
        native.header.dwType = button.RIM_TYPEKEYBOARD
        native.header.hDevice = 42
        native.keyboard.MakeCode = 0x1C
        native.keyboard.VKey = 0x0D
        native.keyboard.Flags = button.RI_KEY_BREAK | button.RI_KEY_E0
        path = '\\\\?\\HID#VID_1234&PID_5678#A'
        user = Mock()

        def raw_data(handle, command, buffer, count, header_size):
            self.assertEqual((handle, command, header_size),
                             (77, button.RID_INPUT, ctypes.sizeof(button.RAWINPUTHEADER)))
            count._obj.value = ctypes.sizeof(native)
            if buffer is not None:
                ctypes.memmove(buffer, ctypes.byref(native), ctypes.sizeof(native))
            return ctypes.sizeof(native)

        def device_info(handle, command, buffer, count):
            self.assertEqual((handle, command), (42, button.RIDI_DEVICENAME))
            count._obj.value = len(path) + 1
            if buffer is not None:
                ctypes.memmove(buffer, ctypes.create_unicode_buffer(path),
                               (len(path) + 1) * ctypes.sizeof(ctypes.c_wchar))
            return len(path) + 1

        user.GetRawInputData.side_effect = raw_data
        user.GetRawInputDeviceInfoW.side_effect = device_info
        self.assertEqual(button._read_key(user, 77),
                         (42, 0x0D, 0x1C, button.RI_KEY_BREAK | button.RI_KEY_E0))
        self.assertEqual(button._device_id(user, 42), DEVICE)

    def test_one_reader_shared_by_observers_and_stopped_after_last(self):
        hub = button._RawKeyboardHub()
        user = Mock()
        kernel = Mock()
        kernel.GetCurrentThreadId.return_value = 123
        user.CreateWindowExW.return_value = 321
        user.RegisterRawInputDevices.return_value = True
        entered = threading.Event()
        release = threading.Event()

        def get_message(_msg, _hwnd, _min, _max):
            entered.set()
            release.wait(2)
            return 0

        user.GetMessageW.side_effect = get_message
        with patch.object(button.C, 'WinDLL', side_effect=lambda name, **_: user if name == 'user32' else kernel):
            first, second = object(), object()
            hub.subscribe(first, Mock())
            self.assertTrue(entered.wait(1))
            hub.subscribe(second, Mock())
            self.assertEqual(user.RegisterRawInputDevices.call_count, 1)
            registered = ctypes.cast(user.RegisterRawInputDevices.call_args_list[0].args[0],
                                     ctypes.POINTER(button.RAWINPUTDEVICE)).contents
            self.assertEqual((registered.usUsagePage, registered.usUsage, registered.dwFlags),
                             (1, 6, button.RAW_FLAGS))
            self.assertEqual(user.CreateWindowExW.call_args.args[8].value,
                             ctypes.c_void_p(-3).value)
            hub.unsubscribe(first)
            self.assertEqual(user.PostThreadMessageW.call_count, 0)
            release.set()
            hub.unsubscribe(second)
            self.assertEqual(user.RegisterRawInputDevices.call_count, 2)
            removed = ctypes.cast(user.RegisterRawInputDevices.call_args_list[1].args[0],
                                  ctypes.POINTER(button.RAWINPUTDEVICE)).contents
            self.assertEqual(removed.dwFlags, button.RIDEV_REMOVE)
            self.assertFalse(removed.hwndTarget)
            self.assertEqual(user.DestroyWindow.call_count, 1)

    def test_unexpected_reader_exit_is_reported_and_same_observer_can_restart(self):
        hub = button._RawKeyboardHub()
        user = Mock()
        kernel = Mock()
        kernel.GetCurrentThreadId.return_value = 123
        user.CreateWindowExW.return_value = 321
        user.RegisterRawInputDevices.return_value = True
        exits = [threading.Event(), threading.Event()]
        entered = [threading.Event(), threading.Event()]
        calls = 0

        def get_message(*_args):
            nonlocal calls
            index = calls
            calls += 1
            entered[index].set()
            exits[index].wait(2)
            return 0

        user.GetMessageW.side_effect = get_message
        observer = button.RawKeyboardEvents()
        source = button.KeyboardButton(0x0D, DEVICE)
        source.observer = observer
        with patch.object(button, '_HUB', hub), patch.object(
                button.C, 'WinDLL', side_effect=lambda name, **_: user if name == 'user32' else kernel):
            observer.start(Mock())
            self.assertTrue(observer.running)
            self.assertTrue(entered[0].wait(1))
            first_worker = hub.worker
            exits[0].set()
            first_worker.join(1)
            self.assertFalse(observer.running)
            self.assertIn('The keyboard reader stopped', observer.error)
            self.assertEqual(source.error, observer.error)
            observer.start(Mock())
            self.assertTrue(entered[1].wait(1))
            self.assertTrue(observer.running)
            self.assertEqual(user.RegisterRawInputDevices.call_count, 3)
            exits[1].set()
            observer.stop()
            self.assertEqual(user.RegisterRawInputDevices.call_count, 4)


if __name__ == '__main__':
    unittest.main()
