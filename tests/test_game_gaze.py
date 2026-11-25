import os
import struct
import time
import unittest
import uuid
from unittest.mock import Mock, patch

from context_operator.core.contracts import Gaze, GazeMeasurement, Rect
from context_operator.providers.game_gaze import (
    FLAG_ACTIVE, MAGIC, PACKET, PACKET_SIZE, VERSION, GameGazePublisher,
    eligible_gaze, gaze_packet,
)


class GameGazePacketTests(unittest.TestCase):
    bounds = Rect(-1920, 0, 3840, 1080)

    def gaze_pair(self, *, captured=10.0, x=-1919.0, y=100.0, source="iphone", supported=True):
        return (Gaze(x, y, captured, 42.0, source),
                GazeMeasurement(captured, (0.0,) * 22, (x, y), supported))

    def test_packet_is_fixed_little_endian_contract_and_preserves_capture_time(self):
        gaze, measurement = self.gaze_pair()
        active, packet = gaze_packet(9, gaze, measurement, self.bounds, 10.05, phone_present=True)
        self.assertTrue(active)
        self.assertEqual(len(packet), 128)
        self.assertEqual(struct.unpack('<8sIIQ9d32s', packet),
                         (b'COGAZE1\0', 1, 1, 9, 10., 10.05, -1919., 100.,
                          -1920., 0., 3840., 1080., 42., bytes(32)))

    def test_inactive_packet_zeroes_coordinates_and_geometry(self):
        active, packet = gaze_packet(2, *self.gaze_pair(), self.bounds, 11.0, phone_present=False)
        self.assertFalse(active)
        self.assertEqual(len(packet), 128)
        magic, version, flags, seq, *tail = PACKET.unpack(packet)
        self.assertEqual((magic, version, flags, seq), (MAGIC, VERSION, 0, 2))
        self.assertEqual(tail[:9], [0.0, 11.0] + [0.0] * 7)

    def test_rejects_stale_future_mismatched_unsupported_and_attention_data(self):
        cases = (
            (self.gaze_pair(captured=9.8), 10.0),
            (self.gaze_pair(captured=10.01), 10.0),
            ((Gaze(-1919, 100, 10.0), GazeMeasurement(9.99, (0.0,) * 22, (-1919, 100), True)), 10.01),
            (self.gaze_pair(supported=False), 10.01),
            (self.gaze_pair(source="mouse"), 10.01),
        )
        for (gaze, measurement), now in cases:
            with self.subTest(gaze=gaze, measurement=measurement):
                self.assertFalse(eligible_gaze(gaze, measurement, self.bounds, now,
                                               phone_present=True))

    def test_pointer_setup_pause_disconnect_and_off_desktop_revoke(self):
        gaze, measurement = self.gaze_pair()
        common = dict(phone_present=True)
        for gates in ({"pointer_mode": True}, {"mouse_control": True},
                      {"setup": True}, {"running": False},
                      {"phone_present": False}):
            with self.subTest(gates=gates):
                self.assertFalse(eligible_gaze(gaze, measurement, self.bounds, 10.01,
                                               **(common | gates)))
        outside, supported = self.gaze_pair(x=1920)
        self.assertFalse(eligible_gaze(outside, supported, self.bounds, 10.01, **common))

    def test_rejects_nonfinite_gaze_and_empty_desktop(self):
        common = dict(phone_present=True)
        self.assertFalse(eligible_gaze(*self.gaze_pair(x=float("nan")), self.bounds, 10.01, **common))
        self.assertFalse(eligible_gaze(*self.gaze_pair(x=0.0), Rect(0, 0, 0, 10), 10.01, **common))


@unittest.skipUnless(os.name == "nt", "named mapping ownership requires Windows")
class GameGazeWindowsTests(unittest.TestCase):
    def mapping_names(self):
        suffix = uuid.uuid4().hex
        prefix = rf"Local\ContextOperator.Gaze.Test.{suffix}"
        return patch.multiple(
            "context_operator.providers.game_gaze",
            MAPPING_NAME=prefix + ".Mapping",
            LOCK_NAME=prefix + ".Lock",
            WRITER_NAME=prefix + ".Writer",
        )

    def test_abi_clock_is_native_windows_uptime_not_performance_counter(self):
        import ctypes
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetTickCount64.restype = ctypes.c_uint64
        with self.mapping_names():
            publisher = GameGazePublisher()
            self.addCleanup(publisher.close)
            before = kernel.GetTickCount64() / 1000
            self.assertTrue(publisher.publish_inactive())
            after = kernel.GetTickCount64() / 1000
            stamp = struct.unpack('<8sIIQ9d32s', publisher._mapping[:128])[5]
        self.assertLessEqual(before, stamp)
        self.assertLessEqual(stamp, after)

    def test_named_mapping_publishes_and_enforces_single_writer(self):
        with self.mapping_names():
            publisher = GameGazePublisher()
            try:
                with self.assertRaises(RuntimeError):
                    GameGazePublisher()
                packet = publisher._mapping[:PACKET_SIZE]
                self.assertEqual(PACKET.unpack(packet)[2], 0)
                gaze = Gaze(50, 50, time.monotonic(), 25)
                measurement = GazeMeasurement(gaze.captured, (0.0,) * 22, (50, 50), True)
                self.assertEqual(publisher.publish(gaze, measurement, Rect(0, 0, 100, 100),
                                                   gaze.captured + 0.001, phone_present=True), (True, True))
                self.assertEqual(PACKET.unpack(publisher._mapping[:PACKET_SIZE])[2], FLAG_ACTIVE)
            finally:
                publisher.close()


    def test_open_and_close_wait_for_lock_but_frame_publishing_does_not(self):
        with patch('ctypes.WinDLL') as load_kernel, patch('mmap.mmap') as open_mapping:
            kernel = load_kernel.return_value
            kernel.CreateMutexW.side_effect = [1, 2]
            kernel.WaitForSingleObject.side_effect = [0, 0, 0x102, 0]
            publisher = GameGazePublisher()
            try:
                self.assertEqual(publisher.publish(None, None, None, 1., phone_present=False), (False, False))
            finally:
                publisher.close()
        self.assertEqual([call.args for call in kernel.WaitForSingleObject.call_args_list],
                         [(1, 0), (2, 100), (2, 0), (2, 100)])
        self.assertEqual(open_mapping.return_value.write.call_count, 2)

    def test_open_fails_when_the_previous_packet_cannot_be_cleared(self):
        with patch('ctypes.WinDLL') as load_kernel, patch('mmap.mmap') as open_mapping:
            kernel = load_kernel.return_value
            kernel.CreateMutexW.side_effect = [1, 2]
            kernel.WaitForSingleObject.side_effect = [0, 0x102, 0x102]
            with self.assertRaises(OSError):
                GameGazePublisher()
        open_mapping.return_value.write.assert_not_called()
        open_mapping.return_value.close.assert_called_once_with()


class ControllerGameGazeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QApplication.instance() or QApplication([])

    def controller(self, enabled, factory):
        from context_operator.controller import Controller
        from context_operator.settings import DEFAULTS
        controller = Controller(dict(DEFAULTS, game_gaze_enabled=enabled), game_gaze_factory=factory)
        controller.timer.stop()
        controller.win = Mock()
        controller.win.emergency.return_value = False
        controller.win.shift_down.return_value = False
        controller.win.bounds.return_value = Rect(0, 0, 100, 100)
        controller.running = True
        self.addCleanup(controller.close)
        return controller

    def test_disabled_setting_does_not_construct_publisher(self):
        factory = Mock()
        c = self.controller(False, factory)
        c.tick()
        factory.assert_not_called()
        self.assertIsNone(c.game_gaze_publisher)

    def test_nonrunning_tick_does_not_reclaim_writer_after_pause(self):
        publisher = Mock()
        publisher.publish.return_value = (False, True)
        factory = Mock(return_value=publisher)
        c = self.controller(True, factory)
        c.tick()
        factory.assert_called_once_with()
        publisher.close.assert_not_called()

        c.pause()
        publisher.close.assert_called_once_with()
        self.assertIsNone(c.game_gaze_publisher)
        c.tick()
        factory.assert_called_once_with()
        self.assertIsNone(c.game_gaze_publisher)

    def test_open_and_publish_failures_report_once_until_the_feed_writes_again(self):
        from types import SimpleNamespace
        from context_operator.controller import Controller
        publisher=Mock()
        publisher.publish.side_effect=[OSError('write failed'), (False, False), (False, True), OSError('write failed')]
        factory=Mock(side_effect=[RuntimeError('writer busy'), RuntimeError('writer busy'), publisher,
                                  publisher, publisher])
        controller=SimpleNamespace(settings={"game_gaze_enabled":True},game_gaze_factory=factory,
            game_gaze_publisher=None,game_gaze_unavailable=False,game_gaze_retry_at=0.,
            message=Mock(),phone=None,gaze=None,pointer_mode=False,mouse_control=False,
            setup_close=None)
        controller._close_game_gaze=lambda:Controller._close_game_gaze(controller)
        for now in (1.,2.,3.1,5.2,7.3,7.4,7.5):
            Controller._publish_game_gaze(controller,now,running=True)
        self.assertEqual(factory.call_count,4)
        self.assertEqual([call.args for call in controller.message.emit.call_args_list],
                         [('Game gaze feed','Unavailable: writer busy'),
                          ('Game gaze feed','Unavailable: write failed')])
        self.assertEqual(publisher.publish.call_args.args, (None, None, None, 7.5))
        self.assertEqual(publisher.publish.call_args.kwargs, dict(
            phone_present=False, pointer_mode=False, mouse_control=False, setup=False, running=True))
        self.assertTrue(controller.game_gaze_unavailable)


if __name__ == "__main__":
    unittest.main()
