import os
import struct
import unittest
import uuid
from unittest.mock import patch

from context_operator.core.contracts import Gaze, GazeMeasurement, Rect
from context_operator.providers.game_gaze import (
    FLAG_ACTIVE, MAGIC, PACKET, PACKET_SIZE, VERSION, GameGazePublisher,
    eligible_gaze, pack_packet,
)


class GameGazePacketTests(unittest.TestCase):
    bounds = Rect(-1920, 0, 3840, 1080)

    def gaze_pair(self, *, captured=10.0, x=-1919.0, y=100.0, source="iphone",
                  valid=True, supported=True):
        return (Gaze(x, y, captured, 42.0, source, valid),
                GazeMeasurement(captured, (0.0,) * 22, (x, y), supported))

    def test_packet_is_fixed_little_endian_contract_and_preserves_capture_time(self):
        gaze, measurement = self.gaze_pair()
        self.assertTrue(eligible_gaze(gaze, measurement, self.bounds, 10.05,
                                      phone_present=True))
        packet = pack_packet(9, gaze.captured, 10.05, gaze.x, gaze.y,
                             self.bounds, gaze.radius, active=True)
        self.assertEqual(len(packet), PACKET_SIZE)
        magic, version, flags, seq, *tail = PACKET.unpack(packet)
        self.assertEqual((magic, version, flags, seq), (MAGIC, VERSION, FLAG_ACTIVE, 9))
        self.assertEqual(tail[:2], [10.0, 10.05])
        self.assertEqual(tail[2:4], [-1919.0, 100.0])
        self.assertEqual(tail[-1], bytes(32))

    def test_inactive_packet_zeroes_coordinates_and_geometry(self):
        packet = pack_packet(2, 0.0, 11.0)
        self.assertEqual(len(packet), 128)
        magic, version, flags, seq, *tail = PACKET.unpack(packet)
        self.assertEqual((magic, version, flags, seq), (MAGIC, VERSION, 0, 2))
        self.assertEqual(tail[2:9], [0.0] * 7)

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

    def test_pointer_training_pause_disconnect_and_off_desktop_revoke(self):
        gaze, measurement = self.gaze_pair()
        common = dict(phone_present=True)
        for gates in ({"pointer_mode": True}, {"mouse_control": True},
                      {"training": True}, {"setup": True}, {"running": False},
                      {"phone_present": False}):
            with self.subTest(gates=gates):
                self.assertFalse(eligible_gaze(gaze, measurement, self.bounds, 10.01,
                                               **(common | gates)))
        outside, supported = self.gaze_pair(x=1920)
        self.assertFalse(eligible_gaze(outside, supported, self.bounds, 10.01, **common))

    def test_active_packet_rejects_nonfinite_or_invalid_geometry(self):
        with self.assertRaises(ValueError):
            pack_packet(1, 1.0, 1.01, float("nan"), 2, self.bounds, 10, active=True)
        with self.assertRaises(ValueError):
            pack_packet(1, 1.0, 1.01, 0, 0, Rect(0, 0, 0, 10), 10, active=True)
        with self.assertRaises(ValueError):
            pack_packet(1, 2.0, 1.99, 0, 0, self.bounds, 10, active=True)


@unittest.skipUnless(os.name == "nt", "named mapping ownership requires Windows")
class GameGazeWindowsTests(unittest.TestCase):
    def test_abi_clock_is_native_windows_uptime_not_performance_counter(self):
        import ctypes
        import time
        kernel=ctypes.WinDLL("kernel32")
        kernel.GetTickCount64.restype=ctypes.c_uint64
        before=kernel.GetTickCount64()/1000
        stamp=time.monotonic()
        after=kernel.GetTickCount64()/1000
        self.assertEqual(time.get_clock_info("monotonic").implementation,"GetTickCount64()")
        self.assertLessEqual(before,stamp)
        self.assertLessEqual(stamp,after)

    def test_named_mapping_publishes_and_enforces_single_writer(self):
        suffix = uuid.uuid4().hex
        prefix = rf"Local\ContextOperator.Gaze.Test.{suffix}"
        with patch.multiple(
            "context_operator.providers.game_gaze",
            MAPPING_NAME=prefix + ".Mapping",
            LOCK_NAME=prefix + ".Lock",
            WRITER_NAME=prefix + ".Writer",
        ):
            publisher = GameGazePublisher()
            try:
                with self.assertRaises(RuntimeError):
                    GameGazePublisher()
                packet = publisher._mapping[:PACKET_SIZE]
                self.assertEqual(PACKET.unpack(packet)[2], 0)
                gaze = Gaze(50, 50, __import__("time").monotonic(), 25)
                measurement = GazeMeasurement(gaze.captured, (0.0,) * 22, (50, 50), True)
                publisher.publish(gaze, measurement, Rect(0, 0, 100, 100),
                                  gaze.captured + 0.001, phone_present=True)
                self.assertEqual(PACKET.unpack(publisher._mapping[:PACKET_SIZE])[2], FLAG_ACTIVE)
            finally:
                publisher.close()


class ControllerGameGazeIntegrationTests(unittest.TestCase):
    def test_disabled_setting_does_not_construct_publisher(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from context_operator.controller import Controller
        factory = Mock()
        controller = SimpleNamespace(settings={"game_gaze_enabled": False},
                                     game_gaze_factory=factory, game_gaze_publisher=None,
                                     game_gaze_unavailable=False, _close_game_gaze=lambda: None)
        Controller._publish_game_gaze(controller, 1.0, running=False)
        factory.assert_not_called()

    def test_nonrunning_tick_does_not_reclaim_writer_after_pause(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from context_operator.controller import Controller
        publisher = Mock()
        controller = SimpleNamespace(settings={"game_gaze_enabled": True},
                                     game_gaze_factory=lambda: publisher,
                                     game_gaze_publisher=None, game_gaze_unavailable=False,
                                     module=SimpleNamespace(emit=Mock()), phone=None,
                                     gaze=Gaze(1, 1, 1), pointer_mode=False,
                                     mouse_control=False,
                                     setup_close=None, _close_game_gaze=Mock())
        Controller._publish_game_gaze(controller, 2.0, running=False)
        controller._close_game_gaze.assert_called_once()
        publisher.publish.assert_not_called()

    def test_transient_writer_conflict_retries_after_backoff(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from context_operator.controller import Controller
        publisher=Mock();factory=Mock(side_effect=[RuntimeError('writer busy'),publisher])
        controller=SimpleNamespace(settings={"game_gaze_enabled":True},game_gaze_factory=factory,
            game_gaze_publisher=None,game_gaze_unavailable=False,module=SimpleNamespace(emit=Mock()),
            phone=None,gaze=None,pointer_mode=False,mouse_control=False,
            setup_close=None,_close_game_gaze=Mock())
        Controller._publish_game_gaze(controller,1.,running=True)
        Controller._publish_game_gaze(controller,2.,running=True)
        self.assertEqual(factory.call_count,1)
        Controller._publish_game_gaze(controller,3.1,running=True)
        self.assertEqual(factory.call_count,2)
        publisher.publish.assert_called_once_with(
            controller.gaze, None, None, 3.1, phone_present=False,
            pointer_mode=False, mouse_control=False,
            setup=False, running=True)
        self.assertFalse(controller.game_gaze_unavailable)


if __name__ == "__main__":
    unittest.main()
