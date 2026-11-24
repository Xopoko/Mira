import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from context_operator.controller import Controller
from context_operator.core.click_learning import HumanClick
from context_operator.core.contracts import Gaze, GazeMeasurement, Rect
from context_operator.providers.game_gaze import (
    FLAG_ACTIVE,
    PACKET,
    eligible_gaze,
    pack_packet,
)
from context_operator.providers.learning import GazeLearning
from context_operator.settings import DEFAULTS


APP = QApplication.instance() or QApplication([])


def measurement(stamp, point=(25.0, 12.0)):
    features = np.zeros(22)
    return GazeMeasurement(stamp, tuple(features), point, True)


class SyntheticClicks:
    """Passive click-source fake with the same drain/start contract as runtime."""

    def __init__(self):
        self.events = []
        self.error = ''
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.events = []

    def drain(self):
        events, self.events = self.events, []
        return events

    def drain_motion(self):
        return []


class InMemoryGamePublisher:
    """Exercise the real eligibility and packet encoder without Win32 mappings."""

    def __init__(self):
        self.packets = []
        self.closed = False

    def publish(self, gaze, sample, desktop, now, **gates):
        active = eligible_gaze(gaze, sample, desktop, now, **gates)
        packet = pack_packet(
            len(self.packets) + 1,
            gaze.captured if gaze is not None else now,
            now,
            gaze.x if active else 0.0,
            gaze.y if active else 0.0,
            desktop if active else None,
            gaze.radius if active else 0.0,
            active=active,
        )
        self.packets.append(packet)

    def close(self):
        self.closed = True


class QuietRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.click_source = SyntheticClicks()
        settings = dict(
            DEFAULTS,
            quiet_mode=True,
            vision=False,
            ranker=False,
            anticipation=False,
            game_gaze_enabled=True,
        )
        self.publisher = InMemoryGamePublisher()
        with patch('context_operator.controller.LOCAL', Path(self.temp.name)):
            self.controller = Controller(
                settings,
                click_source=self.click_source,
                game_gaze_factory=lambda: self.publisher,
            )
        self.controller.timer.stop()
        self.controller.running = True
        self.controller.win = Mock()
        self.controller.win.emergency.return_value = False
        self.controller.win.shift_down.return_value = False
        self.controller.win.bounds.return_value = Rect(0, 0, 1000, 700)
        self.controller.learning = GazeLearning(Path(self.temp.name) / 'learning.json')
        self.controller.learning.bind('quiet-test-profile', Rect(0, 0, 1000, 700), np.ones(22))
        self.controller.phone = Mock()
        self.controller.phone.invalid_captured = None
        self.controller.scene = None

    def tearDown(self):
        self.controller.close()
        self.temp.cleanup()

    def tick_with(self, now, sample):
        self.controller.phone.measurement = sample
        self.controller.phone.poll.return_value = (
            Gaze(sample.point[0], sample.point[1], sample.captured, 230.0),
            'synthetic gaze',
            60,
        )
        with patch('context_operator.controller.time.monotonic', return_value=now):
            self.controller.tick()

    def test_quiet_keeps_real_click_learning_and_active_game_packet(self):
        self.controller._start_clicks()
        self.assertTrue(self.click_source.started)

        for now, captured in ((9.94, 9.92), (9.98, 9.96), (10.01, 9.99)):
            self.tick_with(now, measurement(captured))

        self.click_source.events = [HumanClick(30, 18, 10.02, 10.08)]
        self.tick_with(10.12, measurement(10.11))

        self.assertTrue(self.controller.settings['quiet_mode'])
        self.assertTrue(self.click_source.started)
        self.assertEqual(self.controller.click_counts['accepted'], 1)
        self.assertEqual(self.controller.learning.count, 1)
        self.assertFalse(self.controller.mouse_control)
        self.assertGreaterEqual(len(self.publisher.packets), 1)
        self.assertEqual(PACKET.unpack(self.publisher.packets[-1])[2], FLAG_ACTIVE)

    def test_quiet_does_not_bypass_stale_or_pause_export_gates(self):
        publisher = InMemoryGamePublisher()
        stale_sample = measurement(9.70)
        state = type('PublisherState', (), {})()
        state.settings = {'quiet_mode': True, 'game_gaze_enabled': True}
        state.game_gaze_factory = lambda: publisher
        state.game_gaze_publisher = None
        state.game_gaze_unavailable = False
        state.module = type('Signals', (), {'emit': Mock()})()
        state.phone = type('Phone', (), {'measurement': stale_sample})()
        state.gaze = Gaze(25, 12, 9.70, 230.0)
        state.pointer_mode = False
        state.mouse_control = False
        state.setup_close = None
        state._close_game_gaze = lambda: Controller._close_game_gaze(state)
        Controller._publish_game_gaze(
            state, 10.0, desktop=Rect(0, 0, 1000, 700), running=True
        )
        self.assertEqual(len(publisher.packets), 1)
        self.assertEqual(PACKET.unpack(publisher.packets[-1])[2], 0)

        Controller._publish_game_gaze(
            state, 10.01, desktop=Rect(0, 0, 1000, 700), running=False
        )
        Controller._close_game_gaze(state)
        self.assertTrue(publisher.closed)
        self.assertIsNone(state.game_gaze_publisher)
        self.assertEqual(len(publisher.packets), 1)


if __name__ == '__main__':
    unittest.main()
