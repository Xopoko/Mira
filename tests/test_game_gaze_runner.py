import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from context_operator.core.contracts import Gaze, GazeMeasurement, Rect
from context_operator.game_gaze_session import GameGazeSession
from context_operator.providers.learning import GazeLearning


class FakePublisher:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.calls = []
    def publish(self, *args, **kwargs):
        self.events.append('publish')
        self.calls.append((args, kwargs))
        return True
    def close(self):
        self.events.append('publisher.close')


class FakePhone:
    method = 'points'
    def __init__(self, events=None, *, gaze=None, measurement=None, error=None, worker=None):
        self.events = events if events is not None else []
        self.gaze = gaze
        self.measurement = measurement
        self.error = error
        self.worker = worker
        self.polls = 0
    def start(self):
        self.events.append('phone.start')
    def poll(self, _radius):
        self.polls += 1
        self.events.append('phone.poll')
        if self.error:
            raise self.error
        return self.gaze, ('Eyes visible' if self.gaze else 'Waiting for gaze'), 60
    def stop(self):
        self.events.append('phone.stop')


class FakeLearning:
    def __init__(self, events=None):
        self.events = events if events is not None else []
    def correct(self, gaze, _measurement, _now, _radius):
        self.events.append('learning.correct')
        return gaze


class GameGazeSessionTests(unittest.TestCase):
    bounds = Rect(-100, 0, 500, 300)

    def make_session(self, **phone_options):
        now = [20.0]
        events = []
        phone = FakePhone(events, **phone_options)
        learning = FakeLearning(events)
        publisher = FakePublisher(events)
        session = GameGazeSession(phone, learning, self.bounds, radius=40,
                                  publisher_factory=lambda: publisher,
                                  clock=lambda: now[0])
        return session, phone, publisher, events, now

    def test_construction_is_inactive_and_click_start_tick_publish_fresh_real_gaze(self):
        measurement = GazeMeasurement(20.0, (0.0,) * 22, (10.0, 20.0), True)
        gaze = Gaze(10.0, 20.0, 20.0, 30.0)
        session, phone, publisher, events, _ = self.make_session(
            gaze=gaze, measurement=measurement,
            worker=SimpleNamespace(is_alive=lambda: True, frames=1))
        self.assertFalse(session.running)
        self.assertEqual(phone.polls, 0)
        self.assertTrue(session.start())
        self.assertEqual(session.tick(), 'Gaze live · receiving fresh data')
        self.assertEqual(phone.polls, 1)
        self.assertEqual(events[:4], ['phone.start', 'phone.poll', 'learning.correct', 'publish'])
        self.assertEqual(publisher.calls[0][1]['profile_method'], 'points')
        self.assertTrue(publisher.calls[0][1]['running'])

    def test_missing_tracking_is_distinct_from_active_and_stop_closes_in_safe_order(self):
        worker = SimpleNamespace(is_alive=lambda: True, frames=3)
        session, _phone, _publisher, events, _ = self.make_session(worker=worker)
        self.assertTrue(session.start())
        self.assertIn('tracking unavailable', session.tick())
        session.stop()
        self.assertEqual(events[-2:], ['publisher.close', 'phone.stop'])

    def test_failed_phone_read_closes_publisher_and_stops_source(self):
        session, phone, _publisher, events, _ = self.make_session(error=OSError('reader failed'))
        self.assertTrue(session.start())
        status = session.tick()
        self.assertIn('Couldn’t read gaze', status)
        self.assertFalse(session.running)
        self.assertEqual(events[-2:], ['publisher.close', 'phone.stop'])
        self.assertEqual(phone.polls, 1)

    def test_read_only_bind_does_not_create_legacy_migration_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learning.json'
            key = 'profile-test'
            path.write_text(json.dumps({'version': 4, 'profile': key,
                                        'display': self.bounds.as_list(),
                                        'anchors': [], 'pose_anchors': [],
                                        'count': 0, 'click_anchors': []}), encoding='utf-8')
            learning = GazeLearning(path)
            learning.bind(key, self.bounds, read_only=True)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name('learning.before-safe-clicks.json').exists())
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['version'], 4)


if __name__ == '__main__':
    unittest.main()
