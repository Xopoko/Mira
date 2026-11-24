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
        self.result = 'inactive'
    def publish(self, *args, **kwargs):
        self.events.append('publish')
        self.calls.append((args, kwargs))
        return self.result
    def close(self):
        self.events.append('publisher.close')


class FakePhone:
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
        publisher.result = 'active'
        self.assertEqual(session.tick(), 'Gaze live · receiving fresh data')
        self.assertEqual(phone.polls, 1)
        self.assertEqual(events[:4], ['phone.start', 'phone.poll', 'learning.correct', 'publish'])
        self.assertEqual(publisher.calls[0][0], (gaze, measurement, self.bounds, 20.0))
        self.assertTrue(publisher.calls[0][1]['phone_present'])
        self.assertTrue(publisher.calls[0][1]['running'])
        publisher.result = 'busy'
        self.assertEqual(session.tick(), 'Gaze live · output busy, update skipped')

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

    def test_bind_loads_saved_corrections_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learning.json'
            key = 'profile-test'
            path.write_text(json.dumps({'version': 5, 'profile': key,
                                        'display': self.bounds.as_list(),
                                        'count': 1, 'click_accepted': 1,
                                        'click_anchors': [{'features': [0.] * 22, 'offset': [5., 0.],
                                                           'target': [10., 20.], 'count': 1, 'revision': 1}]}),
                            encoding='utf-8')
            before = path.read_bytes()
            learning = GazeLearning(path)
            learning.bind(key, self.bounds)
            self.assertEqual(learning.count, 1)
            self.assertEqual(learning.click_accepted, 1)
            self.assertEqual(len(learning.click_anchors), 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
