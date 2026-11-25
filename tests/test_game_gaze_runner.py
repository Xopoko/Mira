import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_operator.core.contracts import Gaze, GazeMeasurement, Rect
from context_operator.game_gaze_session import GameGazeSession, create_saved_session


class FakePublisher:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.calls = []
        self.result = (False, True)
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

    def test_session_waits_for_start_then_forwards_gaze_and_reports_publish_status(self):
        measurement = GazeMeasurement(20.0, (0.0,) * 22, (10.0, 20.0), True)
        gaze = Gaze(10.0, 20.0, 20.0, 30.0)
        session, phone, publisher, events, _ = self.make_session(
            gaze=gaze, measurement=measurement,
            worker=SimpleNamespace(is_alive=lambda: True, frames=1))
        self.assertFalse(session.running)
        self.assertEqual(phone.polls, 0)
        self.assertTrue(session.start())
        publisher.result = (True, True)
        self.assertEqual(session.tick(), 'Gaze live · receiving fresh data')
        self.assertEqual(phone.polls, 1)
        self.assertEqual(events[:4], ['phone.start', 'phone.poll', 'learning.correct', 'publish'])
        self.assertEqual(publisher.calls[0][0], (gaze, measurement, self.bounds, 20.0))
        self.assertTrue(publisher.calls[0][1]['phone_present'])
        self.assertTrue(publisher.calls[0][1]['running'])
        publisher.result = (True, False)
        self.assertEqual(session.tick(), 'Gaze live · output busy, update skipped')
        publisher.result = (False, False)
        self.assertNotIn('Gaze live', session.tick())

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

    def test_create_saved_session_loads_saved_corrections_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            key = 'profile-test'
            path = folder / f'gaze-adaptation-personal-{key}.json'
            settings = folder / 'settings.json'
            settings.write_text('{"radius": 40}', encoding='utf-8')
            anchor = {'features': [0.] * 22, 'offset': [5., 0.],
                      'target': [10., 20.], 'count': 1, 'revision': 1}
            path.write_text(json.dumps({'version': 5, 'profile': key,
                                        'display': self.bounds.as_list(),
                                        'count': 1, 'click_accepted': 1,
                                        'click_anchors': [anchor]}), encoding='utf-8')
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            profile = SimpleNamespace(arrays={'scale': [1.] * 22, 'training_x': [], 'training_y': []})
            with patch('context_operator.settings.LOCAL', folder), \
                 patch('context_operator.providers.gaze.LOCAL', folder), \
                 patch('context_operator.personalization.ProfileStore') as store, \
                 patch('context_operator.providers.windows.Windows') as windows:
                store.return_value.load.return_value = profile, {'profile_key': key, 'report': {'p90_px': 120}}
                windows.return_value.bounds.return_value = self.bounds
                session = create_saved_session()
            self.assertEqual(session.radius, 40)
            self.assertEqual(session.learning.click_anchors, [anchor])
            self.assertEqual(session.learning.click_accepted, 1)
            self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
