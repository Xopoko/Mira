"""Small no-screen-capture runtime for explicit local game gaze sessions."""
from __future__ import annotations

import time

from .core.contracts import Rect
from .providers.game_gaze import GameGazePublisher


class GameGazeSession:
    """Own one phone reader and publisher; dependencies are injectable for tests."""

    def __init__(self, phone, learning, desktop: Rect, *, desktop_bounds, radius=230,
                 publisher_factory=GameGazePublisher, clock=time.monotonic):
        self.phone = phone
        self.learning = learning
        self.desktop = desktop
        self.desktop_bounds = desktop_bounds
        self.radius = radius
        self.publisher_factory = publisher_factory
        self.clock = clock
        self.publisher = None
        self.running = False
        self.status = 'Off · connect gaze to begin'

    def start(self):
        if self.running:
            return True
        try:
            self.publisher = self.publisher_factory()
            self.phone.start()
            self.running = True
            self.status = 'Connecting to iPhone · waiting for fresh gaze data'
            return True
        except Exception as exc:
            cleanup_error = self._close_resources()
            detail = f' · error while stopping: {type(cleanup_error).__name__}: {cleanup_error}' if cleanup_error else ''
            self.status = f'Couldn’t connect: {type(exc).__name__}: {exc}{detail}'
            return False

    def tick(self):
        if not self.running:
            return self.status
        now = self.clock()
        try:
            if self.desktop_bounds() != self.desktop:
                return self.stop(reason='The screen changed. Connect gaze again for this screen.')
            raw_gaze, reason, _fps = self.phone.poll(self.radius)
            measurement = self.phone.measurement
            gaze = self.learning.correct(raw_gaze, measurement, now, self.radius)
            active, written = self.publisher.publish(
                gaze, measurement, self.desktop, now,
                phone_present=True,
                pointer_mode=False, mouse_control=False,
                setup=False, running=True,
            )
            if active and written:
                self.status = 'Gaze live · receiving fresh data'
            elif active:
                self.status = 'Gaze live · output busy, update skipped'
            else:
                self.status = self._waiting_status(reason, measurement)
            return self.status
        except Exception as exc:
            cleanup_error = self._close_resources()
            detail = f' · error while stopping: {type(cleanup_error).__name__}: {cleanup_error}' if cleanup_error else ''
            self.status = f'Couldn’t read gaze: {type(exc).__name__}: {exc}{detail}'
            return self.status

    def _waiting_status(self, reason, measurement):
        worker = self.phone.worker
        if worker is not None and not worker.is_alive():
            detail = reason or 'iPhone stopped sending data'
            return f'No iPhone connection · {detail}'
        if measurement is None and worker is not None and worker.frames == 0:
            return 'Connecting · waiting for gaze data'
        detail = reason or 'no recent usable frame'
        return f'iPhone connected · tracking unavailable: {detail}'

    def _close_resources(self):
        self.running = False
        publisher, self.publisher = self.publisher, None
        first_error = None
        try:
            if publisher is not None:
                publisher.close()  # close revokes the mapping before the phone worker stops
        except Exception as exc:
            first_error = exc
        finally:
            try:
                self.phone.stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def stop(self, *, reason='Off · connect gaze to begin'):
        error = self._close_resources()
        if error is None:
            self.status = reason
        else:
            self.status = f'Couldn’t stop: {type(error).__name__}: {error}'
        return self.status


def create_saved_session():
    """Load the current saved source and corrections without writing local state."""
    from .providers.gaze import PhoneGaze, learning_path
    from .providers.learning import GazeLearning
    from .providers.phone_connection import PhoneConnection
    from .providers.windows import Windows
    from .settings import LOCAL, load

    settings = load()
    desktop_bounds = Windows().bounds
    desktop = desktop_bounds()
    phone = PhoneGaze(desktop, connection=PhoneConnection.from_settings(settings))
    learning = GazeLearning(learning_path(phone.profile_key, LOCAL))
    learning.bind(phone.profile_key, desktop, phone.feature_scale)
    learning.trainer.set_seed(phone.profile)
    return GameGazeSession(phone, learning, desktop, desktop_bounds=desktop_bounds,
                           radius=settings.get('radius', 230))
