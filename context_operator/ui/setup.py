"""One native, raw TrueDepth first setup. No command or input dispatch."""

import queue
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QShortcut, QKeySequence
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from ..foundations.gaze.arkit import ARKitWorker
from ..personalization import StarterSession, ProfileStore
from ..personalization.history import preserve_attempt
from ..personalization.session import MIN_DURATION, MIN_FRAMES
from ..providers.phone_connection import PhoneConnection
from ..settings import ASSETS, LOCAL
from .theme import (CARD, DANGER, MUTED, RAIL, RING, SUCCESS, TEXT, WAITING, keyboard_focus, label,
    paint_progress, set_primary, theme, ui_font)



class SetupCanvas(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.owner = window
        self.setObjectName('canvas')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(RAIL))
        fraction = self.owner.session.target_fraction if self.owner.running else None
        if fraction is None:
            return
        x, y = self.width()*fraction[0], self.height()*fraction[1]
        accent = SUCCESS if self.owner.session.ready_to_confirm else WAITING
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(RING), 2))
        painter.drawEllipse(QRectF(x-42, y-42, 84, 84))
        painter.setPen(QPen(QColor(accent), 3))
        painter.drawEllipse(QRectF(x-15, y-15, 30, 30))
        painter.setBrush(QColor(TEXT))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(x-3, y-3, 6, 6))
        frames = self.owner.session.frames
        duration = max(0., frames[-1].captured-frames[0].captured) if len(frames) >= 2 else 0.
        cap = 1. if self.owner.session.ready_to_confirm else .99
        progress = min(duration/MIN_DURATION, len(frames)/MIN_FRAMES, cap)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(accent), 3))
        painter.drawArc(QRectF(x-27, y-27, 54, 54), 90*16, -int(progress*360*16))
        session = self.owner.session
        total = len(session.check_targets if session.phase == 'check' else session.train_targets)
        message = f'Point {session.index+1} of {total} · ' + (
            (session.hold_reason or f'Hold steady · {progress:.0%} · Esc to close') if self.owner._ready() else
            'Waiting for iPhone · Esc to close')
        card_width = min(640, self.width()-40)
        card_x = (self.width()-card_width)/2
        card_y = self.height()-74
        if abs(x-self.width()/2) < card_width/2+45 and abs(y-card_y) < 90:
            card_y = 16
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(CARD))
        painter.drawRoundedRect(QRectF(card_x, card_y, card_width, 56), 10, 10)
        painter.setPen(QColor(TEXT))
        painter.setFont(ui_font('BODY', 16))
        painter.drawText(QRectF(card_x+12, card_y, card_width-24, 37), Qt.AlignCenter | Qt.TextWordWrap, message)
        paint_progress(painter, QRectF((self.width()-15*total+7)/2, card_y+40, 0, 0), total, session.index)


class SetupWindow(QMainWindow):
    adopted = Signal()

    def __init__(self, controller, parent=None, *, archive=None, store=None,
                 worker_factory=ARKitWorker, on_adopt=None):
        super().__init__(parent)
        self.c = controller
        self.bounds = controller.win.bounds()
        self.store = store or ProfileStore(LOCAL)
        self.archive = archive or LOCAL/'personal-starter-session.json'
        self.worker_factory = worker_factory
        self.on_adopt = on_adopt
        self.worker = None
        self.fit_worker = None
        self.fit_results = queue.SimpleQueue()
        self.running = False
        self.closed = False
        self.candidate = None
        self.error = ''
        self._profile_available = None
        self.last_status_refresh = 0.
        try:
            self.session = StarterSession.resume(self.bounds, self.archive)
        except (OSError, ValueError, TypeError) as exc:
            self.session = StarterSession(self.bounds, self.archive)
            self.error = 'Couldn’t resume the saved setup: ' + str(exc)
        self.candidate = self.session.candidate
        self.setWindowTitle('Mira · Set up iPhone')
        self.setWindowIcon(QIcon(str(ASSETS / 'icon.png')))
        self.setStyleSheet(theme())
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.canvas = SetupCanvas(self)
        self.setCentralWidget(self.canvas)
        outer = QVBoxLayout(self.canvas)
        outer.setContentsMargins(34, 24, 34, 24)
        top = QVBoxLayout()
        self.heading = label('Set up iPhone', 'title')
        self.count = QLabel('')
        self.count.setObjectName('muted')
        for item in (self.heading, self.count):
            top.addWidget(item)
        outer.addLayout(top)
        outer.addStretch(1)
        # One card holds what happened, what it means, and the buttons that act on it.
        self.guide = QWidget()
        self.guide.setObjectName('guide')
        self.guide.setMaximumWidth(720)
        guide_layout = QVBoxLayout(self.guide)
        guide_layout.setContentsMargins(32, 28, 32, 28)
        guide_layout.setSpacing(22)
        self.guide_heading = label('', 'call', True)
        self.guide_stats = QLabel()
        self.guide_stats.setObjectName('metric')
        self.guide_stats.setWordWrap(True)
        self.guide_body = label('', 'muted', True)
        self.status = QLabel('Connect iPhone and start tracking.')
        self.status.setObjectName('muted')
        self.status.setWordWrap(True)
        for text in (self.guide_heading, self.guide_stats, self.guide_body, self.status):
            guide_layout.addWidget(text)
        self.connect_button = QPushButton('Connect iPhone')
        self.connect_button.clicked.connect(self.connect_phone)
        self.begin_button = QPushButton('Start setup')
        self.begin_button.setObjectName('primary')
        self.begin_button.clicked.connect(self.begin)
        self.retry_button = QPushButton('Start over')
        self.retry_button.clicked.connect(self.restart)
        self.apply_button = QPushButton('Use gaze setup')
        self.apply_button.setObjectName('primary')
        self.apply_button.clicked.connect(self.adopt)
        self.cancel_button = QPushButton('Close · Esc')
        self.cancel_button.clicked.connect(self.close)
        actions = QHBoxLayout()
        for button in (self.apply_button, self.connect_button, self.begin_button, self.retry_button):
            actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        guide_layout.addLayout(actions)
        outer.addWidget(self.guide, 0, Qt.AlignHCenter)
        outer.addStretch(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(35)
        self.confirm_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.confirm_shortcut.setAutoRepeat(False)
        self.confirm_shortcut.activated.connect(self.confirm_target)
        keyboard_focus(self)
        self.refresh()

    def connect_phone(self):
        if not self.stop_worker():
            self.error = 'The previous connection is still closing. Try again in a second.'
            self.refresh()
            return
        try:
            connection = PhoneConnection.from_settings(self.c.settings)
            self.worker = self.worker_factory(connection=connection)
            self.worker.start()
            self.session.last_seq = -1
            self.session.frames.clear()
            self.session.ready_to_confirm = False
            self.error = ''
            self.connect_button.setText('Reconnect iPhone')
        except (OSError, RuntimeError, ValueError) as exc:
            self.error = str(exc)
            self.worker = None
        self.refresh()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop_event.set()
            self.worker.join(timeout=.7)
            if self.worker.is_alive():
                return False
            self.worker = None
        return True

    def _ready(self, sample=None):
        if self.worker is None:
            return False
        s = self.worker.latest() if sample is None else sample
        try:
            x = np.asarray(s.features, dtype=float)
            origins = np.asarray(s.eye_origins, dtype=float)
            return bool(not s.error and not s.blink and 0 <= time.monotonic()-s.captured < .20
                        and x.shape == (22,) and origins.shape == (2, 3)
                        and np.isfinite(x).all() and np.isfinite(origins).all())
        except (AttributeError, TypeError, ValueError):
            return False

    def begin(self):
        try:
            self._begin()
        except OSError as exc:
            self.error = 'Couldn’t save setup progress: ' + str(exc)
            self.running = False
            self.refresh()

    def _begin(self):
        if self.fit_worker and self.fit_worker.is_alive():
            return
        self.error = ''
        needs_live_sample = self.session.phase in ('train', 'check') or (self.session.phase == 'fit' and self.candidate is not None)
        if needs_live_sample and self._ready() and self.session.distance_hint(self.worker.latest()):
            self.refresh()
            return
        if self.session.phase == 'result':
            if self.candidate is None:
                self.session.phase = 'fit'
                self._fit()
            elif self.session.candidate_key is None:
                self._save_result()
        elif self.session.phase == 'fit':
            if self.candidate is None:
                self._fit()
            elif self._ready():
                preserve_attempt(self.archive, self.session.report)
                self.session.begin_check(self.candidate, time.monotonic())
                self.running = True
        elif self.session.phase in ('train', 'check') and self._ready():
            self.session.settle_until = time.monotonic() + .85
            self.running = True
        self.refresh()

    def _fit(self):
        self.running = False
        self.error = ''
        def fit():
            try:
                self.fit_results.put((self.session.fit(), ''))
            except Exception as exc:
                self.fit_results.put((None, str(exc)))
        self.fit_worker = threading.Thread(target=fit, daemon=True, name='personal-gaze-fit')
        self.fit_worker.start()
        self.refresh()

    def tick(self):
        if self.closed:
            return
        if not self.fit_results.empty():
            candidate, error = self.fit_results.get()
            if candidate is None:
                self.error = 'Couldn’t create the gaze profile: ' + error
            else:
                self.candidate = candidate
                self.error = ''
            self.refresh()
        if self.running:
            if not self._ready():
                self.session.frames.clear()
                self.session.ready_to_confirm = False
                sample = self.worker.latest() if self.worker else None
                if sample is None or sample.error or not self.worker.is_alive():
                    self.running = False
                    self.error = ((sample.error if sample else '') or 'iPhone stopped sending gaze data.') + ' Reconnect to continue from this point.'
                    self.refresh()
            else:
                self.session.feed(self.worker.latest(), time.monotonic())
        if time.monotonic()-self.last_status_refresh >= .25:
            self.refresh()
            self.last_status_refresh = time.monotonic()
        self.canvas.update()

    def confirm_target(self):
        if not self.running or not self._ready():
            return
        try:
            now = time.monotonic()
            self.session.feed(self.worker.latest(), now)
            if self.session.confirm(now):
                self._target_completed()
        except OSError as exc:
            self.running = False
            self.error = 'Couldn’t save the point: ' + str(exc)
            self.refresh()

    def _target_completed(self):
        if self.session.phase == 'fit':
            self._fit()
        elif self.session.phase == 'result':
            self.running = False
            self._save_result()
        self.refresh()

    def _save_result(self):
        try:
            self.session.report = self.session.evaluate()
            self.session.persist_result()
        except (OSError, ValueError, TypeError) as exc:
            self.error = 'Couldn’t save the check result: ' + str(exc)
            return
        try:
            preserve_attempt(self.archive, self.session.report)
        except (OSError, ValueError, TypeError) as exc:
            self.error = 'Result saved. Couldn’t add it to the setup history: ' + str(exc)

    def restart(self):
        if self.fit_worker and self.fit_worker.is_alive():
            return
        replacement = StarterSession(self.bounds, self.archive)
        try:
            preserve_attempt(self.archive, self.session.report)
            replacement._save()
        except OSError as exc:
            self.error = 'Couldn’t start a new setup: ' + str(exc)
            self.refresh()
            return
        self.running = False
        self.candidate = None
        self.session = replacement
        self.error = ''
        self._profile_available = None
        self.refresh()

    def _has_profile(self):
        """Whether a gaze setup was already in use before this attempt."""
        if self._profile_available is None:
            self._profile_available = bool(self.store.status(self.bounds).get('available'))
        return self._profile_available

    def _can_adopt(self):
        return (self.session.phase == 'result' and self.candidate is not None and
                self.session.candidate_key is not None and self.session.report is not None and
                self.session.report.get('usable') is True)

    def adopt(self):
        if not self._can_adopt():
            return
        try:
            self.store.adopt(self.candidate, self.session.report, self.bounds,
                             self.archive, activate=self.on_adopt)
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = 'Couldn’t apply the gaze profile: ' + str(exc)
            self.refresh()
            return
        self.close()
        self.adopted.emit()

    def refresh(self):
        phase = self.session.phase
        sample = self.worker.latest() if self.worker is not None else None
        ready = self._ready(sample)
        report = self.session.report if phase == 'result' else None
        usable = bool(report and report.get('quality') == 'starter' and report.get('usable') is True)
        connection_error = False
        if self.error:
            message = self.error
        elif report:
            message = (f"Offset {report['bias_px'][0]:+.0f}/{report['bias_px'][1]:+.0f} px · "
                       f"jitter {report['jitter_px']:.0f} px · coverage {report['coverage']:.0%}")
        elif phase == 'fit':
            message = ('Creating your gaze profile…' if self.fit_worker and self.fit_worker.is_alive() else
                       'Profile ready to check on five new points.' if self.candidate else
                       'Setup points saved. Mira will rebuild the profile and start the check again.')
        elif self.running:
            if ready:
                message = self.session.hold_reason or 'Keep looking at the center of the point until the ring fills.'
            else:
                message = (sample.error or sample.reason if sample else '') or 'Waiting for iPhone'
        elif ready:
            message = 'Mira sees your eyes. Start when you’re comfortable.'
        elif sample is not None and sample.error:
            connection_error = True
            message = sample.error
            if self.c.settings.get('phone_transport') == 'wifi':
                message += ' To enter the address manually, close setup and open Connection settings.'
        elif sample is not None:
            message = sample.reason or 'Connecting to iPhone…'
        else:
            message = 'Open Mira on your iPhone and tap Start Tracking, then connect it here.'
        if not self.running and ready and phase in ('train', 'check', 'fit'):
            message = self.session.distance_hint(sample) or message
        self.status.setText(message)
        self.status.setStyleSheet('color:' + (DANGER if self.error or connection_error else MUTED))
        self.guide.setVisible(not self.running)
        self.guide_stats.setVisible(bool(report))
        if report:
            self.guide_heading.setText('Gaze setup ready' if usable else 'Gaze setup needs another try')
            self.guide_stats.setText(f"Typical error {report['median_px']:.0f} px\nHard points {report['p90_px']:.0f} px")
            kept = 'Mira kept your current gaze setup. ' if self._has_profile() else ''
            self.guide_body.setText('Mira checked five new points with your head in one comfortable position. '
                                   'It works best for broad areas such as panels and large buttons, '
                                   'and clicking near where you look refines it as you use Mira.'
                                   if usable else '\n'.join(report.get('problems', [])) +
                                   '\n\n' + kept + 'Measurements saved for review.')
        elif phase == 'fit':
            self.guide_heading.setText('Check your gaze' if self.candidate else 'Build your gaze setup')
            self.guide_body.setText('Look at five new points the same way you did in setup. '
                                   'Mira uses them to check your gaze setup.')
        else:
            self.guide_heading.setText('Look, then press Space')
            self.guide_body.setText('Place your iPhone where the TrueDepth camera can see your face. Keep it still. '
                                   'Look at the center of each point. When the ring fills, press Space while still looking at the center. '
                                   'The point stays until you confirm.\n\n'
                                   'Keep your head in a comfortable position. After 9 setup points, '
                                   'Mira checks your gaze on 5 new points. '
                                   'Control is paused during setup.')
        self.heading.setText('Check your gaze' if phase == 'check' else
                             'Setup result' if phase == 'result' else
                             'Set up iPhone')
        if self.running and phase in ('train', 'check'):
            count = len(self.session.check_targets if phase == 'check' else self.session.train_targets)
            point_label = 'check points' if phase == 'check' else 'setup points'
            self.count.setText(f'Point {self.session.index+1}/{count} · {point_label}')
        else:
            self.count.setText('Mira stores measurements on this PC and doesn’t save video.')
        refit = phase == 'result' and self.candidate is None and bool(report and report.get('usable'))
        retry_save = phase == 'result' and self.candidate is not None and self.session.candidate_key is None
        self.begin_button.setText('Rebuild gaze setup' if refit else
                                  'Save result' if retry_save else
                                  'Check on new points' if phase == 'fit' and self.candidate else
                                  'Rebuild gaze setup' if phase == 'fit' else
                                  'Start setup' if phase == 'train' else 'Continue check')
        self.begin_button.setVisible(not self.running and (phase != 'result' or refit or retry_save))
        self.begin_button.setEnabled(((ready and not self.session.distance_hint(sample))
                                      or refit or retry_save or phase == 'fit' and self.candidate is None)
                                     and not (self.fit_worker and self.fit_worker.is_alive()))
        failed = bool(report) and not usable
        self.retry_button.setText('Try setup again' if failed else 'Start over')
        set_primary(self.retry_button, failed)
        self.retry_button.setVisible(not self.running and (phase == 'result' or self.error != '' or phase == 'fit'))
        self.apply_button.setVisible(self._can_adopt())
        self.connect_button.setVisible(not self.running)
        set_primary(self.connect_button, self.worker is None and phase != 'result')
        self.cancel_button.setVisible(not self.running)
        show_hud = not self.running or not ready
        for item in (self.heading, self.count):
            item.setVisible(show_hud)
        self._size_guide()
        self.canvas.update()

    def _size_guide(self):
        width = max(300, min(720, self.width()-68))
        self.guide.setFixedWidth(width)
        labels = [label for label in (self.guide_heading, self.guide_stats, self.guide_body, self.status)
                  if not label.isHidden()]
        for label in labels:
            label.setMinimumHeight(0)
            label.setMaximumHeight(16777215)
            label.setFixedHeight(max(label.fontMetrics().height(), label.heightForWidth(width-64)))
        self.guide.setFixedHeight(56 + sum(label.height() for label in labels) + 22*len(labels)
                                  + self.cancel_button.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._size_guide()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed = True
        self.running = False
        self.timer.stop()
        self.stop_worker()
        if getattr(getattr(self.c, 'setup_close', None), '__self__', None) is self:
            self.c.setup_close = None
        super().closeEvent(event)


def setup_geometry(controller, targets):
    screens = QApplication.screens()
    if not screens:
        raise RuntimeError('No screen found for setup')
    area = screens[0].geometry()
    for screen in screens[1:]:
        area = area.united(screen.geometry())
    scale = float(screens[0].devicePixelRatio())
    if any(abs(float(screen.devicePixelRatio())-scale) > .01 for screen in screens):
        raise RuntimeError('Use the same Windows scaling on each screen to set up gaze')
    physical = np.asarray([area.x()*scale, area.y()*scale,
                           area.width()*scale, area.height()*scale], dtype=float)
    bounds = np.asarray(controller.win.bounds().as_list(), dtype=float)
    if not np.allclose(physical, bounds, rtol=0, atol=3):
        raise RuntimeError('The screen layout changed. Check Windows scaling and run setup again')
    for fraction in targets:
        target = bounds[:2] + bounds[2:]*fraction
        if not any(screen.geometry().x()*scale <= target[0] <
                   (screen.geometry().x()+screen.geometry().width())*scale and
                   screen.geometry().y()*scale <= target[1] <
                   (screen.geometry().y()+screen.geometry().height())*scale
                   for screen in screens):
            raise RuntimeError('A setup point falls between screens. Adjust the screen arrangement in Windows')
    return area, len(screens) == 1


def open_setup(controller, parent=None):
    """Pause every action source before raw collection; caller owns the window."""
    if controller.busy or controller.output_pending:
        raise RuntimeError('Wait for the current action to finish')
    area, single_screen = setup_geometry(controller, StarterSession.train_targets + StarterSession.check_targets)
    controller.pause()
    window = SetupWindow(controller, parent,
                         on_adopt=controller.activate_profile)
    controller.setup_close = window.close
    if single_screen:
        window.showFullScreen()
    else:
        window.setGeometry(area)
        window.show()
    return window
