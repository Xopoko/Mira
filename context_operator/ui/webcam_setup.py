"""Independent webcam setup. The controller stays paused throughout collection."""
import time

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QIcon, QShortcut, QKeySequence
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy

from ..core.contracts import GAZE_TTL
from ..personalization.webcam import WebcamSession, WebcamProfileStore
from ..personalization.webcam_attempts import WebcamAttemptStore
from ..providers.webcam import WebcamGaze
from ..settings import ASSETS, LOCAL
from .setup import setup_geometry
from .theme import (DANGER, GROUND, MUTED, RAIL, RING, SUCCESS, TEXT, WAITING, dark_title_bar, keyboard_focus,
    label, paint_progress, theme, ui_font)


REACTION_SECONDS = .50


class CameraPreview(QWidget):
    """Ephemeral camera preview; this widget never captures or writes images."""
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(RAIL))
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        source = self.owner.provider
        frame = getattr(source, 'preview_image', None)
        captured = getattr(source, 'preview_captured', 0.)
        if frame is not None and not frame.isNull() and 0 <= time.monotonic()-captured < .5:
            image = frame.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawImage((self.width()-image.width())//2, (self.height()-image.height())//2, image)
        else:
            painter.setPen(QColor(MUTED))
            painter.setFont(ui_font('BODY', 15))
            message = 'Your camera preview appears here' if source is None else 'Waiting for camera…'
            painter.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignCenter | Qt.TextWordWrap, message)


class WebcamCanvas(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setObjectName('canvas')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(RAIL if self.owner.collecting else GROUND))
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.owner.collecting:
            return
        target = self.owner.session.current_target_fraction
        if target is None:
            return
        x, y = self.width()*target[0], self.height()*target[1]
        paused = self.owner.paused
        progress = 0. if paused else min(.97, getattr(self.owner.session, 'progress', 0.))
        painter.setPen(QPen(QColor(RING), 4))
        painter.setBrush(Qt.NoBrush)
        ring = QRectF(x-23, y-23, 46, 46)
        painter.drawEllipse(ring)
        painter.setPen(QPen(QColor(WAITING), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(ring, 90*16, -round(360*16*progress))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(MUTED if paused else TEXT))
        painter.drawEllipse(QRectF(x-4, y-4, 8, 8))
        message = self.owner.target_message()
        # Keep the feedback close to the eyes' target, on screen even at corners.
        width = min(340, self.width()-32)
        left = max(16, min(x-width/2, self.width()-width-16))
        top = y+38 if y < self.height()*.75 else y-92
        rect = QRectF(left, top, width, 54)
        painter.setPen(QColor(MUTED))
        painter.setFont(ui_font('BODY', 13))
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, message)
        session = self.owner.session
        total = len(session.train_targets if session.phase == 'train' else session.check_targets)
        paint_progress(painter, QRectF(20, self.height()-24, 0, 0), total, session.index)


class WebcamSetupWindow(QMainWindow):
    def __init__(self, controller, parent=None, *, provider_factory=WebcamGaze, folder=LOCAL):
        super().__init__(parent)
        self.c = controller
        self.bounds = controller.win.bounds()
        self.device_id = controller.settings['webcam_device_id']
        self.session = WebcamSession(self.bounds, self.device_id)
        self.store = WebcamProfileStore(folder)
        self.attempt_store = WebcamAttemptStore(folder)
        self.attempt_path = None
        self.attempt_error = ''
        self.provider_factory = provider_factory
        self.provider = None
        self.collecting = False
        self.paused = False
        self.collect_after = 0.
        self.screen_area = None
        self.single_screen = True
        self.closed = False
        self.applied = False
        self.error = ''
        self.setWindowTitle('Mira · Set up webcam')
        self.setWindowIcon(QIcon(str(ASSETS / 'icon.png')))
        self.setStyleSheet(theme())
        self.setWindowFlag(Qt.Window, True)
        self.resize(1020, 690)
        self.setMinimumSize(820, 590)
        self.canvas = WebcamCanvas(self)
        self.setCentralWidget(self.canvas)
        layout = QVBoxLayout(self.canvas)
        layout.setContentsMargins(32, 26, 32, 26)
        layout.setSpacing(14)
        self.heading = label('Set up webcam', 'title')
        self.status = QLabel('')
        self.status.setObjectName('muted')
        self.status.setWordWrap(True)
        layout.addWidget(self.heading)
        self.subtitle = QLabel('Set up a second angle for when your iPhone loses track of your gaze.')
        self.subtitle.setObjectName('muted')
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)
        self.body = QWidget()
        body = QHBoxLayout(self.body)
        body.setContentsMargins(0, 14, 0, 12)
        body.setSpacing(32)
        self.preview_panel = QWidget()
        preview = QVBoxLayout(self.preview_panel)
        preview.setContentsMargins(0, 0, 0, 0)
        self.preview = CameraPreview(self)
        preview.addWidget(self.preview, 1)
        self.camera_label = QLabel(controller.settings.get('webcam_device_name') or 'Selected webcam')
        self.camera_label.setObjectName('muted')
        self.camera_label.setWordWrap(True)
        preview.addWidget(self.camera_label)
        self.diagnostics = QLabel('Mira doesn’t save video.')
        self.diagnostics.setObjectName('muted')
        self.diagnostics.setWordWrap(True)
        preview.addWidget(self.diagnostics)
        body.addWidget(self.preview_panel, 6)
        self.instructions = QWidget()
        self.instructions.setObjectName('guide')
        instructions = QVBoxLayout(self.instructions)
        instructions.setContentsMargins(20, 16, 20, 16)
        instructions.setSpacing(12)
        instructions.addStretch()
        self.section = label('Sit as you normally do', 'heading', True)
        instructions.addWidget(self.section)
        instructions.addWidget(self.status)
        self.guide = QLabel('Keep the webcam on your monitor and look at the screen.\n\n'
                            'Look at each point. When the ring fills, the next point appears.\n\n'
                            '9 points for setup and 5 for the check. Press Space to pause.')
        self.guide.setObjectName('muted')
        self.guide.setWordWrap(True)
        instructions.addWidget(self.guide)
        self.record_status = QLabel('')
        self.record_status.setObjectName('muted')
        self.record_status.setWordWrap(True)
        instructions.addWidget(self.record_status)
        instructions.addStretch()
        body.addWidget(self.instructions, 5)
        layout.addWidget(self.body, 1)
        self.collection_bar = QWidget()
        bar = QHBoxLayout(self.collection_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        self.collection_label = QLabel('')
        self.collection_label.setObjectName('muted')
        bar.addWidget(self.collection_label, 1)
        self.pause_button = QPushButton('Pause · Space')
        self.pause_button.clicked.connect(self.toggle_pause)
        bar.addWidget(self.pause_button)
        self.camera_button = QPushButton('Check camera')
        self.camera_button.clicked.connect(self.prepare_again)
        bar.addWidget(self.camera_button)
        self.cancel_button = QPushButton('Close · Esc')
        self.cancel_button.clicked.connect(self.close)
        bar.addWidget(self.cancel_button)
        layout.addWidget(self.collection_bar)
        self.collection_space = QWidget()
        layout.addWidget(self.collection_space, 1)
        self.buttons = QWidget()
        row = QHBoxLayout(self.buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.privacy = QLabel('Control is paused during setup.')
        self.privacy.setObjectName('muted')
        self.privacy.setWordWrap(True)
        row.addWidget(self.privacy, 1)
        self.connect_button = QPushButton('Connect webcam')
        self.connect_button.setObjectName('primary')
        self.connect_button.clicked.connect(self.connect_camera)
        self.begin_button = QPushButton('Start setup')
        self.begin_button.setObjectName('primary')
        self.begin_button.clicked.connect(self.begin)
        self.retry_button = QPushButton('Try setup again')
        self.retry_button.clicked.connect(self.restart)
        self.apply_button = QPushButton('Use webcam profile')
        self.apply_button.setObjectName('primary')
        self.apply_button.clicked.connect(self.adopt)
        self.close_button = QPushButton('Close · Esc')
        self.close_button.clicked.connect(self.close)
        for button in (self.connect_button, self.begin_button, self.retry_button, self.apply_button, self.close_button):
            row.addWidget(button)
        for button in (self.pause_button, self.camera_button, self.cancel_button):
            button.setStyleSheet('padding: 6px 12px; font-size: 13px;')
        layout.addWidget(self.buttons)
        self.space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space.setAutoRepeat(False)
        self.space.activated.connect(self.toggle_pause)
        self.timer = QTimer(self)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        dark_title_bar(self)
        keyboard_focus(self)
        self.refresh()

    def connect_camera(self):
        self.stop_provider()
        self.error = ''
        try:
            self.provider = self.provider_factory(self.bounds, self.device_id, calibrated=False)
            self.provider.start()
            self.session.reset_stream()
        except Exception as exc:
            self.stop_provider()
            self.error = str(exc)
        self.refresh()

    def stop_provider(self):
        provider, self.provider = self.provider, None
        if provider is not None:
            provider.stop()

    def begin(self):
        if self.provider is None or not self._ready():
            return
        self.collecting = True
        self.paused = False
        self.session.reset_stream()
        self.collect_after = time.monotonic()+REACTION_SECONDS
        self.error = ''
        self.present_collection(True)
        self.refresh()

    def present_collection(self, enabled):
        # Direct construction supports offscreen tests without changing screens.
        if self.screen_area is None:
            return
        if enabled:
            self._preparation_geometry = self.geometry()
            self.setWindowFlag(Qt.FramelessWindowHint, True)
            if self.single_screen:
                self.showFullScreen()
            else:
                self.setGeometry(self.screen_area)
                self.show()
        else:
            self.setWindowFlag(Qt.FramelessWindowHint, False)
            self.showNormal()
            self.setGeometry(self._preparation_geometry)

    def toggle_pause(self):
        if not self.collecting:
            return
        self.paused = not self.paused
        self.session.reset_stream()
        self.collect_after = time.monotonic()+REACTION_SECONDS
        self.refresh()

    def prepare_again(self):
        if self.collecting:
            self.collecting = self.paused = False
            self.session.reset_stream()
            self.present_collection(False)
        self.refresh()

    def target_message(self):
        if self.paused:
            return 'Paused\nPress Space to continue from this point'
        if time.monotonic() < self.collect_after:
            return 'Look here'
        if self.provider is not None:
            obs = self.provider.latest()
            if obs.reason:
                return obs.reason
            if not 0 <= time.monotonic()-obs.captured < GAZE_TTL:
                return 'Waiting for a fresh camera frame\nCheck the camera'
        reason = self.session.hold_reason or 'Keep looking at the point'
        count = getattr(self.session, 'sample_count', len(self.session.frames))
        if time.monotonic()-self.collect_after > 3 and count < 10:
            return f'{reason}\nFrames: {count}/10 · check the camera if this persists'
        return reason

    def _ready(self):
        if self.provider is None:
            return False
        obs = self.provider.latest()
        return bool(obs.features and not obs.reason and 0 <= time.monotonic()-obs.captured < GAZE_TTL)

    def tick(self):
        if self.closed:
            return
        if self.c.win.emergency():
            self.c.pause()
            self.close()
            return
        if self.c.win.bounds() != self.bounds:
            self.error = 'The screen layout changed. Open setup again.'
            self.prepare_again()
            self.stop_provider()
        if self.provider is not None:
            if self.provider.error:
                self.error = self.provider.error
                self.prepare_again()
                self.stop_provider()
            elif self.collecting and not self.paused:
                now = time.monotonic()
                obs = self.provider.latest()
                # A just-shown target must not inherit frames of the previous one.
                if now >= self.collect_after and obs.captured >= self.collect_after:
                    self.session.feed(obs, now)
                    if self.session.ready:
                        self.confirm()
                elif now >= self.collect_after:
                    self.session.reset_stream()
        self.refresh()

    def confirm(self):
        if not self.collecting or self.paused or time.monotonic() < self.collect_after:
            return
        try:
            # Acceptance uses the current observation, never a stale ready flag.
            if self.provider is not None:
                self.session.feed(self.provider.latest(), time.monotonic())
            if self.session.confirm(time.monotonic()):
                self.collect_after = time.monotonic()+REACTION_SECONDS
                if self.session.phase == 'result':
                    self.prepare_again()
                    self.stop_provider()
                    try:
                        self.attempt_path = self.attempt_store.save(self.session)
                    except (OSError, ValueError, TypeError) as exc:
                        self.attempt_error = 'Couldn’t save measurements: '+str(exc)
        except (ValueError, OSError) as exc:
            self.error = str(exc)
            self.prepare_again()
            self.stop_provider()
        self.refresh()

    def restart(self):
        self.stop_provider()
        self.session = WebcamSession(self.bounds, self.device_id)
        self.collecting = self.applied = False
        self.attempt_path = None
        self.attempt_error = ''
        self.error = ''
        self.refresh()

    def adopt(self):
        try:
            if self.c.win.bounds() != self.bounds:
                raise ValueError('The screen layout changed')
            self.store.adopt(self.session.candidate, self.session.report)
            self.applied = True
            self.c.webcam_status = 'Webcam setup saved · connect the webcam'
            self.c.changed.emit()
        except (OSError, ValueError) as exc:
            self.error = 'Couldn’t save the webcam profile: '+str(exc)
        self.refresh()

    def refresh(self):
        report = self.session.report
        self.canvas.layout().setContentsMargins(*( (20, 12, 20, 12) if self.collecting else (32, 26, 32, 26)))
        for widget in (self.heading, self.subtitle, self.body):
            widget.setVisible(not self.collecting)
        self.buttons.setVisible(not self.collecting)
        self.collection_bar.setVisible(self.collecting)
        self.collection_space.setVisible(self.collecting)
        phase = 'Setup' if self.session.phase == 'train' else 'Check'
        total = 9 if self.session.phase == 'train' else 5
        self.collection_label.setText(f'{phase} · {min(self.session.index+1, total)} of {total} · look at the point')
        self.pause_button.setText('Continue · Space' if self.paused else 'Pause · Space')
        self.connect_button.setVisible(report is None and not self.applied and self.provider is None)
        self.begin_button.setVisible(report is None and not self.applied and self.provider is not None)
        self.begin_button.setEnabled(self._ready())
        self.begin_button.setText('Continue setup' if self.session.train or self.session.check else 'Start setup')
        self.retry_button.setVisible(report is not None or bool(self.error))
        self.apply_button.setVisible(bool(report and report.usable and not self.applied))
        if report is not None:
            self.section.setText('Webcam profile ready' if report.usable else 'Webcam profile needs work')
            self.guide.setText(('Mira checked five new points. You can save this profile.' if report.usable
                               else self.result_reason(report)+' Keep the webcam in place.')
                              + f'\n\nTypical error: {report.median_px:.0f} px.\n90% of measurements were within {report.p90_px:.0f} px.'
                              + '\n\niPhone stays your primary gaze source.')
            self.preview_panel.setVisible(False)
        else:
            self.section.setText('Sit as you normally do')
            self.guide.setText('Keep the webcam on your monitor and look at the screen.\n\n'
                               'Look at each point. When the ring fills, the next point appears.\n\n'
                               '9 points for setup and 5 for the check. Press Space to pause.')
            self.preview_panel.setVisible(True)
        self.record_status.setVisible(report is not None)
        self.record_status.setText(self.attempt_error or ('Mira saved the measurements for review.'
                                   if self.attempt_path and report and not report.usable else ''))
        if self.applied:
            self.status.setText('Saved. Close setup and connect iPhone and the webcam.')
        elif self.error:
            self.status.setText(self.error)
        elif self.provider is not None:
            obs = self.provider.latest()
            self.status.setText('Eyes visible · ready to start' if self._ready() else obs.reason or 'Waiting for a fresh camera frame')
        else:
            self.status.setText('Camera off' if report is None else
                                'Ready to save' if report.usable else 'This profile is inactive')
        self.status.setStyleSheet('color: '+ (DANGER if self.error else SUCCESS if self._ready() or self.applied else MUTED))
        diagnostics = getattr(self.provider, 'diagnostics', None)
        if callable(diagnostics):
            values = diagnostics()
            fps = values.get('processed_fps', 0.)
            self.diagnostics.setText(f'Tracking: {fps:.1f} fps · video stays in memory' if fps
                                     else 'Mira doesn’t save video.')
        else:
            self.diagnostics.setText('Mira doesn’t save video.')
        self.preview.update()
        self.canvas.update()

    @staticmethod
    def result_reason(report):
        failures = []
        uncovered = sum(value < report.limits['coverage'] for value in report.coverage)
        inaccurate = sum(row['p90_px'] > report.limits['p90_px'] for row in report.targets)
        unstable = sum(row['jitter_px'] > report.limits['jitter_px'] for row in report.targets)
        if uncovered:
            failures.append(f'Too few readings were on screen and within the setup range at {uncovered} of 5 points.')
        if inaccurate:
            failures.append(f'Gaze was too far from {inaccurate} of 5 points.')
        if unstable:
            failures.append(f'Gaze readings moved too much at {unstable} of 5 points.')
        if not failures:
            failures.append('The webcam profile needs to be more accurate before Mira can use it.')
        return ' '.join(failures)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed = True
        self.collecting = False
        self.timer.stop()
        self.stop_provider()
        if getattr(getattr(self.c, 'setup_close', None), '__self__', None) is self:
            self.c.setup_close = None
        super().closeEvent(event)


def open_webcam_setup(controller, parent=None):
    if controller.busy or controller.output_pending:
        raise RuntimeError('Wait for the current action to finish')
    if not controller.settings.get('webcam_device_id'):
        raise ValueError('Choose a webcam in Settings')
    area, single_screen = setup_geometry(controller, WebcamSession.train_targets + WebcamSession.check_targets)
    controller.pause()
    window = WebcamSetupWindow(controller, parent)
    controller.setup_close = window.close
    window.screen_area = area
    window.single_screen = single_screen
    window.show()
    return window
