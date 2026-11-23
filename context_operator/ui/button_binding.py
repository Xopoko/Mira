"""Enroll a distinct physical keyboard device without intercepting global keys."""
import time
from PySide6.QtCore import Qt,Signal,QTimer,QEvent
from PySide6.QtWidgets import QDialog,QPushButton,QVBoxLayout
from .window import label, theme
from ..providers.button import RawKeyboardEvents,DeviceKeyEvent,valid_button_vk,valid_device_id


def key_name(vk):
    # Save a readable label alongside the physical device binding.
    if vk==13:return 'Enter'
    if vk==32:return 'Space'
    if 112<=vk<=135:return f'F{vk-111}'
    if 48<=vk<=90:return chr(vk)
    return 'Button'


class ButtonBindingDialog(QDialog):
    received_key=Signal(object)

    def __init__(self,parent=None,*,observer=None):
        super().__init__(parent)
        self.setWindowTitle('Mira · Bind USB button');self.setMinimumWidth(540)
        self.setStyleSheet(theme())
        self.setModal(True);self.binding=None;self.pending=None;self.keyboard_id=None
        self.observer=observer if observer is not None else RawKeyboardEvents();self._listening=False
        self.received_key.connect(self._on_raw)
        self.health_timer=QTimer(self);self.health_timer.setInterval(250);self.health_timer.timeout.connect(self._check_reader)
        layout=QVBoxLayout(self);layout.setContentsMargins(26,24,26,24);layout.setSpacing(12)
        self.title=label('Press Enter on your regular keyboard','title',True)
        layout.addWidget(self.title)
        self.hint=label('Press and release Enter so Mira can tell your keyboard and USB button apart. Keep this window active.','muted',True)
        layout.addWidget(self.hint)
        note=label('If your button can send F13–F24, use one of those keys. If it sends Enter, Windows will still handle Enter as usual.','muted',True)
        layout.addWidget(note)
        cancel=QPushButton('Cancel · Esc');cancel.setAutoDefault(False)
        cancel.installEventFilter(self)
        cancel.clicked.connect(self.reject);layout.addWidget(cancel,0,Qt.AlignRight)
        self.resize(560,230)

    def showEvent(self,event):
        super().showEvent(event)
        if self._listening:return
        self._listening=True
        try:self.observer.start(self.received_key.emit);self.health_timer.start()
        except (OSError,RuntimeError) as exc:
            self._listening=False;self.hint.setText('Couldn’t read input devices: '+str(exc))

    def done(self,result):
        self._listening=False;self.health_timer.stop()
        try:self.observer.stop()
        finally:super().done(result)

    def _check_reader(self):
        error=getattr(self.observer,'error','')
        if isinstance(error,str) and error:
            self._listening=False;self.pending=None;self.health_timer.stop()
            self.hint.setText('The device reader stopped. Close this window and bind the button again.')

    def changeEvent(self,event):
        if event.type()==QEvent.ActivationChange and not self.isActiveWindow():self.pending=None
        super().changeEvent(event)

    def _on_raw(self,event):
        if not self._listening or not self.isActiveWindow() or not isinstance(event,DeviceKeyEvent):return
        if event.removed:
            if self.pending and event.device_id==self.pending.device_id:self.pending=None
            return
        if not valid_device_id(event.device_id) or not 0<=time.monotonic()-event.captured<.5:return
        if not valid_button_vk(event.vk):return
        if event.down:
            if self.pending:return
            if self.keyboard_id is None:
                if event.vk!=13:self.hint.setText('Press Enter on your regular keyboard to continue.');return
            elif event.device_id==self.keyboard_id:
                self.hint.setText('Mira recognized your regular keyboard. Press the separate USB button.');return
            self.pending=event;return
        pending=self.pending
        if pending is None or (event.device_id,event.vk,event.scan_code,event.extended)!=(pending.device_id,pending.vk,pending.scan_code,pending.extended):return
        self.pending=None
        if self.keyboard_id is None:
            self.keyboard_id=event.device_id
            self.title.setText('Press your separate USB button')
            self.hint.setText('Press and release your USB button to bind it. Mira won’t click during setup.')
        else:
            self.binding=(event.vk,key_name(event.vk),event.device_id,event.scan_code,event.extended)
            self.accept()

    def eventFilter(self,watched,event):
        # A bindable key still belongs to the Raw Input reader when Cancel has focus.
        if event.type() in (QEvent.KeyPress,QEvent.KeyRelease) and event.key() in (Qt.Key_Return,Qt.Key_Enter,Qt.Key_Space):
            return True
        return super().eventFilter(watched,event)

    def keyPressEvent(self,event):
        # Consume only this enrollment dialog's normal Qt key event. All other
        # applications and keyboards retain their ordinary Windows input.
        event.accept()
        if event.key()==Qt.Key_Escape:self.reject()

    def keyReleaseEvent(self,event):event.accept()
