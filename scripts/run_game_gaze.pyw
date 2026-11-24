"""Explicit, lightweight Qt launcher for the numeric game gaze feed."""
import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from context_operator.game_gaze_session import create_saved_session

WM_HOTKEY = 0x0312
HOTKEY_ID = 0x434F
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
VK_F12 = 0x7B


class GameGazeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mira · Gaze runner')
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setFixedWidth(350)
        self.session = None
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int,
                                               wintypes.UINT, wintypes.UINT)
        self.user32.RegisterHotKey.restype = wintypes.BOOL
        self.user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
        self.user32.UnregisterHotKey.restype = wintypes.BOOL
        self.hwnd = wintypes.HWND(int(self.winId()))
        self.hotkey_registered = bool(self.user32.RegisterHotKey(
            self.hwnd, HOTKEY_ID, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F12))

        self.status = QLabel('Off · gaze never starts on its own')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(48)
        self.button = QPushButton('Connect gaze')
        self.button.clicked.connect(self.toggle)
        self.hotkey_status = QLabel('Ctrl + Alt + F12 stops gaze' if self.hotkey_registered
                                    else 'The global shortcut is taken. Stop with the button or close this window.')
        self.hotkey_status.setWordWrap(True)
        self.hotkey_status.setStyleSheet('color: #A6A6A0; font-size: 12px;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(9)
        layout.addWidget(self.status)
        layout.addWidget(self.button)
        layout.addWidget(self.hotkey_status)
        self.setStyleSheet('''
            QWidget { background: #141414; color: #F2F2EE; font: 13px "Segoe UI Variable Text"; }
            QLabel { background: transparent; }
            QPushButton { background: #FFFFFF; color: #141414; border: none;
                          border-radius: 8px; padding: 10px 18px; font-weight: 600; }
            QPushButton:hover { background: #EDEDE8; }
            QPushButton:pressed { background: #D9D9D3; }
            QPushButton:focus { outline: none; border: 2px solid #3D6BFF; }
        ''')
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.tick)

    def _set_status(self, text):
        if self.status.text() != text:
            self.status.setText(text)

    def toggle(self):
        if self.session and self.session.running:
            self.stop_tracking()
            return
        try:
            session = create_saved_session()
            if not session.start():
                self._set_status(session.status)
                self.session = None
                return
        except Exception as exc:
            self._set_status(f'Couldn’t prepare gaze: {type(exc).__name__}: {exc}')
            self.session = None
            return
        self.session = session
        self.button.setText('Disconnect gaze')
        self.timer.start()
        self._set_status(session.status)

    def stop_tracking(self):
        if self.session and self.session.running:
            self.timer.stop()
            status = self.session.stop()
            self.session = None
            self._set_status(status)
            self.button.setText('Connect gaze')

    def tick(self):
        if self.session is None:
            self.timer.stop()
            return
        self._set_status(self.session.tick())
        if not self.session.running:
            self.timer.stop()
            self.session = None
            self.button.setText('Connect gaze')

    def nativeEvent(self, event_type, message):
        try:
            native = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            if native.message == WM_HOTKEY and native.wParam == HOTKEY_ID:
                self.stop_tracking()
                return True, 0
        except (TypeError, ValueError, OSError):
            pass
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event):
        self.timer.stop()
        if self.session:
            self.session.stop()
            self.session = None
        if self.hotkey_registered:
            self.user32.UnregisterHotKey(self.hwnd, HOTKEY_ID)
            self.hotkey_registered = False
        event.accept()


def main():
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    app = QApplication(sys.argv)
    app.setApplicationName('Mira · Gaze runner')
    app.setQuitOnLastWindowClosed(True)
    window = GameGazeWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
