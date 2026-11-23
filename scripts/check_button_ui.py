"""Render the real controls offscreen; never starts a native input source."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtGui import QFont,QFontDatabase
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS
from context_operator.ui.window import Window
from context_operator.ui.button_binding import ButtonBindingDialog

app=QApplication([])
for name in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
c=Controller(dict(DEFAULTS,vision=False,ranker=False));c.timer.stop()
w=Window(c);w.refresh_timer.stop();w.navigate(3);w.show();w.refresh();app.processEvents()
folder=Path('.local/evidence/usb-button');folder.mkdir(parents=True,exist_ok=True)
w.grab().save(str(folder/'control.png'))
d=ButtonBindingDialog(w,observer=Mock());d.show();app.processEvents();d.grab().save(str(folder/'binding.png'))
cancel=d.findChild(QPushButton)
d.setFocus(Qt.OtherFocusReason)
QTest.keyClick(d,Qt.Key_Tab)
assert cancel.hasFocus() and cancel.text()=='Cancel · Esc'
for key in (Qt.Key_Return,Qt.Key_Enter,Qt.Key_Space):
    QTest.keyClick(cancel,key)
    assert d.isVisible() and d.binding is None and d.keyboard_id is None
QTest.keyClick(cancel,Qt.Key_Escape)
assert not d.isVisible()
d.observer.stop.assert_called_once()
assert not c.active_sources
w.close()
print('Rendered production control and binding dialog; no input sources started')
