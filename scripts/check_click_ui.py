"""Offscreen production UI render. No camera or native input hook is started."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS
from context_operator.ui.window import Window

app=QApplication([])
for name in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
c=Controller(dict(DEFAULTS,vision=False,ranker=False));c.timer.stop()
w=Window(c);w.refresh_timer.stop();w.show();app.processEvents()
folder=Path('.local/evidence/clicks');folder.mkdir(parents=True,exist_ok=True)
w.grab().save(str(folder/'control-offscreen.png'));w.close()
print('Production control screen saved; no input source started')
