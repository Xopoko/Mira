"""Offscreen proof using production painter/widgets; no desktop or input capture."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF,QRectF
from PySide6.QtGui import QImage,QPainter,QColor,QPen,QFont,QFontDatabase
from context_operator.ui.overlay import paint_light
from context_operator.ui.window import Window
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS

app=QApplication([]);folder=Path('.local/evidence/mouse');folder.mkdir(parents=True,exist_ok=True)
for name in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
image=QImage(1200,500,QImage.Format_ARGB32_Premultiplied);image.fill(QColor('#0b121e'))
p=QPainter(image);p.setRenderHint(QPainter.Antialiasing)
elements=[]
for row in range(3):
    for col in range(7):
        identity=f'{row}-{col}';rect=QRectF(36+col*162,125+row*99,143,76);elements.append((identity,rect))
        p.setPen(QPen(QColor('#1d2b3c'),1));p.setBrush(QColor('#111c2c'));p.drawRoundedRect(rect,6,6)
        p.setPen(QColor('#70839c'));p.setFont(QFont('Segoe UI',12));p.drawText(rect.adjusted(17,12,-10,-10),identity)
p.setPen(QColor('#9fe9d3'));p.setFont(QFont('Segoe UI',23));p.drawText(36,51,'Light reveals parts of nearby elements')
p.setPen(QColor('#8ca7c1'));p.setFont(QFont('Segoe UI',12));p.drawText(36,83,'Application rendering with the pointer between two tiles')
paint_light(p,elements,QPointF(492,256),270,'1-2',.6);p.end();image.save(str(folder/'radial-light.png'))
c=Controller(dict(DEFAULTS,vision=False,ranker=False));c.timer.stop();w=Window(c)
w.resize(1400,1080);w.show();app.processEvents();w.grab().save(str(folder/'window-offscreen.png'));w.close()
print(json.dumps({'radial_preview':str(folder/'radial-light.png'),'widget_preview':str(folder/'window-offscreen.png')}))
