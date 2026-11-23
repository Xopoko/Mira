"""Offline numeric timing and production Qt rendering, without live input."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase,QImage,QPainter,QColor,QPen
from PySide6.QtCore import QRectF,QPointF,Qt
from context_operator.core.anticipation import IntentPredictor,OnlineRanker
from context_operator.core.click_learning import PointerMotion
from context_operator.core.contracts import Rect,Element,Scene,Gaze
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS
from context_operator.ui.window import Window
from context_operator.ui.overlay import paint_light,paint_prediction

out=Path('.local/evidence/anticipation');out.mkdir(parents=True,exist_ok=True)
p=IntentPredictor();bounds=Rect(0,0,5120,1440)
elements=tuple(Element(str(i),'Button','Button',Rect(50+(i%30)*165,70+(i//30)*150,130,70),10,pid=20) for i in range(240))
timings=[]
for i in range(84):
    now=10+i*.025;p.observe(PointerMotion(100+i*25,570,now,10,20));s=Scene(i,now,bounds,elements)
    start=time.perf_counter();p.predict(s,Gaze(3500,550,now),now);elapsed=(time.perf_counter()-start)*1000
    if i>=20:timings.append(elapsed)
model=OnlineRanker();x=np.random.default_rng(8).normal(size=(48,18));training=[]
for _ in range(64):
    start=time.perf_counter();model.learn(x,4);training.append((time.perf_counter()-start)*1000)
result=dict(scope='synthetic numeric only; 240 screen candidates, 48 ranked',
            prediction_median_ms=float(np.median(timings)),prediction_p95_ms=float(np.percentile(timings,95)),
            learning_median_ms=float(np.median(training)),learning_p95_ms=float(np.percentile(training,95)))
(out/'timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

app=QApplication([])
for name in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
c=Controller(dict(DEFAULTS,vision=False,ranker=False));c.timer.stop()
w=Window(c);w.refresh_timer.stop();w.show();app.processEvents();w.grab().save(str(out/'control-offscreen.png'));w.close()

image=QImage(1100,330,QImage.Format_ARGB32_Premultiplied);image.fill(QColor('#0c1422'))
paint=QPainter(image);paint.setRenderHint(QPainter.Antialiasing);paint.setFont(QFont('Segoe UI',15))
rects=[('current',QRectF(75,140,185,65)),('next',QRectF(660,140,185,65)),('other',QRectF(880,140,150,65))]
for name,rect in rects:
    paint.setPen(QPen(QColor('#344357'),1));paint.setBrush(QColor('#172334'));paint.drawRoundedRect(rect,6,6)
paint_light(paint,rects,QPointF(150,170),180,'current')
paint_prediction(paint,rects[1][1],.95)
paint.setPen(QColor('#e8eef4'));paint.drawText(rects[0][1],Qt.AlignCenter,'Current selection');paint.drawText(rects[1][1],Qt.AlignCenter,'Next target')
paint.setPen(QColor('#89daff'));paint.drawText(QRectF(55,40,450,50),Qt.AlignCenter,'Cyan + yellow · current focus')
paint.setPen(QColor('#c6a0ff'));paint.drawText(QRectF(590,40,470,50),Qt.AlignCenter,'Violet · early prediction')
paint.setPen(QColor('#a2b0c4'));paint.setFont(QFont('Segoe UI',11));paint.drawText(QRectF(50,265,1000,30),Qt.AlignCenter,'Synthetic rendering sample')
paint.end();image.save(str(out/'glow-offscreen.png'))
