"""Exercise the actual Qt selector and render a synthetic focus-switch sequence."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF,QRectF
from PySide6.QtGui import QImage,QPainter,QColor,QPen,QFont,QFontDatabase
from PIL import Image
from context_operator.settings import DEFAULTS,LOCAL
from context_operator.settings import load
from context_operator.controller import Controller
from context_operator.ui.window import Window
from context_operator.ui.overlay import paint_scene_light,paint_magnetic_focus
from context_operator.core.contracts import Element,Rect,Scene,Gaze
from context_operator.core.fusion import Fusion

app=QApplication.instance() or QApplication([])
for font in ('segoeui.ttf','seguisb.ttf','segoeuib.ttf'):
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
out=LOCAL/'evidence/v1_3';out.mkdir(parents=True,exist_ok=True)
c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False,click_learning=False))
c.timer.stop();window=Window(c);window.refresh_timer.stop();window.show();window.navigate(3)
with tempfile.TemporaryDirectory() as folder, patch('context_operator.settings.LOCAL',Path(folder)):
    window.highlight_mode.setCurrentIndex(1);app.processEvents()
    assert c.settings['highlight_mode']=='magnetic'
    assert not window.show_gaze_check.isEnabled() and not window.anticipation_check.isEnabled()
    assert load()['highlight_mode']=='magnetic'
    window.grab().save(str(out/'magnetic-settings.png'))
    with patch('context_operator.ui.window.save',side_effect=OSError('test disk failure')):
        window.highlight_mode.setCurrentIndex(0);app.processEvents()
    assert c.settings['highlight_mode']=='magnetic' and window.highlight_mode.currentData()=='magnetic'
    window.highlight_mode.setCurrentIndex(0);app.processEvents()
    assert window.show_gaze_check.isEnabled() and window.anticipation_check.isEnabled()
    c.busy=True
    window.highlight_mode.setCurrentIndex(1);app.processEvents()
    assert c.settings['highlight_mode']=='radial'
    c.busy=False
window.close()

bounds=Rect(0,0,490,260)
elements=tuple(Element(str(i),name,'Button',Rect(35+i*155,100,110,48),1)
               for i,name in enumerate(('Overview','Profile','Settings')))
magnet=Fusion();radial=Fusion();frames=[];targets=[]
for frame in range(105):
    now=frame/30
    if now<.85:x=90+5*math.sin(now*35)
    elif now<1.2:x=90+(now-.85)/.35*155
    elif now<2.15:x=245+5*math.sin(now*35)
    elif now<2.5:x=245+(now-2.15)/.35*155
    else:x=400+5*math.sin(now*35)
    gaze=Gaze(x,124,now,120)
    scene=Scene(frame,now,bounds,elements,1)
    focus=magnet.choose(scene,gaze,now,magnetic=True)
    normal=radial.choose(scene,gaze,now)
    targets.append(focus.target.id if focus.target else None)
    image=QImage(1000,370,QImage.Format_RGBA8888);image.fill(QColor('#101419'))
    p=QPainter(image);p.setRenderHint(QPainter.Antialiasing)
    for column,title in enumerate(('Follow gaze','Magnetic focus')):
        p.save();p.translate(column*500,0)
        p.setPen(QColor('#edf6f3'));p.setFont(QFont('Segoe UI',16));p.drawText(QRectF(24,14,460,40),title)
        p.translate(0,45)
        for e in elements:
            r=QRectF(*e.rect.as_list());p.setPen(QPen(QColor('#34433f'),1));p.setBrush(QColor('#1b252c'));p.drawRoundedRect(r,8,8)
            p.setPen(QColor('#edf6f3'));p.setFont(QFont('Segoe UI',10));p.drawText(r,0x84,e.label)
        if column:
            paint_magnetic_focus(p,scene,focus.target,now)
        else:
            paint_scene_light(p,scene,QPointF(gaze.x,gaze.y),gaze.radius,now,normal.target.id if normal.target else None)
            p.setPen(QPen(QColor('#beeaff'),1));p.setBrush(QColor('#beeaff'));p.drawEllipse(QPointF(gaze.x,gaze.y),2.5,2.5)
        p.restore()
    p.setPen(QColor('#93a7ad'));p.setFont(QFont('Segoe UI',10))
    p.drawText(QRectF(24,305,950,44),0x84,'Synthetic gaze: jitter → next element → hold')
    p.end()
    frames.append(Image.frombytes('RGBA',(image.width(),image.height()),bytes(image.constBits())).convert('RGB'))
    if frame==55:image.save(str(out/'magnetic-comparison.png'))
transitions=[target for i,target in enumerate(targets) if i==0 or target!=targets[i-1]]
assert transitions==['0','1','2'],transitions
frames[0].save(out/'magnetic-comparison.gif',save_all=True,append_images=frames[1:],duration=33,loop=0)
result={'selector_save_reload_and_failure':'passed','one_target_sequence':transitions,'frames':len(frames),'live_eye_accuracy':False}
(out/'magnetic-ui-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
