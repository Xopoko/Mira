"""Bounded production raster benchmark; synthetic elements, no screen capture/input."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json,time,statistics,argparse
import importlib.util
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage,QPainter
from context_operator.core.contracts import Rect,Element,WindowMap,Scene
from context_operator.ui.overlay import paint_scene_light

def fixture(count):
    bounds=Rect(0,0,5120,1440)
    elements=[Element(str(i),'','Text',Rect((i%64)*80,(i//64)*25,74,21),1,captured=10) for i in range(count)]
    # Real accessible trees also expose nested rows/cards covering much of a window.
    elements.extend(Element('large'+str(i),'','ListItem',Rect(250+i*2,40+i*2,4300-i*4,1300-i*4),1,captured=10) for i in range(30))
    w=WindowMap(1,1,bounds,10,tuple(elements),(bounds,))
    return Scene(1,10,bounds,tuple(elements),1,window_rect=bounds,windows=(w,))

def run(count,frames=25,churn=False,paint=paint_scene_light):
    scene=fixture(count);image=QImage(5120,1440,QImage.Format_ARGB32_Premultiplied);times=[]
    for i in range(frames+3):
        start=time.perf_counter();image.fill(0);p=QPainter(image)
        point=QPointF(500+(i*193)%4000,150+(i*79)%1000) if churn else QPointF(2500+i*3,520+i*2)
        if churn:
            # Content updates and rapidly changing targets force real cache misses.
            from dataclasses import replace
            changed=replace(scene.elements[0],rect=Rect(i*5,10,74,21))
            elements=(changed,)+scene.elements[1:];w=replace(scene.windows[0],elements=elements)
            scene=replace(scene,elements=elements,windows=(w,))
        paint(p,scene,point,230,10.2,selected=str(i if churn else 100))
        p.end()
        if i>=3:times.append((time.perf_counter()-start)*1000)
    return {'elements':len(scene.elements),'frames':frames,'moving_content':churn,'median_ms':round(statistics.median(times),3),'p95_ms':round(sorted(times)[int(len(times)*.95)],3),'max_ms':round(max(times),3)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save');parser.add_argument('--churn',action='store_true');parser.add_argument('--baseline');args=parser.parse_args();app=QApplication([])
    paint=paint_scene_light
    if args.baseline:
        spec=importlib.util.spec_from_file_location('overlay_baseline',args.baseline);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);paint=module.paint_scene_light
    results=[run(count,churn=args.churn,paint=paint) for count in (200,1000,3000)]
    if args.save:
        path=Path(args.save);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2))
    print(json.dumps(results))
