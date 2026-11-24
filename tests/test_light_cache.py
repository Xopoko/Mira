import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF,QRectF,QRect
from PySide6.QtGui import QImage,QPainter,QPaintEvent
from context_operator.ui.light_cache import LightCache
from context_operator.ui.overlay import ink_region,paint_scene_light,FocusOverlay
from context_operator.core.contracts import Rect,Element,WindowMap,Scene,Gaze
from context_operator.core.anticipation import Anticipation

APP=QApplication.instance() or QApplication([])

def render(cache,center=QPointF(100,100),selected='a',elements=None,dpr=1.):
    image=QImage(int(600*dpr),int(400*dpr),QImage.Format_ARGB32_Premultiplied);image.setDevicePixelRatio(dpr);image.fill(0)
    p=QPainter(image);cache.paint(p,elements or [('a',QRectF(80,80,60,50)),('b',QRectF(190,80,60,50))],center,160,selected,0,None);p.end();return image

class CacheTests(unittest.TestCase):
    def test_subpixel_motion_reuses_geometry_but_moves_light(self):
        cache=LightCache();a=render(cache);builds=cache.builds;b=render(cache,QPointF(120,100))
        self.assertEqual(cache.builds,builds)
        self.assertNotEqual(a.pixelColor(210,80).alpha(),b.pixelColor(210,80).alpha())
    def test_selection_and_geometry_invalidate_pixels(self):
        cache=LightCache();a=render(cache);before=cache.builds;b=render(cache,selected='b')
        self.assertGreater(cache.builds,before);self.assertGreater(a.pixelColor(100,80).red(),a.pixelColor(100,80).blue())
        self.assertGreater(b.pixelColor(100,80).blue(),b.pixelColor(100,80).red())
        moved=render(cache,elements=[('a',QRectF(320,80,60,50))]);self.assertEqual(moved.pixelColor(100,80).alpha(),0)
    def test_bounded_cache_and_dpr_separation(self):
        cache=LightCache(max_pixels=1_000_000);render(cache);render(cache,dpr=1.25);render(cache,dpr=1.5)
        for x in range(150,1200,130):render(cache,QPointF(x,110))
        self.assertLessEqual(cache.pixels,cache.max_pixels);self.assertLessEqual(len(cache.images),16)
        self.assertEqual(cache.pixels,sum(i.width()*i.height() for i in cache.images.values()))
    def test_stretched_editor_edges_have_no_tile_seams_at_fractional_dpi(self):
        for dpr in (1.,1.25,2.):
            cache=LightCache();image=QImage(int(1000*dpr),int(150*dpr),QImage.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(dpr);image.fill(0);p=QPainter(image)
            cache._shape(p,QRectF(25,30,891,55),True,1.,dpr,QRectF(0,0,1000,150));p.end()
            for x in (35,41,42,89,400,900,901,907):
                self.assertGreater(image.pixelColor(round(x*dpr),round(30*dpr)).alpha(),100)
            self.assertEqual(image.pixelColor(0,0).alpha(),0)
    def test_damage_clears_old_light_and_separate_prediction_at_negative_origin(self):
        prediction=Anticipation(Element('p','','Button',Rect(350,100,50,30)),10,10,1.)
        old=ink_region(QPointF(-850,100),100,None,(-1000,0),.5)
        new=ink_region(QPointF(-500,100),100,prediction,(-1000,0),.5)
        dirty=old.united(new)
        self.assertTrue(dirty.contains(QPointF(75,50).toPoint()));self.assertTrue(dirty.contains(QPointF(250,50).toPoint()))
        self.assertTrue(dirty.contains(QPointF(685,55).toPoint()));self.assertFalse(dirty.contains(QPointF(450,250).toPoint()))
    def test_freshness_and_occlusion_are_checked_after_cached_geometry(self):
        bounds=Rect(-200,0,600,400);el=Element('a','','Button',Rect(-100,70,60,40),1,pid=1,captured=10)
        w=WindowMap(1,1,bounds,10,(el,),(bounds,));scene=Scene(1,10,bounds,(el,),1,windows=(w,))
        def frame(now,source=scene):
            image=QImage(600,400,QImage.Format_ARGB32_Premultiplied);image.fill(0);p=QPainter(image)
            paint_scene_light(p,source,QPointF(-80,90),160,now,origin=(-200,0));p.end();return image
        self.assertGreater(frame(10.1).pixelColor(120,70).alpha(),0)
        self.assertEqual(frame(12).pixelColor(120,70).alpha(),0)
        from dataclasses import replace
        hidden=replace(scene,windows=(replace(w,visible=(Rect(0,0,400,400),)),),elements=())
        self.assertEqual(frame(10.2,hidden).pixelColor(120,70).alpha(),0)
    def test_idle_prediction_expiry_clears_once_then_stops_repainting(self):
        from PySide6.QtGui import QRegion
        forecast=Anticipation(Element('p','','Button',Rect(100,100,60,30)),10,10,1.)
        win=Mock();win.rect.return_value=Rect(0,0,1000,700)
        state=SimpleNamespace(attention=None,center=None,previous=None,energy=0.,last_tick=10,prediction=forecast,
                              highlight_mode='radial',scene=None,focus_target=None,selected_until=0.,
                              ink=QRegion(),win=win,winId=lambda:1,devicePixelRatioF=lambda:1.,rect=lambda:QRect(0,0,1000,700),update=Mock())
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.1):FocusOverlay.animate(state)
        lit=state.update.call_args.args[0];self.assertFalse(lit.isEmpty())
        state.update.reset_mock()
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.3):FocusOverlay.animate(state)
        self.assertEqual(state.update.call_args.args[0],lit);self.assertTrue(state.ink.isEmpty())
        state.update.reset_mock()
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.4):FocusOverlay.animate(state)
        state.update.assert_not_called()

    def test_gaze_prediction_uses_its_own_expiry_and_exact_scene_identity(self):
        bounds=Rect(0,0,600,400);element=Element('a','A','Button',Rect(100,100,60,30),1,pid=5,captured=10)
        scene=Scene(1,10,bounds,(element,),1,window_rect=bounds)
        forecast=Anticipation(element,10.22,10.18,.7,'gaze')
        def frame(now,source_scene=scene,source_forecast=forecast):
            image=QImage(600,400,QImage.Format_ARGB32_Premultiplied);image.fill(0);p=QPainter(image)
            paint_scene_light(p,source_scene,None,100,now,prediction=source_forecast);p.end();return image
        fresh=frame(10.4).pixelColor(130,98)
        self.assertGreater(fresh.blue(),fresh.green())
        self.assertEqual(frame(forecast.valid_until).pixelColor(130,98).alpha(),0)
        replacement=Element('a','Changed','Edit',element.rect,1,pid=5,captured=10)
        changed=Scene(2,10,bounds,(replacement,),1,window_rect=bounds)
        self.assertEqual(frame(10.4,changed).pixelColor(130,98).alpha(),0)

    def test_translucent_paint_erases_old_frames_on_motion_and_pause(self):
        bounds=Rect(0,0,600,400);element=Element('a','','Button',Rect(70,70,60,40),1,pid=1,captured=10)
        window=WindowMap(1,1,bounds,10,(element,),(bounds,))
        scene=Scene(1,10,bounds,(element,),1,windows=(window,))
        win=Mock();win.rect.return_value=bounds
        state=SimpleNamespace(prediction=None,center=QPointF(100,100),attention=Gaze(100,100,10,80),
            scene=scene,focus_target=None,selected_until=0.,highlight_mode='radial',energy=0.,born={},show_point=False,
            win=win,winId=lambda:1,devicePixelRatioF=lambda:1.)
        image=QImage(600,400,QImage.Format_ARGB32_Premultiplied);image.fill(0)
        def paint():
            p=QPainter(image)
            with patch('context_operator.ui.overlay.QPainter',return_value=p) as constructor, \
                 patch('context_operator.ui.overlay.time.monotonic',return_value=10.1):
                constructor.CompositionMode_Source=QPainter.CompositionMode_Source
                constructor.CompositionMode_SourceOver=QPainter.CompositionMode_SourceOver
                constructor.Antialiasing=QPainter.Antialiasing
                FocusOverlay.paintEvent(state,QPaintEvent(QRect(0,0,600,400)))
            p.end()
        paint();once=image.copy();paint();self.assertEqual(image,once)
        state.center=QPointF(450,300);paint();self.assertEqual(image.pixelColor(100,70).alpha(),0)
        state.center=QPointF(100,100);paint();self.assertGreater(image.pixelColor(100,70).alpha(),0)
        state.center=state.attention=None;paint();self.assertEqual(image.pixelColor(100,70).alpha(),0)

if __name__=='__main__':unittest.main()
