import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint,QPointF,QRectF
from PySide6.QtGui import QImage,QPainter,QPaintEvent
from context_operator.ui.light_cache import LightCache
from context_operator.ui.overlay import paint_scene_light,FocusOverlay
from context_operator.core.contracts import Rect,Element,WindowMap,Scene,Gaze,Focus
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
        cache = LightCache(max_pixels=1_000_000)
        render(cache)
        for dpr in (1.25, 1.5):
            with self.subTest(dpr=dpr):
                actual = render(cache, dpr=dpr)
                self.assertEqual(actual, render(LightCache(), dpr=dpr))
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

    def test_overlay_damage_erases_moved_light_and_expired_idle_prediction(self):
        bounds=Rect(-1200,0,1200,800)
        left=Element('a','A','Button',Rect(-1060,140,120,80),1,pid=1,captured=10)
        right=Element('b','B','Button',Rect(-360,500,120,80),1,pid=1,captured=10)
        predicted=Element('p','P','Button',Rect(-700,220,120,80),1,pid=1,captured=10)
        elements=(left,right,predicted)
        scene=Scene(1,10,bounds,elements,1,windows=(WindowMap(1,1,bounds,10,elements,(bounds,)),))
        forecast=Anticipation(predicted,10,10,1.,'gaze')
        win=Mock();win.rect.return_value=bounds;win.u.GetWindowLongW.return_value=0
        overlay=FocusOverlay(APP.primaryScreen(),win)
        self.addCleanup(overlay.close)
        overlay.timer.stop();overlay.resize(600,400)
        image=QImage(1200,800,QImage.Format_ARGB32_Premultiplied);image.setDevicePixelRatio(2.);image.fill(0)
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.) as clock, \
             patch.object(overlay,'devicePixelRatioF',return_value=2.),patch.object(overlay,'update') as update:
            def frame(now):
                clock.return_value=now
                update.reset_mock();overlay.animate();update.assert_called_once()
                dirty=update.call_args.args[0]
                self.assertFalse(dirty.contains(QPoint(300,300)))
                painter=QPainter(image);painter.setClipRegion(dirty)
                try:
                    with patch('context_operator.ui.overlay.QPainter',return_value=painter,
                               CompositionMode_Source=QPainter.CompositionMode_Source,
                               CompositionMode_SourceOver=QPainter.CompositionMode_SourceOver,
                               Antialiasing=QPainter.Antialiasing):
                        overlay.paintEvent(QPaintEvent(dirty))
                finally:painter.end()

            overlay.sync(Focus(),Gaze(*left.rect.center,10,100,'mouse'),scene,prediction=forecast)
            frame(10.1)
            self.assertGreater(image.pixelColor(200,140).alpha(),0)
            self.assertGreater(image.pixelColor(560,214).alpha(),0)
            once=image.copy();frame(10.1);self.assertEqual(image,once)
            clock.return_value=10.2
            overlay.sync(Focus(),Gaze(*right.rect.center,10.2,100,'mouse'),scene,prediction=forecast)
            frame(10.2)
            self.assertEqual(image.pixelColor(200,140).alpha(),0)
            self.assertGreater(image.pixelColor(900,500).alpha(),0)
            overlay.sync(Focus(),None,scene,prediction=forecast)
            frame(10.25)
            self.assertEqual(image.pixelColor(900,500).alpha(),0)
            self.assertGreater(image.pixelColor(560,214).alpha(),0)
            frame(10.33)
            self.assertEqual(image.pixelColor(560,214).alpha(),0)
            clock.return_value=10.4
            update.reset_mock();overlay.animate();update.assert_not_called()

if __name__=='__main__':unittest.main()
