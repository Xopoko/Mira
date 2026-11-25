import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor,QImage,QPainter
from PySide6.QtCore import QRectF,QPointF,QPoint
from context_operator.ui.overview_cache import OverviewCache
from context_operator.ui.light_cache import tile_bounds
from context_operator.ui.theme import RAIL
from context_operator.ui.window import SceneCanvas
from context_operator.core.contracts import Rect,Element,WindowMap,Scene,Focus,Gaze
from context_operator.core.fusion import Fusion

APP=QApplication.instance() or QApplication([])

class OverviewTests(unittest.TestCase):
    def draw(self,cache,rects,dpr=1.25):
        image=QImage(round(500*dpr),round(200*dpr),QImage.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr);image.fill(0);p=QPainter(image)
        cache.paint(p,rects,500,200);p.end();return image

    def test_geometry_cache_tracks_move_removal_dpr_and_memory_limit(self):
        cache=OverviewCache(max_pixels=450000);rect=QRectF(20,30,330,42)
        a=self.draw(cache,[rect]);self.draw(cache,[rect]);self.assertEqual(cache.builds,1)
        b=self.draw(cache,[rect.translated(50,0)])
        self.assertGreater(a.pixelColor(25,40).alpha(),0);self.assertEqual(b.pixelColor(25,40).alpha(),0)
        empty=self.draw(cache,[]);self.assertEqual(empty.pixelColor(100,50).alpha(),0)
        for dpr in (1.,1.25,1.5,2.):self.draw(cache,[rect],dpr)
        self.assertLessEqual(cache.pixels,cache.max_pixels);self.assertLessEqual(len(cache.images),3)

    def test_miniature_tile_is_small_but_covers_continuous_light_at_every_scale(self):
        for radius in (1,40,184,230):
            for point in (QPointF(-.01,-210.2),QPointF(127.9,64.1),QPointF(128.1,64.1)):
                bounds=tile_bounds(point,radius)
                self.assertTrue(bounds.contains(QRectF(point.x()-radius,point.y()-radius,2*radius,2*radius)))
        self.assertLessEqual(tile_bounds(QPointF(200,100),40).width(),112)

    def test_overview_does_not_retain_expired_scene_and_uses_own_light_cache(self):
        bounds=Rect(0,0,1000,600);item=Element('a','','Button',Rect(100,100,60,30),1,pid=1,captured=10)
        window=WindowMap(1,1,bounds,10,(item,),(bounds,))
        c=SimpleNamespace(running=True,scene=Scene(1,10,bounds,(item,),1,windows=(window,)),attention=None,prediction=None,
                          focus=Focus(),fusion=Fusion(),settings={})
        canvas=SceneCanvas(c);canvas.resize(500,240)
        self.addCleanup(canvas.close)
        image=QImage(500,240,QImage.Format_ARGB32_Premultiplied)
        def frame(now):
            image.fill(0);p=QPainter(image)
            with patch('context_operator.ui.window.time.monotonic',return_value=now):canvas.render(p,QPoint())
            p.end()
        c.attention = Gaze(130,115,10.1)
        frame(10.15)
        self.assertGreater(canvas.light_cache.builds, 0)
        from context_operator.ui.light_cache import LIGHT_CACHE
        self.assertIsNot(canvas.light_cache, LIGHT_CACHE)
        c.attention = None
        background = QColor(RAIL)
        element = (126, 58)
        frame(10.2)
        self.assertNotEqual(image.pixelColor(*element), background)
        frame(11.3)
        self.assertEqual(image.pixelColor(*element), background)


if __name__=='__main__':unittest.main()
