import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF,QRectF
from PySide6.QtGui import QImage,QPainter
from context_operator.ui.light_cache import LIGHT_CACHE

APP=QApplication.instance() or QApplication([])

def render(center=QPointF(110,100),energy=0.):
    image=QImage(520,260,QImage.Format_ARGB32_Premultiplied);image.fill(0)
    p=QPainter(image)
    LIGHT_CACHE.paint(p,[('primary',QRectF(70,70,80,60)),('neighbor',QRectF(190,70,280,60))],center,180,'primary',energy,None)
    p.end();return image

class LightTests(unittest.TestCase):
    def test_neighbor_is_partly_revealed_and_empty_space_is_transparent(self):
        image=render()
        self.assertGreater(image.pixelColor(195,70).alpha(),image.pixelColor(265,70).alpha())
        self.assertEqual(image.pixelColor(390,70).alpha(),0)
        self.assertEqual(image.pixelColor(110,180).alpha(),0)
        self.assertGreater(image.pixelColor(110,70).alpha(),image.pixelColor(195,70).alpha())
    def test_light_moves_without_moving_the_elements_and_motion_brightens(self):
        before=render();after=render(QPointF(390,100));moving=render(energy=1.)
        self.assertEqual(after.pixelColor(110,70).alpha(),0)
        self.assertGreater(after.pixelColor(390,70).alpha(),before.pixelColor(390,70).alpha())
        self.assertGreater(moving.pixelColor(265,70).alpha(),before.pixelColor(265,70).alpha())

if __name__=='__main__':unittest.main()
