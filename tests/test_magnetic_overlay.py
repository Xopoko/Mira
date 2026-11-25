import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRect,QPointF
from PySide6.QtGui import QImage,QPainter,QPaintEvent,QRegion
from context_operator.core.contracts import Element,Focus,Rect,Scene,WindowMap,Gaze
from context_operator.ui.overlay import paint_magnetic_focus,current_focus_target,FocusOverlay,focus_ink

APP=QApplication.instance() or QApplication([])


class MagneticOverlayTests(unittest.TestCase):
    def setUp(self):
        self.bounds=Rect(-200,0,600,400)
        self.a=Element('a','A','Button',Rect(-100,80,100,50),1,pid=2,captured=10)
        self.b=Element('b','B','Button',Rect(100,80,100,50),1,pid=2,captured=10)
        self.scene=Scene(1,10,self.bounds,(self.a,self.b),1,windows=(
            WindowMap(1,2,self.bounds,10,(self.a,self.b),(self.bounds,)),))

    def frame(self,target=None,scene=None,now=10.1,scale=1.):
        image=QImage(600,400,QImage.Format_ARGB32_Premultiplied);image.fill(0)
        p=QPainter(image)
        paint_magnetic_focus(p,scene or self.scene,target or self.a,now,(-200,0),scale)
        p.end();return image

    def test_whole_object_is_lit_without_dot_or_neighbor_field(self):
        image=self.frame()
        for x in (105,150,195):
            color=image.pixelColor(x,80)
            self.assertGreater(color.alpha(),100)
            self.assertGreater(color.red(),color.blue())
        self.assertGreater(image.pixelColor(150,72).blue(),image.pixelColor(150,72).red())
        self.assertEqual(image.pixelColor(150,105),image.pixelColor(160,105))
        self.assertEqual(image.pixelColor(300,80).alpha(),0)
        self.assertEqual(image.pixelColor(250,220).alpha(),0)

    def test_switch_erases_old_shape_and_does_not_draw_a_bridge(self):
        win = Mock()
        win.rect.return_value = self.bounds
        state = SimpleNamespace(highlight_mode='radial', scene=None, focus_target=None, born={},
                                last_tick=10., ink=QRegion(), win=win, winId=lambda: 1,
                                devicePixelRatioF=lambda: 1., rect=lambda: QRect(0, 0, 600, 400),
                                update=Mock())
        image = QImage(600, 400, QImage.Format_ARGB32_Premultiplied)
        image.fill(0)

        def paint_focus(target, now):
            with patch('context_operator.ui.overlay.time.monotonic', return_value=now):
                FocusOverlay.sync(state, Focus(target), Gaze(*target.rect.center, now),
                                  self.scene, highlight_mode='magnetic')
                FocusOverlay.animate(state)
                dirty = state.update.call_args.args[0]
                painter = QPainter(image)
                painter.setClipRegion(dirty)
                try:
                    with patch('context_operator.ui.overlay.QPainter', return_value=painter) as make:
                        make.CompositionMode_Source = QPainter.CompositionMode_Source
                        make.CompositionMode_SourceOver = QPainter.CompositionMode_SourceOver
                        make.Antialiasing = QPainter.Antialiasing
                        FocusOverlay.paintEvent(state, QPaintEvent(dirty))
                finally:
                    painter.end()

        paint_focus(self.a, 10.1)
        self.assertGreater(image.pixelColor(150,80).alpha(),100)
        paint_focus(self.b, 10.2)
        self.assertEqual(image.pixelColor(150,80).alpha(),0)
        self.assertEqual(image.pixelColor(250,80).alpha(),0)
        self.assertGreater(image.pixelColor(350,80).alpha(),100)

    def test_expired_replaced_and_occluded_elements_are_not_painted(self):
        self.assertEqual(self.frame(now=11.3).pixelColor(150,80).alpha(),0)
        changed=replace(self.scene,elements=(replace(self.a,label='new'),self.b))
        self.assertEqual(self.frame(scene=changed).pixelColor(150,80).alpha(),0)
        clipped=replace(self.scene,windows=(replace(self.scene.windows[0],visible=(Rect(-40,0,440,400),)),))
        self.assertEqual(self.frame(scene=clipped).pixelColor(120,80).alpha(),0)
        self.assertGreater(self.frame(scene=clipped).pixelColor(180,80).alpha(),0)

    def test_stale_gaze_clears_damage_once_at_scaled_negative_origin(self):
        win=Mock();win.rect.return_value=self.bounds
        state=SimpleNamespace(highlight_mode='magnetic',scene=self.scene,focus_target=self.a,
            selected_until=0.,magnet_since=10.,magnetic_dirty=False,
            attention=Gaze(-50,100,10),last_tick=10,ink=QRegion(),win=win,
            winId=lambda:1,devicePixelRatioF=lambda:1.25,rect=lambda:QRect(0,0,480,320),update=Mock())
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.1):FocusOverlay.animate(state)
        lit=state.update.call_args.args[0]
        self.assertTrue(lit.contains(QPointF(120,64).toPoint()))
        state.update.reset_mock()
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.3):FocusOverlay.animate(state)
        self.assertEqual(state.update.call_args.args[0],lit)
        self.assertTrue(state.ink.isEmpty())
        state.update.reset_mock()
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.4):FocusOverlay.animate(state)
        state.update.assert_not_called()

    def test_source_expiry_and_full_identity_guard(self):
        gaze=Gaze(-50,100,10)
        self.assertEqual(current_focus_target(self.scene,self.a,gaze,10.1),self.a)
        self.assertIsNone(current_focus_target(self.scene,self.a,gaze,10.3))
        self.assertIsNone(current_focus_target(self.scene,replace(self.a,pid=3),gaze,10.1))
        self.assertEqual(current_focus_target(self.scene,self.a,None,10.1,10.2),self.a)
        self.assertIsNone(current_focus_target(self.scene,self.a,gaze,10.2,10.2))
        self.assertIsNone(current_focus_target(self.scene,replace(self.a,visual_hash='old'),None,10.1,10.2))
        self.assertTrue(focus_ink(None).isEmpty())

    def test_settled_focus_is_not_repainted_for_every_eye_frame(self):
        win=Mock();win.rect.return_value=self.bounds
        state=SimpleNamespace(highlight_mode='magnetic',scene=self.scene,focus_target=self.a,magnet_since=9.,
            selected_until=0.,magnetic_dirty=False,
            attention=Gaze(-50,100,10),last_tick=10,ink=focus_ink(self.a,(-200,0)),win=win,
            winId=lambda:1,devicePixelRatioF=lambda:1.,rect=lambda:QRect(0,0,600,400),update=Mock())
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.1):FocusOverlay.animate(state)
        state.update.assert_not_called()
        state.magnetic_dirty=True
        with patch('context_operator.ui.overlay.time.monotonic',return_value=10.12):FocusOverlay.animate(state)
        state.update.assert_called_once()


if __name__=='__main__':unittest.main()
