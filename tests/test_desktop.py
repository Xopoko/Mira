import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import queue,time,threading,unittest
from dataclasses import replace
from unittest.mock import Mock,patch
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage,QPainter
from PySide6.QtCore import QPointF
from context_operator.core.contracts import Rect,Element,WindowMap,Scene,Gaze,Focus,Intent
from context_operator.core.desktop import visible_regions,compose
from context_operator.core.fusion import Fusion
from context_operator.providers.desktop import ScreenProvider
from context_operator.providers.visual_cache import VisualCache
from context_operator.providers.windows import WindowsOutput
from context_operator.ui.overlay import paint_scene_light
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS

APP=QApplication.instance() or QApplication([])
BOUNDS=Rect(-500,0,1000,600)

def maps(stamp=10):
    a=Element('a','Left','Button',Rect(-70,100,60,40),1,pid=11,captured=stamp)
    b=Element('b','Right','Button',Rect(10,100,60,40),2,pid=22,captured=stamp)
    return (WindowMap(1,11,Rect(-500,0,500,600),stamp,(a,),(Rect(-500,0,500,600),),'Left'),
            WindowMap(2,22,Rect(0,0,500,600),stamp,(b,),(Rect(0,0,500,600),),'Right'))

def desktop(stamp=10):
    windows=maps(stamp);return compose(1,stamp,BOUNDS,windows,{w.window:w for w in windows},1)

class GeometryTests(unittest.TestCase):
    def test_window_clipping_subtracts_occluders_without_negative_origin_loss(self):
        parts=visible_regions(BOUNDS,[Rect(-100,100,200,200)],BOUNDS)
        self.assertEqual(sum(r.w*r.h for r in parts),560000)
        self.assertFalse(any(r.contains(0,200) for r in parts));self.assertTrue(any(r.contains(-499,10) for r in parts))
    def test_two_apps_stay_in_one_scene_and_gaze_crosses_without_rebuilding(self):
        scene=desktop();f=Fusion(0)
        self.assertEqual({e.window for e in scene.elements},{1,2})
        self.assertEqual(f.choose(scene,Gaze(-40,120,10,70),10).target.id,'a')
        self.assertEqual(f.choose(scene,Gaze(40,120,10.02,70),10.02).target.id,'b')
    def test_refreshing_one_window_does_not_make_old_neighbour_fresh(self):
        a,b=maps();new=replace(a,captured=12,elements=tuple(replace(e,captured=12) for e in a.elements))
        scene=compose(2,12,BOUNDS,(new,b),{1:new,2:b},1)
        self.assertEqual([e.id for e in scene.elements],['a'])
        # An already-published scene cannot keep a dead map alive either.
        self.assertFalse(desktop().current_elements(11.3))
    def test_fresh_visual_element_cannot_revive_invalid_scene_or_window(self):
        now = 12.
        scene = desktop(now)
        a, b = scene.windows
        visual = replace(a.elements[0], source='ocr')
        a = replace(a, elements=(visual,))
        scene = replace(scene, elements=(visual, *b.elements), windows=(a, b))
        self.assertIn(visual, scene.current_elements(now))
        for changed in (replace(a, captured=now - 1.25), replace(a, captured=now + .1),
                        replace(a, pid=999), replace(a, blocked_capture=True), None):
            with self.subTest(window=changed):
                windows = (changed, b) if changed else (b,)
                self.assertEqual(replace(scene, windows=windows).current_elements(now), b.elements)
        for captured in (now - 3, now + .1):
            self.assertEqual(replace(scene, captured=captured).current_elements(now), ())
        for captured in (now - 1.25, now + .1):
            old = replace(visual, captured=captured)
            window = replace(a, elements=(old,))
            self.assertEqual(replace(scene, elements=(old, *b.elements), windows=(window, b))
                             .current_elements(now), b.elements)
        locked_capture = replace(b, blocked_capture=True)
        self.assertEqual(replace(scene, windows=(locked_capture,)).current_elements(now), b.elements)

    def test_current_target_matches_a_fresh_enabled_element(self):
        scene = desktop()
        target = scene.elements[0]
        current = replace(target, captured=10.1)
        window = replace(scene.windows[0], elements=(current,))
        scene = replace(scene, elements=(current,), windows=(window,))
        self.assertIs(scene.current_target(target, 10.1), current)
        for replacement in (replace(current, label='Replacement'), replace(current, enabled=False),
                            replace(current, rect=Rect(-70, 100, 0, 40))):
            changed = replace(scene, elements=(replacement,), windows=(replace(window, elements=(replacement,)),))
            old = replacement if replacement.rect.w == 0 else target
            self.assertIsNone(changed.current_target(old, 10.1))
        self.assertIsNone(scene.current_target(target, 11.25))

    def test_move_close_pid_change_and_occlusion_drop_old_targets(self):
        a,b=maps()
        for changed in (replace(b,rect=Rect(10,0,490,600)),replace(b,pid=999),replace(b,visible=(Rect(200,0,300,600),))):
            result=compose(2,10.1,BOUNDS,(a,changed),{1:a,2:b},1)
            self.assertNotIn('b',[e.id for e in result.elements])
        self.assertEqual([e.id for e in compose(2,10.1,BOUNDS,(a,),{1:a,2:b},1).elements],['a'])
    def test_one_halo_paints_both_apps_and_never_the_occluded_region(self):
        scene=desktop();image=QImage(1000,600,QImage.Format_ARGB32_Premultiplied);image.fill(0)
        p=QPainter(image);paint_scene_light(p,scene,QPointF(0,120),120,10,origin=(-500,0));p.end()
        self.assertGreater(image.pixelColor(460,100).alpha(),0);self.assertGreater(image.pixelColor(540,100).alpha(),0)
        a,b=maps();b=replace(b,visible=(Rect(40,0,460,600),))
        clipped=compose(2,10,BOUNDS,(a,b),{1:a,2:b},1)
        image.fill(0);p=QPainter(image);paint_scene_light(p,clipped,QPointF(0,120),120,10,origin=(-500,0));p.end()
        self.assertEqual(image.pixelColor(525,100).alpha(),0)
        self.assertGreater(image.pixelColor(550,100).alpha(),0)

class ProviderTests(unittest.TestCase):
    def test_slow_window_cannot_block_fast_neighbour_and_stop_discards_work(self):
        started=time.monotonic();windows=maps(started);release=threading.Event()
        class Reader:
            def scan(self,surface):
                if surface.window==1:release.wait(2)
                now=time.monotonic();return replace(surface,captured=now,elements=tuple(replace(e,captured=now) for e in surface.elements))
        win=Mock();win.visible_windows.return_value=windows;win.foreground.return_value=1
        win.cursor.return_value=(0,120);win.bounds.return_value=BOUNDS
        p=ScreenProvider(win,Reader,workers=2)
        try:
            deadline=time.monotonic()+.8;result=p.observe()
            while not any(e.window==2 for e in result.elements) and time.monotonic()<deadline:
                time.sleep(.01);result=p.observe()
            self.assertEqual({e.window for e in result.elements},{2});win.activate.assert_not_called()
            release.set()
            while len(result.windows)<2 and time.monotonic()<deadline:
                time.sleep(.01);result=p.observe()
            self.assertEqual(len(result.windows),2)
            win.visible_windows.return_value=(windows[1],)
            self.assertEqual({e.window for e in p.observe().elements},{2})
        finally:release.set();p.close()
        self.assertTrue(all(not t.is_alive() for t in p.threads))

    def test_fast_scans_under_load_do_not_leave_windows_pending(self):
        windows=tuple(WindowMap(n,n,BOUNDS,0.,visible=(BOUNDS,)) for n in range(1,7))
        win=Mock();win.visible_windows.return_value=windows;win.foreground.return_value=1
        win.cursor.return_value=BOUNDS.center;win.bounds.return_value=BOUNDS
        class Reader:
            def scan(self,surface):
                return replace(surface,captured=time.monotonic())
        provider=ScreenProvider(win,Reader,workers=1)
        self.addCleanup(provider.close)
        results=provider.results;submit=provider.jobs.put_nowait;completed=threading.Event()

        def finish_scan(result):
            try:results.put_nowait(result)
            finally:completed.set()

        def submit_and_wait(surface):
            completed.clear()
            submit(surface)
            self.assertTrue(completed.wait(1))

        # Let the reader finish each job before the producer can submit the next.
        provider.results=Mock(wraps=results,put=finish_scan,put_nowait=finish_scan)
        provider.jobs.put_nowait=submit_and_wait
        for _ in range(4):
            scene=provider.observe()
        self.assertEqual({window.window for window in scene.windows if window.captured>0},set(range(1,7)))

    def test_ocr_rotates_nearby_windows_and_capture_is_clipped(self):
        p=ScreenProvider.__new__(ScreenProvider);p.vision_requested={};scene=desktop()
        first=p.vision_scene(scene,Gaze(0,120,10),10);second=p.vision_scene(scene,Gaze(0,120,10.1),10.1)
        self.assertEqual({first.window,second.window},{1,2})
        p.win=Mock();p.win.title.return_value='Fixture';p.win.pid.return_value=22;p.win.rect.return_value=second.window_rect;p.win.unobscured.return_value=False
        self.assertIsNone(p.capture(second,Rect(-100,0,200,300)))
        self.assertEqual(p.win.unobscured.call_args.args[1],Rect(0,0,100,300))

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False,click_learning=False));self.c.timer.stop()
        self.c.running=True;self.c.scene=desktop();self.c.focus=Focus(self.c.scene.elements[0]);self.c.fusion.current='a'
    def tearDown(self):self.c.close()
    def test_neighbour_refresh_does_not_reset_current_focus_or_action_epoch(self):
        a,b=maps();b=replace(b,elements=(replace(b.elements[0],id='changed'),))
        s=compose(2,10.1,BOUNDS,(a,b),{1:a,2:b},2)
        epoch=self.c.action_epoch
        self.c._scene(s,self.c.generation,epoch)
        self.assertEqual(self.c.fusion.current,'a');self.assertEqual(self.c.action_epoch,epoch)
    def test_ocr_result_for_inactive_neighbour_survives_unrelated_window_change(self):
        now=time.monotonic();self.c.scene=desktop(now)
        a,b=self.c.scene.windows
        source=Scene(1,now,BOUNDS,b.elements,2,'Right',0,b.rect,(b,))
        extra=Element('ocr','Icon','Icon',Rect(100,200,40,40),2,'ocr',True,22,captured=now)
        self.c._vision((source,now,(extra,),self.c.action_epoch),self.c.generation)
        self.c._scene(self.c.scene,self.c.generation,self.c.action_epoch)
        self.assertIn('ocr',[e.id for e in self.c.scene.elements])
        moved=replace(b,rect=Rect(10,0,500,600))
        self.c._scene(compose(3,now,BOUNDS,(a,moved),{1:a,2:moved},1),self.c.generation,self.c.action_epoch)
        self.assertNotIn('ocr',[e.id for e in self.c.scene.elements])
    def test_verified_icon_survives_caption_change_in_its_own_window(self):
        now=time.monotonic();self.c.scene=desktop(now);a,b=self.c.scene.windows
        source=Scene(1,now,BOUNDS,b.elements,2,'Right',0,b.rect,(b,))
        extra=Element('icon','Icon','Icon',Rect(100,200,40,40),2,'icon',True,22,captured=now)
        changed=replace(b,elements=(replace(b.elements[0],id='new-caption',label='Streaming'),))
        self.c.scene=compose(2,now,BOUNDS,(a,changed),{1:a,2:changed},1)
        self.c._vision((source,now,(extra,),self.c.action_epoch),self.c.generation)
        self.c._scene(self.c.scene,self.c.generation,self.c.action_epoch)
        self.assertIn('icon',[e.id for e in self.c.scene.elements])
    def test_output_checks_target_window_rect_not_only_foreground_window(self):
        now=time.monotonic();s=desktop(now);target=s.elements[1]
        output=WindowsOutput();output.win=Mock();output.win.u.IsWindow.return_value=True
        output.win.pid.return_value=22;output.win.rect.return_value=Rect(20,0,480,600);output.semantic=Mock()
        with self.assertRaises(RuntimeError):output.execute(Intent('click'),target,s)
        output.semantic.assert_not_called()

    def test_unchanged_neighbour_target_survives_primary_resize_but_own_resize_rejects(self):
        now=time.monotonic();source=desktop(now);a,b=source.windows
        changed=replace(a,rect=Rect(-500,0,480,600))
        self.c.scene=compose(2,now,BOUNDS,(changed,b),{1:changed,2:b},1)
        self.c._decision(Intent('select',target_id='b'),source,self.c.generation,now)
        self.assertGreater(self.c.fusion.selection_until(b.elements[0],now),now)
        self.c.fusion.reset();moved=replace(b,rect=Rect(20,0,480,600))
        self.c.scene=compose(3,now,BOUNDS,(a,moved),{1:a,2:moved},1)
        self.c._decision(Intent('select',target_id='b'),source,self.c.generation,now)
        self.assertEqual(self.c.fusion.selection_until(b.elements[0],now),0.)

    def test_verified_ocr_of_two_windows_stays_present_during_slow_third_inference(self):
        # Cache verification has its own cadence, independent of model completion.
        now=time.monotonic()
        for tick in range(20):
            stamp=now+tick*.1;source=desktop(stamp)
            self.c.scene=source
            for window in source.windows:
                item=Element('ocr'+str(window.window),'Fixture','Icon',Rect(-300 if window.window==1 else 300,200,40,40),window.window,'ocr',True,window.pid,captured=stamp)
                local=Scene(tick,stamp,BOUNDS,window.elements,window.window,'',0,window.rect,(window,))
                with patch('context_operator.controller.time.monotonic',return_value=stamp):
                    self.c._vision((local,stamp,(item,),self.c.action_epoch),self.c.generation)
            with patch('context_operator.controller.time.monotonic',return_value=stamp):self.c._scene(source,self.c.generation,self.c.action_epoch)
            self.assertEqual({e.id for e in self.c.scene.elements if e.source=='ocr'},{'ocr1','ocr2'})

    def test_empty_screen_map_displays_scan_error_until_recovery(self):
        empty=replace(desktop(time.monotonic()),elements=(),windows=())
        self.c.scene_provider=Mock(last_error='COMError')
        self.c._scene(empty,self.c.generation,self.c.action_epoch)
        self.assertEqual(self.c.status,'Can’t read screen elements (COMError)')
        self.c.scene_provider.last_error=''
        self.c._scene(empty,self.c.generation,self.c.action_epoch)
        self.assertEqual(self.c.status,'Ready · connect iPhone or name an element')

class VisualCacheTests(unittest.TestCase):
    def test_close_releases_client_after_unread_results(self):
        entered=threading.Event();release=threading.Event()
        next_started=threading.Event();closed=threading.Event()
        first=Image.new('RGB',(20,20));other=first.copy();rect=Rect(0,0,20,20)
        class Client:
            child=None
            def ready(self,stop):pass
            def observe(self,image,stop):
                if image is first:entered.set();release.wait(2)
                else:next_started.set()
                return {'items':[]}
            def close(self):closed.set()
        cache=VisualCache(threading.Event(),Client)
        results=cache.results

        def finish_frame_after_empty_read():
            try:return results.get_nowait()
            except queue.Empty:
                release.set()
                self.assertTrue(next_started.wait(1))
                raise

        try:
            cache.lookup_frame((1,),rect,first)
            self.assertTrue(entered.wait(1))
            cache.lookup_frame((2,),rect,other)
            # The worker publishes the first result and takes the second frame
            # after this lookup sees an empty result queue.
            with patch.object(cache,'results',Mock(wraps=results,get_nowait=finish_frame_after_empty_read)):
                cache.lookup_frame((3,),rect,other)
            self.assertEqual(len(cache.pending),2)
        finally:release.set();cache.close()
        self.assertTrue(closed.is_set())
        self.assertFalse(cache.worker.is_alive())

if __name__=='__main__':unittest.main()
