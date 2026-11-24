import ctypes as C
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from context_operator.core.contracts import Rect, WindowMap
from context_operator.providers.desktop import ScreenProvider
from context_operator.providers.uia_reader import WindowReader


class ReaderTests(unittest.TestCase):
    def fixture(self, controls, count=None):
        reader=WindowReader.__new__(WindowReader)
        reader.win=Mock();reader.uia=Mock();reader.cache=Mock();reader.condition=Mock();reader.secret_condition=Mock()
        reader.names={50000:'ButtonControl'}
        rect=Rect(0,0,800,600)
        reader.win.pid.return_value=7;reader.win.rect.return_value=rect
        reader.win.u.IsWindowVisible.return_value=True;reader.win.u.IsIconic.return_value=False
        root=reader.uia.ElementFromHandle.return_value
        root.FindFirst.return_value=None
        found=root.FindAllBuildCache.return_value
        found.Length=len(controls) if count is None else count
        found.GetElement.side_effect=controls
        return reader,WindowMap(12,7,rect,0,title='Fixture'),root

    def button(self):
        return SimpleNamespace(CachedIsPassword=False,CachedIsOffscreen=False,CachedIsEnabled=True,
            CachedControlType=50000,CachedBoundingRectangle=SimpleNamespace(left=10,top=20,right=50,bottom=60),
            CachedName='Back',CachedAutomationId='back')

    def test_password_guard_is_separate_from_public_property_cache(self):
        reader,surface,root=self.fixture([self.button()])
        root.FindFirst.return_value=object()
        result=reader.scan(surface)
        self.assertTrue(result.blocked_capture)
        self.assertEqual([e.label for e in result.elements],['Back'])
        root.FindFirst.assert_called_once_with(4,reader.secret_condition)
        root.FindAllBuildCache.assert_called_once_with(4,reader.condition,reader.cache)

    def test_connection_wait_is_not_part_of_snapshot_but_read_time_is(self):
        reader,surface,root=self.fixture([self.button()])
        with patch('context_operator.providers.uia_reader.time.monotonic',side_effect=[10.,10.7]):
            result=reader.scan(surface)
        self.assertEqual(result.captured,10.7)
        self.assertEqual(result.elements[0].captured,10.7)
        self.assertFalse(result.blocked_capture)

    def test_incomplete_or_failed_cached_read_cannot_authorize_capture(self):
        for controls,count in (([self.button()],4001),([RuntimeError('provider disconnected')],1)):
            reader,surface,_=self.fixture(controls,count)
            result=reader.scan(surface)
            self.assertTrue(result.blocked_capture)

    def test_reader_worker_is_mta_and_recovers_after_provider_error(self):
        attempts=[];apartments=[];created=[]
        class Reader:
            def __init__(self):
                created.append(threading.get_ident())
                apartment=C.c_int();qualifier=C.c_int()
                status=C.OleDLL('ole32').CoGetApartmentType(C.byref(apartment),C.byref(qualifier))
                self.assert_status=status;apartments.append(apartment.value)
            def scan(self,surface):
                attempts.append(surface.window)
                if len(attempts)==1:raise RuntimeError('provider disconnected')
                return surface
        rect=Rect(0,0,800,600);surface=WindowMap(12,7,rect,time.monotonic(),visible=(rect,))
        win=Mock();win.visible_windows.return_value=(surface,);win.bounds.return_value=rect
        win.foreground.return_value=12;win.cursor.return_value=(20,20)
        provider=ScreenProvider(win,Reader,workers=1)
        try:
            provider.observe()
            first=provider.results.get(timeout=1)
            self.assertEqual(first[2],'RuntimeError')
            provider.results.put(first);provider.observe()
            provider.requested.clear();provider.observe()
            second=provider.results.get(timeout=1)
            self.assertIs(second[1],surface)
            self.assertEqual(apartments,[1,1])  # APTTYPE_MTA, for both reader instances.
            self.assertEqual(len(set(created)),1)
        finally:provider.close()
        self.assertFalse(provider.threads[0].is_alive())

    def test_failed_scan_blocks_capture_without_refreshing_old_outline_then_recovers(self):
        reader,surface,_=self.fixture([self.button()])
        with patch('context_operator.providers.uia_reader.time.monotonic',return_value=100.):
            cached=reader.scan(surface)
        from dataclasses import replace
        surface=replace(surface,visible=(surface.rect,))
        win=Mock();win.visible_windows.return_value=(surface,);win.bounds.return_value=surface.rect
        win.foreground.return_value=12;win.cursor.return_value=(20,20)
        provider=ScreenProvider.__new__(ScreenProvider)
        provider.win=win;provider.sequence=0;provider.practice_window=0;provider.primary=12;provider.attention=None
        provider.maps={12:cached};provider.pending={12};provider.requested={12:100.};provider.vision_requested={}
        provider.pending_limit=2
        provider.jobs=queue.Queue();provider.results=queue.Queue();provider.results.put((surface,None,'COMError'))
        with patch('context_operator.providers.desktop.time.monotonic',return_value=100.5):scene=provider.observe()
        self.assertEqual(provider.last_error,'COMError')
        self.assertEqual(scene.windows[0].captured,100.)
        self.assertEqual(scene.windows[0].elements,cached.elements)
        self.assertTrue(scene.windows[0].blocked_capture)
        self.assertIsNone(provider.vision_scene(scene,None,100.5))
        provider.results.put((surface,replace(cached,captured=100.6),''))
        with patch('context_operator.providers.desktop.time.monotonic',return_value=100.7):scene=provider.observe()
        self.assertEqual(provider.last_error,'')
        self.assertFalse(scene.windows[0].blocked_capture)
        self.assertIsNotNone(provider.vision_scene(scene,None,100.7))


if __name__=='__main__':unittest.main()
