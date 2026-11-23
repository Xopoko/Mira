"""Passive Windows mouse hook. Never injects input or consumes an event."""
import ctypes as C
from ctypes import wintypes as W
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import replace
from ..core.click_learning import HumanClick,PointerMotion


class ClickFilter:
    def __init__(self):self.down=None

    def motion_kind(self,buttons):
        if not buttons:return 'move'
        return 'held' if buttons=={0x201} and self.down is not None else 'drag'

    def event_motion_kind(self,message,buttons,*,injected=False,own_window=False,emit_move=False):
        if injected or own_window or message in (0x204,0x207):return 'blocked'
        if message==0x201:return 'press'
        if message in (0x202,0x205,0x208):return 'release'
        if emit_move:return self.motion_kind(buttons)
        return None

    def feed(self,message,x,y,stamp,*,injected=False,own_window=False):
        if injected:self.down=None;return None
        if message==0x201:  # WM_LBUTTONDOWN
            self.down=None if own_window else (x,y,stamp)
        elif message==0x200 and self.down and math.dist((x,y),self.down[:2])>6:
            self.down=None  # A drag is not a fixation label, even if it returns.
        elif message==0x202:  # WM_LBUTTONUP
            down,self.down=self.down,None
            if down and not own_window and math.dist((x,y),down[:2])<=6 and 0<=stamp-down[2]<=1.5:
                return HumanClick(*down[:2],down[2],stamp)
        return None


class MouseClicks:
    def __init__(self):
        self.events=queue.Queue(maxsize=64);self.worker=None;self.thread_id=0
        self.stop_event=threading.Event();self.error='';self.ready=threading.Event()
        self.motions=deque(maxlen=256)

    def start(self):
        if self.worker and self.worker.is_alive():return
        self.stop_event.clear();self.ready.clear();self.error='';self.drain();self.drain_motion()
        self.worker=threading.Thread(target=self._run,daemon=True,name='human-click-observer');self.worker.start()

    def drain(self):
        result=[]
        while True:
            try:result.append(self.events.get_nowait())
            except queue.Empty:return result

    def stop(self):
        self.stop_event.set()
        if self.thread_id:
            C.windll.user32.PostThreadMessageW(self.thread_id,0x12,0,0)
        if self.worker:self.worker.join(timeout=.7)
        self.drain();self.drain_motion()

    def drain_motion(self):
        result=[]
        while True:
            try:result.append(self.motions.popleft())
            except IndexError:return result

    def _run(self):
        hook=None
        try:
            user=C.WinDLL('user32',use_last_error=True);kernel=C.WinDLL('kernel32',use_last_error=True)
            proc=C.WINFUNCTYPE(C.c_ssize_t,C.c_int,W.WPARAM,W.LPARAM)
            class Mouse(C.Structure):
                _fields_=[('pt',W.POINT),('data',W.DWORD),('flags',W.DWORD),('time',W.DWORD),('extra',C.c_size_t)]
            user.SetWindowsHookExW.argtypes=[C.c_int,proc,W.HINSTANCE,W.DWORD];user.SetWindowsHookExW.restype=W.HHOOK
            user.CallNextHookEx.argtypes=[W.HHOOK,C.c_int,W.WPARAM,W.LPARAM];user.CallNextHookEx.restype=C.c_ssize_t
            user.UnhookWindowsHookEx.argtypes=[W.HHOOK]
            user.GetMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT]
            user.WindowFromPoint.argtypes=[W.POINT];user.WindowFromPoint.restype=W.HWND
            user.GetAncestor.argtypes=[W.HWND,W.UINT];user.GetAncestor.restype=W.HWND
            user.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
            user.SetThreadDpiAwarenessContext.argtypes=[C.c_void_p];user.SetThreadDpiAwarenessContext(C.c_void_p(-4))
            kernel.GetModuleHandleW.argtypes=[W.LPCWSTR];kernel.GetModuleHandleW.restype=W.HMODULE
            kernel.GetTickCount64.restype=C.c_ulonglong
            self.thread_id=kernel.GetCurrentThreadId();filter=ClickFilter()
            last_motion=0.;buttons=set();down_owner=(0,0)
            # Create the message queue before exposing readiness to stop().
            msg=W.MSG();user.PeekMessageW(C.byref(msg),None,0,0,0)
            @proc
            def callback(code,message,data):
                nonlocal last_motion,down_owner
                try:
                    if code>=0 and not self.stop_event.is_set():
                        m=C.cast(data,C.POINTER(Mouse)).contents
                        age=((kernel.GetTickCount64()&0xffffffff)-m.time)&0xffffffff
                        if age>250:
                            filter.down=None;self.motions.append(PointerMotion(0,0,time.monotonic(),kind='blocked'))
                        else:
                            stamp=time.monotonic()-age/1000;injected=bool(m.flags&3)
                            emit_move=message==0x200 and stamp-last_motion>=.008
                            pid=W.DWORD();hwnd=0
                            if emit_move or message in (0x201,0x202,0x204,0x205,0x207,0x208):
                                child=user.WindowFromPoint(m.pt);hwnd=int(user.GetAncestor(child,2) or child or 0)
                                user.GetWindowThreadProcessId(hwnd,C.byref(pid))
                            own=pid.value==os.getpid()
                            if not injected:
                                if message in (0x201,0x204,0x207):buttons.add(message)
                                if message in (0x202,0x205,0x208):buttons.discard(message-1)
                                if message==0x201:down_owner=(hwnd,pid.value)
                            click=filter.feed(message,m.pt.x,m.pt.y,stamp,injected=injected,own_window=own)
                            kind=filter.event_motion_kind(message,buttons,injected=injected,own_window=own,emit_move=emit_move)
                            if kind:
                                self.motions.append(PointerMotion(m.pt.x,m.pt.y,stamp,hwnd,pid.value,kind));last_motion=stamp
                            if click:
                                try:self.events.put_nowait(replace(click,window=down_owner[0],pid=down_owner[1]))
                                except queue.Full:pass
                except Exception as exc:self.error=type(exc).__name__
                return user.CallNextHookEx(None,code,message,data)
            hook=user.SetWindowsHookExW(14,callback,kernel.GetModuleHandleW(None),0)
            if not hook:raise C.WinError(C.get_last_error())
            self.ready.set()
            while not self.stop_event.is_set():
                result=user.GetMessageW(C.byref(msg),None,0,0)
                if result<=0:break
        except Exception as exc:self.error=str(exc)
        finally:
            if hook:user.UnhookWindowsHookEx(hook)
            self.thread_id=0;self.ready.set()
