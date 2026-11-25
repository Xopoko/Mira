"""Window geometry and input for Windows: UI Automation calls first, synthetic input otherwise."""
import ctypes as C
from ctypes import wintypes as W
import hashlib
import os
import time
import threading
from ..core.contracts import Rect, WindowMap, MAP_TTL, SCENE_TTL, TARGET_ACTIONS, WINDOWS_ACTIONS, SCROLL_DIRECTIONS
from ..core.desktop import visible_regions

class MI(C.Structure):
    _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),('dwFlags',W.DWORD),('time',W.DWORD),('extra',C.c_size_t)]
class KI(C.Structure):
    _fields_=[('vk',W.WORD),('scan',W.WORD),('flags',W.DWORD),('time',W.DWORD),('extra',C.c_size_t)]
class IU(C.Union): _fields_=[('mi',MI),('ki',KI)]
class INPUT(C.Structure): _anonymous_=['u'];_fields_=[('type',W.DWORD),('u',IU)]

class Windows:
    def __init__(self):
        self.u=C.WinDLL('user32',use_last_error=True)
        declarations={
            'GetForegroundWindow':([],W.HWND),'GetAncestor':([W.HWND,W.UINT],W.HWND),
            'GetWindowRect':([W.HWND,C.POINTER(W.RECT)],W.BOOL),
            'GetWindowTextW':([W.HWND,W.LPWSTR,C.c_int],C.c_int),
            'GetClassNameW':([W.HWND,W.LPWSTR,C.c_int],C.c_int),
            'GetWindowThreadProcessId':([W.HWND,C.POINTER(W.DWORD)],W.DWORD),
            'GetCursorPos':([C.POINTER(W.POINT)],W.BOOL),
            'WindowFromPoint':([W.POINT],W.HWND),
            'GetSystemMetrics':([C.c_int],C.c_int),'GetAsyncKeyState':([C.c_int],W.SHORT),
            'IsWindow':([W.HWND],W.BOOL),'IsWindowVisible':([W.HWND],W.BOOL),
            'IsIconic':([W.HWND],W.BOOL),'SetForegroundWindow':([W.HWND],W.BOOL),
            'ShowWindow':([W.HWND,C.c_int],W.BOOL),
            'SendInput':([W.UINT,C.POINTER(INPUT),C.c_int],W.UINT),
            'SetWindowDisplayAffinity':([W.HWND,W.DWORD],W.BOOL),
            'SetWindowLongW':([W.HWND,C.c_int,W.LONG],W.LONG),
            'GetWindowLongW':([W.HWND,C.c_int],W.LONG),
            'SetThreadDpiAwarenessContext':([C.c_void_p],C.c_void_p),
            'EnumWindows':([C.c_void_p,W.LPARAM],W.BOOL),
        }
        for name,(args,result) in declarations.items():
            fn=getattr(self.u,name);fn.argtypes=args;fn.restype=result
        self.u.SetThreadDpiAwarenessContext(C.c_void_p(-4))

    def bounds(self):return Rect(*(self.u.GetSystemMetrics(i) for i in (76,77,78,79)))
    def foreground(self):return int(self.u.GetForegroundWindow() or 0)
    def pid(self,hwnd):
        pid=W.DWORD();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid));return pid.value
    def rect(self,hwnd):
        r=W.RECT()
        if not self.u.GetWindowRect(hwnd,C.byref(r)):raise RuntimeError('Window closed')
        return Rect(r.left,r.top,r.right-r.left,r.bottom-r.top)
    def title(self,hwnd):
        buf=C.create_unicode_buffer(512);self.u.GetWindowTextW(hwnd,buf,len(buf));return buf.value
    def cursor(self):
        p=W.POINT();self.u.GetCursorPos(C.byref(p));return p.x,p.y
    def visible_windows(self,practice=0):
        bounds=self.bounds();surfaces=[];occluders=[]
        dwm=C.WinDLL('dwmapi');dwm.DwmGetWindowAttribute.argtypes=[W.HWND,W.DWORD,C.c_void_p,W.DWORD]
        @C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
        def visit(hwnd,_):
            hwnd=int(hwnd)
            if not self.u.IsWindowVisible(hwnd) or self.u.IsIconic(hwnd):return True
            if self.u.GetWindowLongW(hwnd,-20)&0x20:return True
            cloaked=W.DWORD()
            if dwm.DwmGetWindowAttribute(hwnd,14,C.byref(cloaked),C.sizeof(cloaked))==0 and cloaked.value:return True
            try:rect=self.rect(hwnd)
            except RuntimeError:return True
            frame=W.RECT()
            frame_rect=(Rect(frame.left,frame.top,frame.right-frame.left,frame.bottom-frame.top)
                        if dwm.DwmGetWindowAttribute(hwnd,9,C.byref(frame),C.sizeof(frame))==0 else rect)
            parts=visible_regions(frame_rect,occluders,bounds);occluders.append(frame_rect)
            if not parts:return True
            pid=self.pid(hwnd)
            if practice and hwnd!=practice:return True
            if pid==os.getpid() and hwnd!=practice:return True
            name=C.create_unicode_buffer(100);self.u.GetClassNameW(hwnd,name,len(name))
            if name.value in ('Progman','WorkerW') and sum(r.w*r.h for r in parts)<100:return True
            surfaces.append(WindowMap(hwnd,pid,rect,0.,visible=parts,title=self.title(hwnd)))
            return True
        self.u.EnumWindows(visit,0)
        return tuple(surfaces)
    def at(self,x,y):
        hwnd=self.u.WindowFromPoint(W.POINT(int(x),int(y)))
        return int(self.u.GetAncestor(hwnd,2) or hwnd or 0)
    def emergency(self):
        return all(self.u.GetAsyncKeyState(k)&0x8000 for k in (0x11,0x12,0x7B))
    def shift_down(self):
        return bool(self.u.GetAsyncKeyState(0x10)&0x8000)
    def unobscured(self,hwnd,region):
        """Only capture pixels owned by this visible window, even without activation."""
        found=False;clear=True
        @C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
        def visit(other,_):
            nonlocal found,clear
            if int(other)==hwnd:found=True;return False
            if not self.u.IsWindowVisible(other) or self.u.IsIconic(other):return True
            if self.u.GetWindowLongW(other,-20)&0x20:return True  # click-through focus overlays
            try:r=self.rect(other)
            except RuntimeError:return True
            if min(r.x+r.w,region.x+region.w)>max(r.x,region.x) and min(r.y+r.h,region.y+region.h)>max(r.y,region.y):
                clear=False;return False
            return True
        self.u.EnumWindows(visit,0)
        return found and clear and bool(self.u.IsWindowVisible(hwnd)) and not self.u.IsIconic(hwnd)
    def send(self,events):
        data=(INPUT*len(events))(*events)
        if self.u.SendInput(len(data),data,C.sizeof(INPUT))!=len(data):
            raise RuntimeError('Windows received only part of the input.')
    def mouse(self,flags,data=0,x=0,y=0):return INPUT(type=0,u=IU(mi=MI(x,y,data&0xffffffff,flags,0,0)))
    def key(self,vk=0,scan=0,flags=0):return INPUT(type=1,u=IU(ki=KI(vk,scan,flags,0,0)))
    def move_event(self,x,y):
        b=self.bounds()
        return self.mouse(0x8000|0x4000|1,x=round((x-b.x)*65535/max(1,b.w-1)),y=round((y-b.y)*65535/max(1,b.h-1)))
    def activate(self,hwnd):
        if self.foreground()!=hwnd:
            if self.u.IsIconic(hwnd):self.u.ShowWindow(hwnd,9)
            self.u.SetForegroundWindow(hwnd)
            time.sleep(.055)
        if self.foreground()!=hwnd:raise RuntimeError('Switch to the window you want to control. Windows couldn’t focus it.')

class WindowsOutput:
    def __init__(self):self.win=Windows();self.guards=threading.local()
    def set_guard(self,guard):self.guards.allowed=guard
    def is_active(self):return getattr(self.guards,'allowed',lambda:True)()
    def _send(self,events):
        if not self.is_active():raise RuntimeError('Control stopped')
        if self.win.shift_down():raise RuntimeError('Release Shift and repeat the command. This target uses standard Windows input.')
        self.win.send(events)
    def semantic(self,intent,target):
        if not target or target.source!='uia':return False
        if not (intent.action=='click' and target.role in ('Button','Hyperlink','TabItem','ListItem','RadioButton') or intent.action=='type_text' and target.role=='Edit'):return False
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread():
            def match(control,depth):
                if control.IsPassword:return False
                r=control.BoundingRectangle
                return (Rect(r.left,r.top,r.right-r.left,r.bottom-r.top)==target.rect
                    and (control.Name[:180] or target.role)==target.label)
            control=getattr(auto,target.role+'Control')(searchFromControl=auto.ControlFromHandle(target.window),
                searchDepth=20,Compare=match)
            if not control.Exists(.4,.05) or control.IsPassword or not control.IsEnabled:
                raise RuntimeError('The selected element changed')
            selection=intent.action=='click' and target.role in ('TabItem','ListItem','RadioButton')
            pattern_id=auto.PatternId.SelectionItemPattern if selection else auto.PatternId.InvokePattern if intent.action=='click' else auto.PatternId.ValuePattern
            pattern=control.GetPattern(pattern_id)
            if pattern is None:return False  # Unsupported pattern, before any dispatch.
            def dispatchable():
                if self.win.at(*target.rect.center)!=target.window:raise RuntimeError('Another window covers the element')
                if not self.is_active():raise RuntimeError('Control stopped')
            dispatchable()
            if selection:accepted=pattern.Select(waitTime=0)
            elif intent.action=='click':accepted=pattern.Invoke(waitTime=0)
            else:
                if pattern.IsReadOnly:raise RuntimeError('This field is read-only')
                # ValuePattern cannot preserve a caret/selection. Use it only for an empty
                # named field; nonempty fields require native insertion, never silent replacement.
                if pattern.Value:return False
                dispatchable()
                accepted=pattern.SetValue(intent.text,waitTime=0)
            if accepted is False:raise RuntimeError('Windows didn’t confirm the action.')
            return True
    def validate_visual(self,target):
        if not target.visual_hash:return
        import mss
        r=target.rect
        with mss.mss() as capture:
            shot=capture.grab(dict(left=int(r.x),top=int(r.y),width=int(r.w),height=int(r.h)))
            digest=hashlib.blake2s(shot.rgb,digest_size=12).hexdigest()
        if digest!=target.visual_hash:raise RuntimeError('The element looks different. Wait for Mira to recognize it again.')
    def validate_control(self,target,focused=False):
        if target.source!='uia':return
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread():
            control=auto.GetFocusedControl() if focused else auto.ControlFromPoint(*map(int,target.rect.center))
            for _ in range(5):
                if not control:break
                if control.IsPassword:raise RuntimeError('This field is protected')
                r=control.BoundingRectangle;rect=Rect(r.left,r.top,r.right-r.left,r.bottom-r.top)
                role=control.ControlTypeName.replace('Control','')
                if role==target.role and rect==target.rect and (control.Name[:180] or role)==target.label:return
                if focused:break
                control=control.GetParentControl()
        raise RuntimeError('The element or focus changed. Select it again.')
    def execute(self,intent,target,scene):
        action = intent.action
        if action not in WINDOWS_ACTIONS:
            raise RuntimeError(f'Unsupported action ({action})')
        if action == 'scroll' and intent.direction not in SCROLL_DIRECTIONS:
            raise RuntimeError('Choose a scroll direction: up or down')
        if action in TARGET_ACTIONS and target is None:
            raise RuntimeError('Look at an element or say its name first')
        if action == 'type_text' and (not intent.text or len(intent.text) > 10000):
            raise RuntimeError('The text is empty or too long')
        now=time.monotonic()
        if not 0<=now-scene.captured<SCENE_TTL:raise RuntimeError('The screen changed. Wait for the target to update.')
        hwnd=target.window if target else scene.window
        if not hwnd or not self.win.u.IsWindow(hwnd):raise RuntimeError('Choose a window')
        window=scene.window_map(hwnd)
        if scene.windows:
            if window is None or not 0<=now-window.captured<MAP_TTL:raise RuntimeError('Updating the window map')
            if self.win.pid(hwnd)!=window.pid or self.win.rect(hwnd)!=window.rect:raise RuntimeError('The window moved or changed')
        if target and scene.current_target(target,now) is None:raise RuntimeError('The target is out of date')
        if target and self.win.pid(hwnd)!=target.pid:raise RuntimeError('The app restarted')
        if hwnd==scene.window and scene.window_rect and self.win.rect(hwnd)!=scene.window_rect:
            raise RuntimeError('The window moved. Look at the target again.')
        # Never fall back or repeat a semantic action that was already sent.
        if self.semantic(intent,target):
            return {'target':target.label,'route':'uia'}
        if not self.is_active():raise RuntimeError('Control stopped')
        if self.win.shift_down():raise RuntimeError('Release Shift and repeat the command. This target uses standard Windows input.')
        self.win.activate(hwnd)
        if action in TARGET_ACTIONS:
            x,y=target.rect.center
            if self.win.at(x,y)!=hwnd:raise RuntimeError('Another window covers the element')
            self.validate_visual(target)
            self.validate_control(target)
            if action=='scroll':
                sign=1 if intent.direction=='up' else -1
                self._send([self.win.move_event(x,y),self.win.mouse(0x800,sign*120*min(10,max(1,intent.amount)))])
            elif action=='move':self._send([self.win.move_event(x,y)])
            elif action in ('click', 'double_click', 'right_click'):
                flags=(8,16) if action=='right_click' else (2,4)
                events=[self.win.move_event(x,y),self.win.mouse(flags[0]),self.win.mouse(flags[1])]
                if action=='double_click':events.extend([self.win.mouse(flags[0]),self.win.mouse(flags[1])])
                self._send(events)
            else:
                raise RuntimeError(f'Action is unavailable ({action})')
        elif action=='type_text':
            # Only an edit field is focused first. Otherwise the text goes to the window's current focus.
            typed_into=target if target and target.role=='Edit' else None
            if typed_into:
                x,y=target.rect.center
                if self.win.at(x,y)!=hwnd:raise RuntimeError('The input field is covered')
                self.validate_visual(target)
                self.validate_control(target)
                self._send([self.win.move_event(x,y),self.win.mouse(2),self.win.mouse(4)])
                time.sleep(.04)
                if self.win.foreground()!=hwnd:raise RuntimeError('The field lost focus before Mira could type')
                self.validate_control(target,focused=True)
            data=intent.text.encode('utf-16-le');events=[]
            for i in range(0,len(data),2):
                char=int.from_bytes(data[i:i+2],'little')
                events.extend((self.win.key(scan=char,flags=4),self.win.key(scan=char,flags=6)))
            self._send(events)
        elif action=='press_keys':
            named={'Ctrl':17,'Alt':18,'Shift':16,'Win':91,'Enter':13,'Tab':9,'Escape':27,'Backspace':8,'Delete':46,'Space':32,'Left':37,'Up':38,'Right':39,'Down':40,'Home':36,'End':35,'Pageup':33,'Pagedown':34}
            keys=[]
            for key in intent.keys:
                normalized=key.title()
                vk=named.get(normalized)
                # Only ASCII letters/digits share their code with Windows virtual keys.
                # Punctuation needs layout-aware mapping; '.' is VK_DELETE, '[' is VK_LWIN.
                if vk is None and len(key)==1 and key.isascii() and key.isalnum():vk=ord(key.upper())
                if vk is None and normalized.startswith('F') and normalized[1:].isdigit() and 1<=int(normalized[1:])<=24:vk=111+int(normalized[1:])
                if vk is None:raise RuntimeError('Unknown key: '+key)
                keys.append(vk)
            if not keys or len(keys)>5:raise RuntimeError('Invalid keyboard shortcut')
            extended=lambda k:1 if k in (33,34,35,36,37,38,39,40,45,46,91) else 0
            self._send([self.win.key(vk=k,flags=extended(k)) for k in keys]+[self.win.key(vk=k,flags=2|extended(k)) for k in reversed(keys)])
        else:
            raise RuntimeError(f'Action is unavailable ({action})')
        if action=='type_text':target=typed_into
        return {'target':target.label if target else window.title if window else scene.title,'route':'native'}
