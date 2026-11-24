"""A live desktop map; slow applications do not block neighbouring windows."""
from dataclasses import replace
import queue,threading,time
import comtypes  # Complete its implicit COM initialization before starting reader threads.
from ..core.contracts import Scene,Rect
from ..core.desktop import compose,intersection,MAP_TTL
from .windows import Windows
from .uia_reader import WindowReader,protected

class ScreenProvider:
    def __init__(self,win=None,reader_factory=WindowReader,workers=3):
        self.win=win or Windows();self.sequence=0;self.practice_window=0;self.attention=None
        self.maps={};self.pending=set();self.requested={};self.vision_requested={};self.primary=0
        self.jobs=queue.Queue(maxsize=workers);self.results=queue.SimpleQueue()
        self.pending_limit=workers*2
        self.stopped=threading.Event();self.threads=[];self.reader_factory=reader_factory;self.last_error=''
        for n in range(workers):
            thread=threading.Thread(target=self._worker,name=f'window-map-{n}',daemon=True)
            self.threads.append(thread);thread.start()

    def _worker(self):
        # UIA desktop clients need MTA workers, not STA threads without a message pump.
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        reader=None
        try:
            while not self.stopped.is_set():
                try:surface=self.jobs.get(timeout=.1)
                except queue.Empty:continue
                if self.stopped.is_set():break
                try:
                    if reader is None:reader=self.reader_factory()
                    result=reader.scan(surface);error=''
                except Exception as exc:
                    result=None;error=type(exc).__name__;reader=None
                if not self.stopped.is_set():
                    self.results.put((surface,result,error))
        finally:
            reader=None  # Release COM objects in the apartment that owns them.
            comtypes.CoUninitialize()

    def close(self):
        self.stopped.set()
        for thread in self.threads:thread.join(timeout=.15)

    def observe(self):
        started=time.monotonic();self.sequence+=1
        surfaces=tuple(s for s in self.win.visible_windows(self.practice_window) if not protected(s.title));by_id={s.window:s for s in surfaces}
        foreground=self.win.foreground()
        self.primary=self.practice_window or (foreground if foreground in by_id else self.primary)
        if self.primary not in by_id:self.primary=surfaces[0].window if surfaces else 0
        while True:
            try:surface,result,error=self.results.get_nowait()
            except queue.Empty:break
            self.pending.discard(surface.window)
            current=by_id.get(surface.window)
            if current and (current.pid,current.rect)==(surface.pid,surface.rect):
                if result:self.maps[surface.window]=result;self.last_error=''
                elif error:
                    self.last_error=error
                    # Existing public outlines may live only until their original
                    # expiry. A failed password check cannot authorize new images.
                    cached=self.maps.get(surface.window)
                    if cached:self.maps[surface.window]=replace(cached,blocked_capture=True)
        for hwnd in list(self.maps):
            s=by_id.get(hwnd);m=self.maps[hwnd]
            if s is None or (s.pid,s.rect)!=(m.pid,m.rect) or started-m.captured>=MAP_TTL:self.maps.pop(hwnd,None)
        gaze=self.attention
        point=(gaze.x,gaze.y) if gaze and gaze.valid and 0<=started-gaze.captured<.3 else self.win.cursor()
        def distance(s):return min(r.distance(*point) for r in s.visible)
        # Warm every visible window; near the halo refresh more often. Oldest due work wins.
        due=[s for s in surfaces if s.window not in self.pending
             and started-self.requested.get(s.window,0)>=(.25 if distance(s)<420 else .65)]
        due.sort(key=lambda s:(s.window in self.maps, self.requested.get(s.window,0),distance(s)))
        for surface in due:
            # Count queued, running and finished scans until their results are read.
            if len(self.pending)>=self.pending_limit:break
            try:self.jobs.put_nowait(surface)
            except queue.Full:break
            self.pending.add(surface.window);self.requested[surface.window]=started
        self.requested={h:t for h,t in self.requested.items() if h in by_id or h in self.pending}
        return compose(self.sequence,started,self.win.bounds(),surfaces,self.maps,self.primary,(time.monotonic()-started)*1000)

    def vision_scene(self,scene,gaze,now):
        if not scene.windows:return scene
        valid=gaze and gaze.valid and 0<=now-gaze.captured<.3
        nearby=[w for w in scene.windows if not w.blocked_capture and (not valid or any(r.distance(gaze.x,gaze.y)<600 for r in w.visible))]
        if not nearby:return None
        window=min(nearby,key=lambda w:self.vision_requested.get(w.window,0))
        self.vision_requested[window.window]=now
        return Scene(scene.id,window.captured,scene.bounds,window.elements,window.window,window.title,0,window.rect,(window,))

    def capture(self,scene,region=None):
        window=scene.window_map(scene.window)
        if not window or window.blocked_capture or not scene.window_rect:return None
        if protected(self.win.title(window.window)):return None
        if self.win.pid(window.window)!=window.pid or self.win.rect(window.window)!=window.rect:return None
        parts=[intersection(r,region or scene.bounds) for r in window.visible]
        parts=[r for r in parts if r and r.w>0 and r.h>0]
        if not parts:return None
        rect=max(parts,key=lambda r:r.w*r.h)
        rect=Rect(int(rect.x),int(rect.y),int(rect.w),int(rect.h))
        if not self.win.unobscured(window.window,rect):return None
        from PIL import Image
        import mss
        with mss.mss() as capture:
            shot=capture.grab(dict(left=int(rect.x),top=int(rect.y),width=int(rect.w),height=int(rect.h)))
            if not self.win.unobscured(window.window,rect) or self.win.rect(window.window)!=window.rect:return None
            return Image.frombytes('RGB',shot.size,shot.rgb),rect
