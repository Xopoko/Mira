"""One bounded inference worker; verifying unchanged windows never waits for it."""
import queue,threading,time
from .visual_regions import VisualFrame,digest,regions,unique_verified

class VisualCache:
    def __init__(self,stop,client_factory,unavailable=lambda reason:None):
        self.parent_stop=stop;self.closed=threading.Event();self.unavailable=unavailable;self.factory=client_factory
        self.jobs=queue.Queue(maxsize=1);self.results=queue.SimpleQueue()
        self.pending=set();self.cache={}
        self.worker=threading.Thread(target=self._run,daemon=True,name='visual-inference');self.worker.start()

    def is_set(self):return self.parent_stop.is_set() or self.closed.is_set()

    def _run(self):
        client=None
        try:
            client=self.factory()
            client.ready(self);failed=False
            while not self.is_set():
                try:key,image=self.jobs.get(timeout=.1)
                except queue.Empty:key=None
                if self.is_set():break
                child=client.child
                if not failed and child is not None and child.poll() is not None:
                    self.unavailable('The isolated perception worker exited.')
                    failed=True
                if key is None:continue
                try:
                    patches=regions(key,image,client.observe(image,self));failed=False
                except Exception as exc:
                    if not failed and not self.is_set():self.unavailable(str(exc) or type(exc).__name__)
                    failed=True;patches=None
                if not self.is_set():self.results.put((key,time.monotonic(),patches))
        except Exception as exc:
            if not self.is_set():self.unavailable(str(exc) or type(exc).__name__)
        finally:
            if client is not None:client.close()

    def lookup_frame(self,owner,rect,image):
        if self.is_set():return ()
        key=VisualFrame(owner,rect,digest(image))
        while True:
            try:completed,created,patches=self.results.get_nowait()
            except queue.Empty:break
            self.pending.discard(completed)
            if patches is not None:
                self.cache[completed]=(created,patches)
                if len(self.cache)>24:self.cache.pop(next(iter(self.cache)))
        # Pending includes finished results until they have been read.
        if (key not in self.cache and key not in self.pending and len(self.pending)<2
                and not self.is_set() and self.worker.is_alive()):
            try:self.jobs.put_nowait((key,image));self.pending.add(key)
            except queue.Full:pass
        now=time.monotonic()
        candidates=(region for frame,(created,patches) in reversed(self.cache.items())
                    if frame.owner==owner and (frame==key or now-created<=12) for region in patches)
        return unique_verified(candidates,image,rect)

    def close(self):
        self.closed.set();self.worker.join(timeout=.3);self.cache.clear();self.pending.clear()
