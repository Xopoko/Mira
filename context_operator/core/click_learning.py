"""Timestamped human click labels; no screen recognition is needed for supervision."""
from collections import deque
from dataclasses import dataclass
import math
from .contracts import GazeMeasurement


@dataclass(frozen=True)
class HumanClick:
    x: float
    y: float
    pressed: float
    released: float
    window: int = 0
    pid: int = 0


@dataclass(frozen=True)
class PointerMotion:
    x: float
    y: float
    captured: float
    window: int = 0
    pid: int = 0
    kind: str = 'move'

@dataclass(frozen=True)
class ClickMatch:
    sample: GazeMeasurement
    center: tuple[float, float]
    distance: float
    limit: float
    dispersion: float
    weight: float


class ClickHistory:
    MIN_SAMPLES=3
    MIN_SPAN=.05
    WINDOW=.45
    MAX_DISPERSION=70.

    def __init__(self):
        self.samples=deque(maxlen=160)
        self.last_rejection=''
        self.last_rejection_code=''

    def clear(self):
        self.samples.clear();self.last_rejection='';self.last_rejection_code=''

    def _reject(self,code,message):
        self.last_rejection_code=code;self.last_rejection=message
        return None

    def observe(self,sample,now,missing_at=None):
        if (not isinstance(sample,GazeMeasurement) or not 0<=now-sample.captured<.2
                or len(sample.features)!=22 or len(sample.point)!=2
                or not all(math.isfinite(v) for v in (*sample.features,*sample.point))):
            # A missing-eye observation is a barrier, not permission to reuse a fixation.
            stamp=missing_at if isinstance(missing_at,(int,float)) and 0<=now-missing_at<.2 else now
            self.samples.append((stamp,None));return
        if self.samples and self.samples[-1][0]>=sample.captured:return
        self.samples.append((sample.captured,sample))

    def match(self,click,now,bounds=None,radius=230.):
        self.last_rejection='';self.last_rejection_code=''
        if not (0<=now-click.released<.35 and 0<=click.released-click.pressed<=1.5):
            return self._reject('stale','the click arrived too late')
        eligible=[(stamp,sample) for stamp,sample in self.samples if stamp<=click.pressed]
        barriers=[stamp for stamp,sample in eligible if sample is None]
        barrier=max(barriers,default=float('-inf'))
        valid=[(stamp,sample) for stamp,sample in eligible
               if sample is not None and stamp>barrier and click.pressed-stamp<=self.WINDOW]
        if len(valid)<self.MIN_SAMPLES:
            return self._reject('missing','too few recent gaze readings before the click')
        if not all(bool(sample.supported) for _,sample in valid):
            return self._reject('unsupported','your gaze is outside the area you set up')
        # A short eye movement after a fixation is allowed. A second settled
        # fixation elsewhere supersedes it; missing eyes always form a barrier.
        runs=[];run=[]
        for item in valid:
            if run and (item[0]-run[-1][0]>.12 or math.dist(item[1].point,run[-1][1].point)>self.MAX_DISPERSION):
                runs.append(run);run=[]
            run.append(item)
        if run:runs.append(run)
        stable=None
        for candidate in reversed(runs):
            stamps=[stamp for stamp,_ in candidate]
            if (len(candidate)>=self.MIN_SAMPLES and len(set(stamps))==len(stamps)
                    and stamps[-1]-stamps[0]>=self.MIN_SPAN and click.pressed-stamps[-1]<=.30):
                stable=candidate;break
        if stable is None:
            return self._reject('unstable','your gaze wasn’t steady before the click')
        points=[(float(sample.point[0]),float(sample.point[1])) for _,sample in stable]
        center=(sorted(p[0] for p in points)[len(points)//2],sorted(p[1] for p in points)[len(points)//2])
        dispersion=max(math.dist(point,center) for point in points)
        if dispersion>self.MAX_DISPERSION:
            return self._reject('unstable','your gaze moved before the click')
        width=getattr(bounds,'w',0)
        height=getattr(bounds,'h',0)
        if not isinstance(width,(int,float)) or not isinstance(height,(int,float)) or width<=0 or height<=0:
            limit=200.
        else:
            radius=float(radius) if isinstance(radius,(int,float)) and math.isfinite(radius) and radius>0 else 230.
            limit=min(650.,.45*min(width,height),2*radius)
        distance=math.dist((click.x,click.y),center)
        if distance>limit:
            return self._reject('away','the click was far from your gaze. Run setup to improve accuracy.')
        tail=[sample.point for stamp,sample in valid if stamp>stable[-1][0]]
        if tail and distance>40:
            travel=(tail[-1][0]-center[0],tail[-1][1]-center[1])
            toward=(click.x-center[0],click.y-center[1])
            if math.hypot(*travel)>30 and travel[0]*toward[0]+travel[1]*toward[1]<-.2*math.hypot(*travel)*distance:
                return self._reject('away','your gaze moved away from the click')
        # Use the observed sample nearest the robust fixation center.
        sample=min(stable,key=lambda item:(math.dist(item[1].point,center),-item[0]))[1]
        weight=max(.15,1-.75*distance/limit)*max(.25,1-dispersion/100)
        return ClickMatch(sample,center,distance,limit,dispersion,weight)

    def before(self,click,now,bounds=None):
        match=self.match(click,now,bounds)
        return match.sample if match else None
