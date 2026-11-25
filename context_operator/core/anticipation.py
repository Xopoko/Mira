"""Causal, personalized next-target ranking. No Qt, device or file dependencies."""
from collections import deque
from dataclasses import dataclass
import copy
import math
import numpy as np
from .contracts import Element, GAZE_TTL

PRIOR=np.array([3.,2.,.6,1.8,3.,1.2,.35,-.3,-2.,.7,.4,.3,.1,.1,.1,0.,0.,0.])
GAZE_PRIOR=np.array([3.8,2.6,.4,.55,.35,.2,.15,.1,0.])


@dataclass(frozen=True)
class Anticipation:
    target: Element
    captured: float
    motion_at: float
    strength: float
    source: str = 'mouse'

    def __post_init__(self):
        if self.source not in ('mouse','gaze'):
            raise ValueError('Unknown anticipation source')

    @property
    def valid_until(self):
        return self.motion_at+(.18 if self.source=='mouse' else .32)

    def is_fresh(self,now):
        return (math.isfinite(now) and math.isfinite(self.captured) and math.isfinite(self.motion_at)
                and math.isfinite(self.valid_until) and self.captured<=now and self.motion_at<=now<self.valid_until)


def softmax(scores):
    values=np.exp(scores-np.max(scores));return values/values.sum()


class _Ranker:
    """Listwise logistic ranking, adaptive gradients and bounded recent replay."""
    prior: np.ndarray

    def __init__(self):
        self.weights=self.prior.copy();self.gradient=np.full_like(self.prior,.1)
        self.replay=[];self.learned=0;self.evaluated=0;self.hits=0;self.unmapped=0
        self.shown=0;self.shown_hits=0;self.prior_hits=0;self.last_role=''

    def scores(self,x):return np.asarray(x)@self.weights

    def learn(self,x,y):
        x=np.asarray(x,float)
        # New example gets most weight; two recent episodes reduce single-click drift.
        batch=[(x,y,1.)]+[(np.asarray(r['x']),r['y'],.2) for r in self.replay[-2:]]
        for _ in range(3):
            grad=np.zeros_like(self.prior)
            for rows,label,weight in batch:grad+=weight*(softmax(self.scores(rows))@rows-rows[label])
            grad+=.025*(self.weights-self.prior)
            self.gradient=.995*self.gradient+grad*grad
            self.weights=np.clip(self.weights-.12*grad/(np.sqrt(self.gradient)+.05),-8,8)
        self.replay=(self.replay+[dict(x=np.round(x,5).tolist(),y=int(y))])[-80:]
        self.learned+=1

    def state(self):
        return dict(schema=1,weights=self.weights.tolist(),gradient=self.gradient.tolist(),replay=copy.deepcopy(self.replay),
                    learned=self.learned,evaluated=self.evaluated,hits=self.hits,unmapped=self.unmapped,
                    shown=self.shown,shown_hits=self.shown_hits,prior_hits=self.prior_hits,last_role=self.last_role)

    @classmethod
    def restore(cls,data):
        model=cls()
        if data.get('schema')!=1:raise ValueError('Unknown intention model')
        for name in ('weights','gradient'):
            values=np.asarray(data[name],float)
            if values.shape!=model.prior.shape or not np.isfinite(values).all():raise ValueError('Invalid intention weights')
            if name=='gradient' and np.any(values<0):raise ValueError('Invalid adaptation state')
            setattr(model,name,values)
        replay=data['replay']
        if len(replay)>80:raise ValueError('Too many replay examples')
        for r in replay:
            x=np.asarray(r['x'])
            if x.ndim!=2 or x.shape[1]!=model.prior.size or not 2<=len(x)<=48 or not np.isfinite(x).all() or not 0<=r['y']<len(x):
                raise ValueError('Invalid replay example')
        model.replay=copy.deepcopy(replay)
        for key in ('learned','evaluated','hits','unmapped','shown','shown_hits','prior_hits'):
            value=data[key]
            if not isinstance(value,int) or value<0:raise ValueError('Invalid intention statistics')
            setattr(model,key,value)
        model.last_role=str(data.get('last_role',''))
        return model


class OnlineRanker(_Ranker):
    prior=PRIOR

    def __init__(self):
        super().__init__()
        self.lead_sum_ms=0.

    def state(self):
        return dict(super().state(),lead_sum_ms=self.lead_sum_ms)

    @classmethod
    def restore(cls,data):
        model=super().restore(data)
        model.lead_sum_ms=float(data['lead_sum_ms'])
        if not math.isfinite(model.lead_sum_ms) or model.lead_sum_ms<0:raise ValueError('Invalid lead time')
        return model


class GazeRanker(_Ranker):
    prior=GAZE_PRIOR


class IntentPredictor:
    GAZE_DWELL=.16
    GAZE_SPREAD=58.

    def __init__(self,model=None,gaze_model=None,own_pid=None):
        self.model=model or OnlineRanker();self.gaze_model=gaze_model or GazeRanker();self.own_pid=own_pid;self.reset()

    def reset(self):
        self.trail=deque(maxlen=48);self.gaze_trail=deque(maxlen=48);self.snapshots=deque(maxlen=100)
        self.scenes=deque(maxlen=96)
        self.current=None;self.pending=None;self.since=0.;self.last_snapshot=0.
        self.gaze_pending=None;self.gaze_since=0.;self.mouse_blocked=False
        self.gaze_blocked=False;self.gaze_min_after=0.

    def observe_scene(self,scene,now):
        # Arrival time matters: a post-press observation cannot label that press.
        if self.scenes and (now<self.scenes[-1][0] or scene is self.scenes[-1][1]):return
        self.scenes.append((now,scene))

    @staticmethod
    def eligible(element,scene,owner):
        r=element.rect;b=scene.bounds
        return (element.enabled and (element.window,element.pid)==owner and r.w>0 and r.h>0
                and all(math.isfinite(v) for v in r.as_list())
                and r.w*r.h<=b.w*b.h*.06 and r.h<=b.h*.5)

    @staticmethod
    def visible(element,scene,point=None):
        if not scene.windows:return True
        window=scene.window_map(element.window)
        if not window or window.pid!=element.pid or window.blocked_capture:return False
        if not any(r.contains(*element.rect.center) for r in window.visible):return False
        return point is None or any(r.contains(*point) for r in window.visible)

    def observe(self,motion):
        if motion.kind=='release':
            self.trail.clear();self.current=None;self.pending=None;self.mouse_blocked=False
            self.gaze_trail.clear();self.gaze_pending=None;self.gaze_blocked=False
            self.gaze_min_after=max(self.gaze_min_after,motion.captured)
            return
        if motion.kind!='move':
            self.trail.clear();self.current=None;self.pending=None
            self.mouse_blocked=True
            self.gaze_trail.clear();self.gaze_pending=None;self.gaze_blocked=True
            self.gaze_min_after=max(self.gaze_min_after,motion.captured)
            if motion.kind not in ('press','held'):
                self.snapshots.clear()
            return
        if not all(math.isfinite(v) for v in (motion.x,motion.y,motion.captured)):return
        if self.mouse_blocked:
            self.trail.clear();self.current=None;self.pending=None;self.mouse_blocked=False
        if self.gaze_blocked:
            self.gaze_trail.clear();self.gaze_pending=None;self.gaze_blocked=False
            self.gaze_min_after=max(self.gaze_min_after,motion.captured)
        if self.trail:
            last=self.trail[-1]
            if motion.captured<=last.captured:return
            if motion.captured-last.captured>.35 or (motion.window,motion.pid)!=(last.window,last.pid):
                self.trail.clear();self.current=None;self.pending=None
        self.trail.append(motion)
        while self.trail and motion.captured-self.trail[0].captured>.3:self.trail.popleft()

    def _gaze_target(self,scene,gaze,now):
        if (gaze is None or gaze.source=='mouse' or
                not 0<=now-gaze.captured<GAZE_TTL):return None
        if scene is None or not 0<=now-scene.captured<.8:return None
        candidates=[]
        for e in scene.current_elements(now,.8):
            if self.own_pid is not None and e.pid==self.own_pid:continue
            owner=(e.window,e.pid)
            if not self.eligible(e,scene,owner):continue
            if not self.visible(e,scene,(gaze.x,gaze.y)):continue
            if e.rect.distance(gaze.x,gaze.y)<=max(32.,min(90.,gaze.radius*.35)):
                candidates.append(e)
        if not candidates:return None
        return min(candidates,key=lambda e:(e.rect.distance(gaze.x,gaze.y),e.rect.w*e.rect.h,e.source!='uia'))

    def _observe_gaze(self,scene,gaze,now,supported):
        target=self._gaze_target(scene,gaze,now) if supported and not self.gaze_blocked else None
        if target is None:
            self.gaze_trail.clear();self.gaze_pending=None;return None
        if gaze.captured<=self.gaze_min_after:return None
        if self.gaze_trail:
            last=self.gaze_trail[-1]
            if gaze.captured==last['at']:
                if target.identity==last['identity']:return self._stable_gaze(now)
                self.gaze_trail.clear();self.gaze_pending=None;return None
            if gaze.captured<last['at']:
                self.gaze_trail.clear();self.gaze_pending=None;return None
            if gaze.captured-last['at']>.16 or target.identity!=last['identity']:
                self.gaze_trail.clear();self.gaze_pending=None
        self.gaze_trail.append(dict(at=gaze.captured,x=gaze.x,y=gaze.y,radius=gaze.radius,
                                    identity=target.identity,target=target))
        while self.gaze_trail and gaze.captured-self.gaze_trail[0]['at']>.32:self.gaze_trail.popleft()
        if self.gaze_trail and math.dist((gaze.x,gaze.y),(self.gaze_trail[0]['x'],self.gaze_trail[0]['y']))>self.GAZE_SPREAD:
            latest=self.gaze_trail[-1];self.gaze_trail.clear();self.gaze_trail.append(latest);self.gaze_pending=None
        return self._stable_gaze(now)

    def _stable_gaze(self,now):
        if len(self.gaze_trail)<3:return None
        first,last=self.gaze_trail[0],self.gaze_trail[-1]
        if not 0<=now-last['at']<GAZE_TTL or last['at']-first['at']<self.GAZE_DWELL:return None
        return last

    def _gaze_features(self,e,gaze,scene,role):
        b=scene.bounds;r=e.rect
        scale=max(80.,math.hypot(b.w,b.h)*.045)
        distance=r.distance(gaze.x,gaze.y)
        center=math.dist(r.center,(gaze.x,gaze.y))
        return [ -math.log1p(distance/scale), float(r.contains(gaze.x,gaze.y)),
                 -math.log1p(center/scale), float(e.role==role),
                 math.log1p(r.w*r.h/(scale*scale)), float(e.role=='Button'),
                 float(e.role=='TabItem'),float(e.role=='Hyperlink'),float(e.role=='Edit') ]

    def _predict_gaze(self,scene,gaze,now,fixation):
        if fixation is None:return None
        point=(fixation['x'],fixation['y']);anchor=fixation['target'];owner=(anchor.window,anchor.pid)
        elements=[];features=[];seen=set()
        for e in sorted(scene.current_elements(now,.8),key=lambda e:e.source!='uia'):
            if self.own_pid is not None and e.pid==self.own_pid:continue
            if not self.eligible(e,scene,owner):continue
            if not self.visible(e,scene):continue
            identity=(e.window,e.pid,e.id,e.rect)
            if identity in seen:continue
            seen.add(identity);f=self._gaze_features(e,gaze,scene,self.gaze_model.last_role)
            if np.isfinite(f).all():elements.append(e);features.append(f)
        if len(elements)<2:return None
        x=np.asarray(features);ids=np.argsort(x@GAZE_PRIOR)[-48:]
        x=x[ids];elements=tuple(elements[i] for i in ids)
        scores=self.gaze_model.scores(x);order=np.argsort(scores);best=int(order[-1]);target=elements[best]
        margin=scores[best]-scores[order[-2]];shape=softmax(scores)
        identity=target.identity
        if identity!=self.gaze_pending:self.gaze_pending=identity;self.gaze_since=now
        shown=bool(shape[best]>=.3 and margin>=.30 and now-self.gaze_since>=.06)
        if shown:self.current=Anticipation(target,now,fixation['at'],float(min(1.,.45+margin*.2)),source='gaze')
        if now-self.last_snapshot>=.045:
            self.snapshots.append(dict(channel='gaze',at=now,elements=elements,x=x.copy(),best=best,
                                       prior=int(np.argmax(x@GAZE_PRIOR)),shown=shown,point=point,
                                       radius=fixation['radius']))
            self.last_snapshot=now
        return self.current

    def predict(self,scene,gaze,now,*,gaze_supported=True):
        self.observe_scene(scene,now)
        self.current=None
        fixation=self._observe_gaze(scene,gaze,now,gaze_supported)
        if scene is None or not 0<=now-scene.captured<.8:return None
        if self.mouse_blocked or len(self.trail)<3:return self._predict_gaze(scene,gaze,now,fixation)
        last=self.trail[-1];first=self.trail[0];dt=last.captured-first.captured
        if not 0<=now-last.captured<.18 or dt<.025:return self._predict_gaze(scene,gaze,now,fixation)
        delta=np.array([last.x-first.x,last.y-first.y]);travel=float(np.linalg.norm(delta))
        if travel<8 or travel/dt<35:return self._predict_gaze(scene,gaze,now,fixation)
        unit=delta/travel;speed=min(6000.,travel/dt)
        point=np.array([last.x,last.y]);end=point+unit*speed*.24
        bounds=scene.bounds;scale=max(80.,math.hypot(bounds.w,bounds.h)*.045)
        valid_gaze=(gaze is not None and gaze_supported and gaze.source!='mouse'
                    and 0<=now-gaze.captured<GAZE_TTL)
        turn=0.
        if len(self.trail)>5:
            mid=self.trail[len(self.trail)//2];tail=point-[mid.x,mid.y];norm=np.linalg.norm(tail)
            if norm>1:turn=1-float(np.dot(unit,tail/norm))
        elements=[];features=[];seen=set()
        for e in sorted(scene.current_elements(now,.8),key=lambda e:e.source!='uia'):
            r=e.rect
            if not self.eligible(e,scene,(last.window,last.pid)):continue
            if not self.visible(e,scene,tuple(point)):continue
            identity=(e.window,e.pid,r)
            if identity in seen:continue
            seen.add(identity)
            vector=np.array(r.center)-point;distance=max(1.,float(np.linalg.norm(vector)))
            cosine=float(np.dot(vector,unit)/distance);cross=abs(vector[0]*unit[1]-vector[1]*unit[0])/distance
            g=math.exp(-.5*(r.distance(gaze.x,gaze.y)/max(60.,gaze.radius))**2) if valid_gaze else 0.
            closing=(math.dist(r.center,(first.x,first.y))-distance)/max(travel,1.)
            f=[cosine,-cross,-math.log1p(r.distance(*point)/scale),-math.log1p(r.distance(*end)/scale),
               g,g*cosine,float(e.role==self.model.last_role),math.log1p(r.w*r.h/(scale*scale)),float(r.contains(*point)),
               float(np.clip(closing,-1,1)),min(speed/1500,2)*cosine,turn*(g-cosine),
               float(e.role=='Button'),float(e.role=='TabItem'),float(e.role=='Hyperlink'),float(e.role=='Edit'),
               (r.center[0]-bounds.x)/bounds.w,(r.center[1]-bounds.y)/bounds.h]
            if np.isfinite(f).all():elements.append(e);features.append(f)
        if len(elements)<2:return self._predict_gaze(scene,gaze,now,fixation)
        x=np.asarray(features);ids=np.argsort(x@PRIOR)[-48:]
        x=x[ids];elements=tuple(elements[i] for i in ids)
        scores=self.model.scores(x);order=np.argsort(scores);best=int(order[-1]);prob=softmax(scores)
        target=elements[best];margin=scores[best]-scores[order[-2]]
        identity=(target.window,target.pid,target.id,target.rect)
        if identity!=self.pending:self.pending=identity;self.since=now
        shown=bool(prob[best]>=.3 and margin>=.35 and now-self.since>=.035
                   and not target.rect.contains(*point))
        if shown:self.current=Anticipation(target,now,last.captured,float(min(1.,.45+margin*.2)))
        if now-self.last_snapshot>=.045:
            self.snapshots.append(dict(channel='mouse',at=now,elements=elements,x=x.copy(),best=best,prior=int(np.argmax(x@PRIOR)),shown=shown,pointer=tuple(point)))
            self.last_snapshot=now
        return self.current or self._predict_gaze(scene,gaze,now,fixation)

    def feedback(self,click,now):
        """Evaluate the old forecast first. Only then consume the click as a label."""
        if not click.is_fresh(now):return False
        candidates=[s for s in self.snapshots if 0<click.pressed-s['at']<=1.25
                    and (s['elements'][0].window,s['elements'][0].pid)==(click.window,click.pid)]
        self.snapshots.clear();self.current=None
        self.gaze_trail.clear();self.gaze_pending=None;self.mouse_blocked=True
        self.gaze_blocked=False;self.gaze_min_after=max(self.gaze_min_after,click.released)
        if not candidates:return False
        before=next((scene for at,scene in reversed(self.scenes) if at<=click.pressed),None)
        if before is None or not 0<=click.pressed-before.captured<.8:return False
        matches=[e for e in before.current_elements(click.pressed,.8)
                 if self.eligible(e,before,(click.window,click.pid)) and self.visible(e,before,(click.x,click.y))
                 and e.rect.contains(click.x,click.y)]
        target=min(matches,key=lambda e:(e.rect.w*e.rect.h,e.source!='uia')) if matches else None
        if target is not None and self.own_pid is not None and target.pid==self.own_pid:target=None

        def label_for(snapshot):
            if target is None:return None
            return next((i for i,e in enumerate(snapshot['elements']) if e.identity==target.identity),None)

        def train_channel(channel,model,snapshots):
            channel_rows=[s for s in snapshots if s['channel']==channel]
            if not channel_rows:return False
            early_rows=[s for s in channel_rows if click.pressed-s['at']>=.15]
            early=early_rows[-1] if early_rows else None
            changed=False
            if early is not None and (channel=='gaze' or target is None or not target.rect.contains(*early['pointer'])):
                label=label_for(early);hit=label is not None and early['best']==label
                model.evaluated+=1;model.hits+=int(hit)
                model.prior_hits+=int(label is not None and early['prior']==label)
                model.shown+=int(early['shown']);model.shown_hits+=int(hit and early['shown'])
                model.unmapped+=int(label is None)
                if channel=='mouse':model.lead_sum_ms+=(click.pressed-early['at'])*1000
                changed=True
            train_rows=channel_rows
            if channel=='gaze':
                # Keep disagreement in metrics, but do not teach gaze intent from an away click.
                train_rows=[s for s in channel_rows
                            if math.dist((click.x,click.y),s['point'])<=max(80.,min(260.,s['radius']*1.2))]
            train_early=early if early in train_rows else None
            # Prefer early data; short mouse movements still train on pre-press snapshots.
            train=([train_early] if train_early is not None else [])+list(reversed(train_rows))
            for snapshot in train:
                label=label_for(snapshot)
                if label is None:continue
                if channel=='mouse' and target is not None and target.rect.contains(*snapshot['pointer']):continue
                model.learn(snapshot['x'],label);model.last_role=target.role
                return True
            return changed

        changed=train_channel('mouse',self.model,candidates)
        return train_channel('gaze',self.gaze_model,candidates) or changed

    def state(self):
        return dict(schema=2,mouse=self.model.state(),gaze=self.gaze_model.state())
