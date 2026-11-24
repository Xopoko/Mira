"""Choose a fresh screen element from gaze, text or an explicit selection."""
import math
import re
from .contracts import Focus, RankedTarget, SCENE_TTL

def words(text): return set(re.findall(r'[\w]+', text.casefold()))

class Fusion:
    def __init__(self, stability=.10):
        self.stability=stability
        self.current=None
        self.pending=None
        self.pending_since=0.
        self._selection=None
        # Magnetic focus is deliberately separate from the legacy hysteresis.
        # It remembers the selected element's full identity and the gaze frame
        # at which that selection was acquired.
        self._magnetic_identity=None
        self._magnetic_anchor=None
        self._magnetic_pending=None
        self._magnetic_since=0.
        self._magnetic_last_frame=None

    def reset(self):
        self.current=self.pending=self._selection=None
        self._clear_magnetic()

    def _clear_magnetic(self):
        self._magnetic_identity=None
        self._magnetic_anchor=None
        self._magnetic_pending=None
        self._magnetic_since=0.
        self._magnetic_last_frame=None

    def select(self, element, now, duration, *, requires_gaze=False):
        self.reset()
        self._selection=(element.identity,now+duration,requires_gaze)

    def selection_until(self, element, now, gaze=None):
        if element is None or not element.enabled or self._selection is None:
            return 0.
        identity, until, requires_gaze = self._selection
        if now >= until or element.identity != identity:
            return 0.
        if not requires_gaze:
            return until
        if gaze is not None and gaze.valid and 0 <= now-gaze.captured < .25:
            return min(until,gaze.captured+.25)
        return 0.

    def _magnetic_select(self, ranked, best, gaze):
        """Select with a short, gaze-driven dwell when focus is magnetic."""
        by_id={r.element.id:r for r in ranked}

        current=by_id.get(self.current)
        # Do not carry a lock when an element was removed, hidden, disabled,
        # went stale, or was replaced in-place with changed identity/geometry.
        if (current is None or self._magnetic_identity is None or
                current.element.identity!=self._magnetic_identity):
            self._clear_magnetic()
            self.current=best
            self._magnetic_identity=by_id[best].element.identity
            self._magnetic_anchor=(gaze.x,gaze.y)
            self._magnetic_last_frame=gaze.captured
            return best

        if best==self.current:
            self._magnetic_pending=None
            # Repeated frames do not move the anchor or advance a dwell.
            if self._magnetic_last_frame is None or gaze.captured>self._magnetic_last_frame:
                self._magnetic_last_frame=gaze.captured
            return self.current

        locked=current
        challenger=by_id[best]
        radius=max(50.,min(gaze.radius,800.)) if math.isfinite(gaze.radius) else 50.
        improvement=locked.distance-challenger.distance
        score_gain=challenger.score-locked.score
        # Edge distance is equal for a pane and a button inside it. A smaller
        # actionable child under the gaze is still better context, but require
        # the same dwell so overlapping hierarchy does not flicker.
        locked_area=locked.element.rect.w*locked.element.rect.h
        challenger_area=challenger.element.rect.w*challenger.element.rect.h
        nested_better_context=(locked.element.rect.contains(gaze.x,gaze.y) and
                               challenger.element.rect.contains(gaze.x,gaze.y) and
                               challenger_area<locked_area*.5 and score_gain>0.)
        # A clearly new/far gaze point should respond immediately. Modest
        # advantage between neighboring controls must survive a short dwell.
        if (score_gain>=.85 or
                (improvement>=max(80.,min(radius*.45,120.)) and
                 challenger.distance<=max(20.,min(radius*.1,60.)))):
            self.current=best
            self._magnetic_identity=challenger.element.identity
            self._magnetic_anchor=(gaze.x,gaze.y)
            self._magnetic_pending=None
            self._magnetic_last_frame=gaze.captured
            return best

        # Require a new timestamp and a sustained shift from the point where
        # the current target was acquired, toward the challenger.
        shifted=False
        if self._magnetic_anchor is not None:
            ax,ay=self._magnetic_anchor
            gx,gy=gaze.x,gaze.y
            cx,cy=challenger.element.rect.center
            vx,vy=cx-ax,cy-ay
            length=math.hypot(vx,vy)
            moved=math.hypot(gx-ax,gy-ay)
            projection=((gx-ax)*vx+(gy-ay)*vy)/length if length>0 else 0.
            threshold=max(18.,min(radius*.10,60.))
            shifted=(moved>=threshold and projection>=threshold and
                     projection/max(moved,1e-9)>=.55 and
                     (challenger.distance<locked.distance or nested_better_context))
        if nested_better_context:
            shifted=True

        fresh_frame=(self._magnetic_last_frame is None or gaze.captured>self._magnetic_last_frame)
        if fresh_frame:
            self._magnetic_last_frame=gaze.captured
            if shifted:
                if self._magnetic_pending!=best:
                    self._magnetic_pending=best
                    self._magnetic_since=gaze.captured
                elif gaze.captured-self._magnetic_since>=.18:
                    self.current=best
                    self._magnetic_identity=challenger.element.identity
                    self._magnetic_anchor=(gaze.x,gaze.y)
                    self._magnetic_pending=None
                    return best
            else:
                self._magnetic_pending=None
        elif not shifted:
            self._magnetic_pending=None
        return self.current

    def choose_pointer(self,scene,pointer,now,window=None):
        """A precise pointer selects what it touches, without gaze hysteresis."""
        self.reset()
        if scene is None or not 0<=now-scene.captured<SCENE_TTL:
            return Focus(reason='Updating the screen map')
        if pointer is None or not pointer.valid or not 0<=now-pointer.captured<.25:
            return Focus(reason='No mouse pointer')
        candidates=[e for e in scene.current_elements(now) if e.enabled and e.rect.w>0 and e.rect.h>0
                    and (window is None or e.window==window) and e.rect.contains(pointer.x,pointer.y)]
        candidates.sort(key=lambda e:(e.rect.w*e.rect.h,e.source!='uia'))
        if not candidates:return Focus(reason='Move the mouse over an element')
        self.current=candidates[0].id
        return Focus(candidates[0],tuple(RankedTarget(e,1.,0.,'Selected with the mouse') for e in candidates[:8]),
                     False,'Selected with the mouse')

    def choose(self, scene, gaze, now, query='', *, magnetic=False):
        if scene is None or not 0<=now-scene.captured<SCENE_TTL:
            self.reset()
            return Focus(reason='Updating the screen map')
        valid_gaze=gaze is not None and gaze.valid and 0 <= now-gaze.captured < .25
        if magnetic and gaze is not None and not valid_gaze:
            self.reset()
        if magnetic and valid_gaze and any(not math.isfinite(v) for v in (gaze.x,gaze.y,gaze.radius)):
            self.reset()
            return Focus(reason='Waiting for fresh gaze data')
        elements=scene.current_elements(now)
        selected=None
        if self._selection is not None:
            selected=next((e for e in elements if e.rect.w>0 and e.rect.h>0
                           and self.selection_until(e,now,gaze)),None)
            if selected is None:
                self.reset()
        if not valid_gaze and not query and selected is None:
            self.reset()
            return Focus(reason='Waiting for fresh gaze data')
        ranked=[]
        requested=words(query)
        for el in elements:
            if not el.enabled or el.rect.w <= 0 or el.rect.h <= 0: continue
            explicit=el is selected
            if not valid_gaze and not query and not explicit: continue
            radius=(max(50.,min(gaze.radius,800.)) if magnetic and valid_gaze else
                    max(50.,gaze.radius) if valid_gaze else 230.)
            distance=el.rect.distance(gaze.x,gaze.y) if valid_gaze else 0.
            if valid_gaze and not query and not explicit and distance > radius*2.4: continue
            spatial=math.exp(-.5*(distance/radius)**2) if valid_gaze else 0.
            # Containing a huge pane is not stronger evidence than a nearby button.
            size=math.log1p(el.rect.w*el.rect.h/12000)
            match=len(requested & words(el.label))/max(1,len(requested))
            center=math.dist(el.rect.center,(gaze.x,gaze.y))/radius if valid_gaze else 0.
            score=spatial*2.2-min(center,5)*.08-min(size,.9)*.22+match*2.6
            if el.source=='uia': score+=.08
            if el.id==self.current and not magnetic: score+=.13
            reason='Matches your command and gaze' if query else 'Near where you’re looking steadily'
            if explicit:reason='Selected by name'
            ranked.append(RankedTarget(el,score,distance,reason))
        ranked.sort(key=lambda item:item.score,reverse=True)
        if not ranked:
            if magnetic:self.reset()
            return Focus(reason='Mira couldn’t find an element near your gaze')
        if selected is not None:
            target=next(r for r in ranked if r.element is selected)
        else:
            best=ranked[0].element.id
            if magnetic and valid_gaze:
                best=self._magnetic_select(ranked,best,gaze)
            elif best!=self.current and any(r.element.id==self.current for r in ranked):
                if best!=self.pending: self.pending,self.pending_since=best,now
                if now-self.pending_since<self.stability: best=self.current
            else: self.pending=None
            target=next(r for r in ranked if r.element.id==best)
        self.current=target.element.id
        alternatives=ranked[:8]
        if selected is not None and target not in alternatives:
            alternatives[-1]=target
        ambiguous=selected is None and len(ranked)>1 and ranked[0].score-ranked[1].score<.18
        return Focus(target.element,tuple(alternatives),ambiguous,target.reason)

    def cycle(self, focus, now, direction='next'):
        if not focus.target: return None
        candidates=[r.element for r in focus.alternatives]
        if direction in ('left','right','up','down'):
            x,y=focus.target.rect.center
            def allowed(e):
                ex,ey=e.rect.center
                return {'left':ex<x-8,'right':ex>x+8,'up':ey<y-8,'down':ey>y+8}[direction]
            candidates=[e for e in candidates if allowed(e)]
            if not candidates:return None
            chosen=min(candidates,key=lambda e:math.dist(e.rect.center,(x,y)))
        else:
            if len(candidates)<2:return None
            index=next((i for i,e in enumerate(candidates) if e.id==focus.target.id),-1)
            chosen=candidates[(index+1)%len(candidates)]
        self.select(chosen,now,2.,requires_gaze=True)
        return chosen
