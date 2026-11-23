"""Personal gaze refinement from corroborated clicks and saved corrections."""
from dataclasses import replace
import copy,json,math,os,time,shutil
from pathlib import Path
import numpy as np
from ..core.contracts import Gaze
from ..foundations.gaze.smoothing import Smoother
from .personal_model import PersonalTrainer,adaptive_prediction,_soft_prediction

class GazeLearning:
    def __init__(self,path):
        self.path=Path(path);self.key=None;self.bounds=None
        self.anchors=[];self.pose_anchors=[];self.click_anchors=[];self.feature_scale=np.ones(22)
        self.pose_smoother=Smoother(.065)
        self.count=0;self.click_accepted=0;self.legacy_quarantined=0;self.last_click_result=''
        self.undo=[];self.rows=[];self.trainer=PersonalTrainer()
        self._training_save_pending=False;self._training_retry_at=0.
    def bind(self,key,bounds,feature_scale=None,*,read_only=False):
        self.key=key;self.bounds=bounds;self.anchors=[];self.pose_anchors=[];self.count=0
        self.click_accepted=0;self.legacy_quarantined=0;self.last_click_result='';self.undo=[]
        self._training_save_pending=False;self._training_retry_at=0.
        self.rows=[];self.click_anchors=[];self.trainer.invalidate()
        self.pose_smoother=Smoother(.065,scale=bounds.h)
        self.feature_scale=np.maximum(np.asarray(feature_scale if feature_scale is not None else np.ones(22),float),.005)
        if self.feature_scale.shape!=(22,) or not np.isfinite(self.feature_scale).all():raise ValueError('Invalid gaze feature scale')
        migration_needed=False
        try:
            data=json.loads(self.path.read_text(encoding='utf-8'))
            if data['profile']!=key or data['display']!=bounds.as_list():return
            anchors=data['anchors']
            if len(anchors)>80 or any(len(a)!=5 or not all(math.isfinite(float(v)) for v in a) or a[4]<1 or abs(a[2])>bounds.w or abs(a[3])>bounds.h for a in anchors):return
            poses=data.get('pose_anchors',[])
            if len(poses)>600:return
            for anchor in poses:
                f=np.asarray(anchor['features']);d=np.asarray(anchor['offset'])
                if f.shape!=(22,) or d.shape!=(2,) or not np.isfinite(np.r_[f,d]).all() or anchor['count']<1:return
            scale=np.asarray(data.get('feature_scale',self.feature_scale),float)
            if scale.shape!=(22,) or not np.isfinite(scale).all() or np.any(scale<=0):return
            if feature_scale is not None and poses and not np.allclose(scale,self.feature_scale):return
            self.anchors=anchors;self.pose_anchors=poses;self.feature_scale=scale;self.count=int(data['count'])
            rows=data.get('mouse_samples',[])
            if len(rows)>1200:raise ValueError('Too many personal samples')
            for row in rows:
                if any(np.asarray(row[k]).shape!=shape or not np.isfinite(row[k]).all()
                       for k,shape in [('features',(22,)),('target',(2,)),('baseline',(2,)),('base',(2,))]):
                    raise ValueError('Invalid mouse sample')
            # Discard obsolete RGB experiment records while retaining all
            # numeric TrueDepth observations used by personal learning.
            rows=[{key:value for key,value in row.items() if key!='paired'} for row in rows]
            clicks=data.get('click_anchors',[])
            if len(clicks)>160:raise ValueError('Too many click anchors')
            for anchor in clicks:
                if any(np.asarray(anchor[k]).shape!=shape or not np.isfinite(anchor[k]).all()
                       for k,shape in [('features',(22,)),('offset',(2,))]):raise ValueError('Invalid click anchor')
            if int(data.get('version',1))<5:
                # Old click labels had no trustworthy proximity gate. Preserve
                # the original file for recovery but do not apply or train on
                # those labels, including any network fitted with them.
                self.legacy_quarantined=max(len(clicks),sum(row.get('source')=='physical-click' for row in rows))
                rows=[row for row in rows if row.get('source')!='physical-click']
                clicks=[]
                self.trainer.invalidate()
                migration_needed=True
            else:
                self.click_accepted=int(data.get('click_accepted',len(clicks)))
                self.legacy_quarantined=int(data.get('legacy_quarantined',0))
                self.trainer.restore(data.get('personal_network'))
            self.click_anchors=clicks;self.rows=rows
        except (OSError,ValueError,KeyError,TypeError):pass
        if migration_needed and not read_only:
            backup=self.path.with_name(self.path.stem+'.before-safe-clicks.json')
            if not backup.exists():shutil.copy2(self.path,backup)
    def save(self):
        if self.key is None:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        if self.path.exists():
            try:
                old=json.loads(self.path.read_text(encoding='utf-8'))
                version=old.get('version',1)
                if type(version) is not int:
                    raise ValueError('Invalid correction data version')
            except (AttributeError,ValueError) as exc:
                raise OSError(f'Saved gaze corrections are damaged ({self.path.name})') from exc
            backup=self.path.with_name(self.path.stem+'.before-pose.json')
            if old.get('version')==1 and not backup.exists():shutil.copy2(self.path,backup)
            backup=self.path.with_name(self.path.stem+'.before-mouse.json')
            if version<3 and not backup.exists():shutil.copy2(self.path,backup)
            backup=self.path.with_name(self.path.stem+'.before-clicks.json')
            if version<4 and not backup.exists():shutil.copy2(self.path,backup)
            backup=self.path.with_name(self.path.stem+'.before-safe-clicks.json')
            if version<5 and not backup.exists():shutil.copy2(self.path,backup)
        temp.write_text(json.dumps({'version':5,'profile':self.key,'display':self.bounds.as_list(),'anchors':self.anchors,
                                   'pose_anchors':self.pose_anchors,'feature_scale':self.feature_scale.tolist(),'count':self.count,
                                   'click_accepted':self.click_accepted,'legacy_quarantined':self.legacy_quarantined,
                                   'click_anchors':self.click_anchors,
                                   'mouse_samples':self.rows,'personal_network':self.trainer.state()}),encoding='utf-8')
        os.replace(temp,self.path)
    def correct(self,gaze,measurement=None,now=None,radius=230):
        now=time.monotonic() if now is None else now
        if measurement is not None and 0<=now-measurement.captured<.2 and self.bounds is not None:
            x=np.asarray(measurement.features);point=np.asarray(measurement.point)
            if x.shape==(22,) and point.shape==(2,) and np.isfinite(np.r_[x,point]).all():
                corrected,route=adaptive_prediction(x,point,self.field_state(),measurement.supported,self.trainer.model)
                if corrected is not None and (route in ('pose','network','click') or self.anchors):
                    smoothed=self.pose_smoother.update(corrected,measurement.captured)
                    return Gaze(*smoothed,measurement.captured,gaze.radius if gaze else radius)
        self.pose_smoother.reset()
        if gaze is None or gaze.source!='iphone' or not self.anchors:return gaze
        x,y=gaze.x,gaze.y
        weights=[math.exp(-.5*(math.hypot(x-a[0],y-a[1])/600)**2) for a in self.anchors]
        total=sum(weights)
        if total<1e-6:return gaze
        strength=min(1.,max(weights)*1.3)
        dx=sum(w*a[2] for w,a in zip(weights,self.anchors))/total*strength
        dy=sum(w*a[3] for w,a in zip(weights,self.anchors))/total*strength
        b=self.bounds
        return replace(gaze,x=min(b.x+b.w-1,max(b.x,x+dx)),y=min(b.y+b.h-1,max(b.y,y+dy)))
    def field_state(self):
        return dict(poses=self.pose_anchors,anchors=self.anchors,clicks=self.click_anchors,scale=self.feature_scale.tolist(),bounds=self.bounds.as_list())
    @property
    def click_progress(self):
        return dict(accepted=self.click_accepted,legacy_quarantined=self.legacy_quarantined,
                    anchors=len(self.click_anchors),total_corrections=self.count)
    def _rebase_promoted_clicks(self):
        """Retain only click correction added after the promoted network's fit."""
        through=int(self.trainer.model.data.get('through_count',0))
        trained=self.trainer.promoted_clicks
        kept=[]
        for anchor in self.click_anchors:
            if int(anchor.get('revision',0))<=through:continue
            nearest=None;distance=float('inf')
            for old in trained:
                delta=(np.asarray(old['features'])-np.asarray(anchor['features']))/self.feature_scale
                d=float(np.sqrt(max(np.mean(delta[4:13]**2),np.mean(delta[np.r_[0:4,13:22]]**2))))
                if d<distance:nearest=old;distance=d
            if nearest is not None and distance<.16:
                increment=np.asarray(anchor['offset'])-np.asarray(nearest['offset'])
                if np.linalg.norm(increment)<.001:continue
                anchor=dict(anchor,offset=increment.tolist(),
                            count=max(1,int(anchor.get('count',1))-int(nearest.get('count',1))))
            kept.append(anchor)
        self.click_anchors=kept
    def _snapshot(self):
        return (self.count,copy.deepcopy(self.rows),copy.deepcopy(self.trainer.state()),
                copy.deepcopy(self.click_anchors),self.click_accepted)

    def _restore(self,snapshot):
        self.count,self.rows,state,self.click_anchors,self.click_accepted=snapshot
        self.trainer.restore(state)

    def learn_click(self,sample,target,*,weight=1.,limit=200.):
        """Apply one bounded click vote; corroborating votes grow a local correction."""
        self.last_click_result='invalid'
        if self.key is None or self.bounds is None or not self.bounds.contains(*target):return False
        x=np.asarray(sample.features);point=np.asarray(sample.point);target=np.asarray(target,float)
        if (not sample.supported or x.shape!=(22,) or point.shape!=(2,)
                or not np.isfinite(np.r_[x,point,target]).all()
                or not math.isfinite(weight) or not 0<weight<=1
                or not math.isfinite(limit) or limit<=0 or np.linalg.norm(target-point)>limit):return False
        soft,_=_soft_prediction(x,point,self.field_state(),sample.supported,self.trainer.model)
        if soft is None:return False
        baseline,_=adaptive_prediction(x,point,self.field_state(),sample.supported,self.trainer.model)
        residual=target-soft
        distance=float(np.linalg.norm(residual))
        if distance>limit:
            self.last_click_result='away';return False
        step=min(40.*weight,distance)
        nearest=None;nearest_distance=float('inf')
        for i,anchor in enumerate(self.click_anchors):
            delta=(np.asarray(anchor['features'])-x)/self.feature_scale
            d=float(np.sqrt(max(np.mean(delta[4:13]**2),np.mean(delta[np.r_[0:4,13:22]]**2))))
            if d<nearest_distance:nearest=i;nearest_distance=d
        if nearest is not None and nearest_distance<.16:
            anchor=self.click_anchors[nearest]
            old=np.asarray(anchor['offset'],float)
            if np.linalg.norm(old)>8 and distance>8 and np.dot(old,residual)<-.25*np.linalg.norm(old)*distance:
                self.last_click_result='conflict';return False
        snapshot=self._snapshot()
        if nearest is not None and nearest_distance<.16:
            old=np.asarray(self.click_anchors[nearest]['offset'],float)
            change=residual-old
            magnitude=float(np.linalg.norm(change))
            updated=old+change*(min(step,magnitude)/magnitude) if magnitude else old
            if np.linalg.norm(updated)>limit:updated*=limit/np.linalg.norm(updated)
            self.click_anchors[nearest]=dict(features=x.tolist(),offset=updated.tolist(),target=target.tolist(),
                                             count=int(self.click_anchors[nearest].get('count',1))+1,
                                             revision=self.count+1)
        else:
            correction=residual*(step/distance) if distance else residual
            self.click_anchors=(self.click_anchors+[dict(features=x.tolist(),offset=correction.tolist(),
                                                         target=target.tolist(),count=1,
                                                         revision=self.count+1)])[-160:]
        # A single click changes only the local field. The global regressor
        # receives a label after three agreeing observations in this pose.
        anchor=self.click_anchors[nearest] if nearest is not None and nearest_distance<.16 else self.click_anchors[-1]
        if anchor['count']>=3 and weight>=.5:
            self.trainer.observe_candidate(x,target,baseline,self.bounds.as_list())
            if self.trainer.promoted:
                self._rebase_promoted_clicks()
            self.rows=(self.rows+[dict(features=x.tolist(),target=target.tolist(),base=point.tolist(),
                                      baseline=(baseline if baseline is not None else point).tolist(),
                                      supported=True,source='physical-click',weight=float(weight))])[-1200:]
        self.count+=1;self.click_accepted+=1
        try:self.save()
        except OSError:
            self._restore(snapshot)
            raise
        self.undo=(self.undo+[snapshot])[-12:]
        self.pose_smoother.reset()
        self.trainer.request(self.rows,self.bounds.as_list(),self.count,self.field_state())
        self.last_click_result='accepted'
        return True

    def poll_training(self):
        if self.trainer.poll():
            self.pose_smoother.reset();self._training_save_pending=True
        if not self._training_save_pending or time.monotonic()<self._training_retry_at:return
        try:self.save()
        except OSError:
            self._training_retry_at=time.monotonic()+5.
            raise
        self._training_save_pending=False;self._training_retry_at=0.

    @property
    def model_status(self):return self.trainer.status(self.rows,self.bounds.as_list() if self.bounds else None)
    def undo_last(self):
        if self.undo:
            previous=self._snapshot()
            self._restore(self.undo[-1])
            try:self.save()
            except OSError:
                self._restore(previous)
                raise
            self.undo.pop()
        else:self.trainer.restore(self.trainer.state())
        self.pose_smoother.reset()
