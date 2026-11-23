"""Small direct gaze regressor. Training runs off the UI thread; inference is NumPy.

Mouse labels are independent of predicted gaze. Validation holds out whole screen
cells, including original calibration points in those cells, not adjacent frames.
"""
import copy
import queue
import threading
import warnings
import numpy as np


def adaptive_prediction(features,point,field,supported=True,model=None,*,clip=True):
    """Corroborated click votes add a bounded local correction."""
    result,route=_soft_prediction(features,point,field,supported,model,clip=clip)
    through_count=model.data.get('through_count',0) if model is not None else 0
    clicks=[a for a in field.get('clicks',[]) if a.get('revision',0)>through_count] if model is not None else field.get('clicks',[])
    if clicks:
        delta=(np.asarray([a['features'] for a in clicks])-features)/np.asarray(field['scale'])
        distance=np.sqrt(np.maximum(np.mean(delta[:,4:13]**2,axis=1),np.mean(delta[:,np.r_[0:4,13:22]]**2,axis=1)))
        index=int(distance.argmin());d=distance[index]
        if d<(.30 if supported else .15):
            b=np.asarray(field['bounds']);target=(np.asarray(result) if result is not None else np.asarray(point))+clicks[index]['offset']
            if clip:target=np.clip(target,b[:2],b[:2]+b[2:]-1)
            # The anchor itself advances only a bounded amount per observation.
            weight=1. if d<=.06 else max(0.,(1-(d-.06)/.24))**2
            if result is None:result=target
            return result*(1-weight)+target*weight,'click'
    return result,route


def _soft_prediction(features,point,field,supported=True,model=None,*,clip=True):
    """Unsmoothed production predictor, also used for a frozen CV comparator."""
    x=np.asarray(features);point=np.asarray(point,float)
    poses=field['poses'];bounds=np.asarray(field['bounds'])
    def constrain(value):return np.clip(value,bounds[:2],bounds[:2]+bounds[2:]-1) if clip else np.asarray(value)
    network=model.predict(x) if model is not None else None
    if poses:
        delta=(np.asarray([a['features'] for a in poses])-x)/np.asarray(field['scale'])
        distance=np.sqrt(np.maximum(np.mean(delta[:,4:13]**2,axis=1),np.mean(delta[:,np.r_[0:4,13:22]]**2,axis=1)))
        # Observations collected AFTER a network's fit must still teach immediately.
        fresh=np.array([a.get('revision',0)>model.data.get('through_count',0) for a in poses]) if model else np.ones(len(poses),bool)
        use=fresh if network is not None else np.ones(len(poses),bool)
        limit=(.45 if supported else .35) if network is not None else (.85 if supported else .35)
        if use.any() and distance[use].min()<limit:
            weights=np.exp(-.5*(distance/.45)**2);weights[(distance>1.2)|~use]=0
            offset=(weights/weights.sum())@np.asarray([a['offset'] for a in poses])
            result=point+offset*min(1.,weights.max()*1.3)
            return constrain(result),'pose'
    if network is not None:return constrain(network),'network'
    if not supported:return None,'unsupported'
    anchors=field['anchors']
    if anchors:
        anchors=np.asarray(anchors);weights=np.exp(-.5*np.sum((anchors[:,:2]-point)**2,axis=1)/600**2)
        if weights.sum()>1e-6:point=point+(weights@anchors[:,2:4])/weights.sum()*min(1.,weights.max()*1.3)
    return constrain(point),'base'


def cells(targets, bounds):
    b=np.asarray(bounds,float)
    uv=(np.asarray(targets)-b[:2])/b[2:]
    return (np.clip((uv[:,0]*5).astype(int),0,4)+5*np.clip((uv[:,1]*3).astype(int),0,2))

def field_without_cells(field,held,bounds,rows):
    """Remove labels from the held-out screen cells before comparing a model."""
    if field is None:return None
    result=copy.deepcopy(field)
    result['clicks']=[a for a in result.get('clicks',[])
                      if 'target' in a and cells([a['target']],bounds)[0] not in held]
    result['anchors']=[a for a in result.get('anchors',[])
                       if cells([np.asarray(a[:2])+np.asarray(a[2:4])],bounds)[0] not in held]
    # Old pose anchors have no target provenance. Keep one only when its
    # features match a row and every nearby row names the same training cell.
    if rows and result.get('poses'):
        features=np.asarray([r['features'] for r in rows])
        groups=cells([r['target'] for r in rows],bounds)
        scale=np.asarray(result['scale'])
        kept=[]
        for a in result['poses']:
            if 'target_cell' in a:
                if a['target_cell'] is not None and a['target_cell'] not in held:kept.append(a)
                continue
            d=np.sqrt(np.mean(((features-np.asarray(a['features']))/scale)**2,axis=1))
            nearby=groups[d<.16]
            if d.min()<1e-6 and len(nearby) and len(set(nearby))==1 and nearby[0] not in held:
                kept.append(a)
        result['poses']=kept
    elif result.get('poses'):
        result['poses']=[a for a in result['poses']
                         if 'target_cell' in a and a['target_cell'] is not None and a['target_cell'] not in held]
    return result


class DirectModel:
    def __init__(self, data):
        self.data=data
        self.mean=np.asarray(data['mean'],float);self.scale=np.asarray(data['scale'],float)
        self.center=np.asarray(data['center'],float);self.extent=np.asarray(data['extent'],float)
        self.weights=[np.asarray(w,float) for w in data['weights']]
        self.biases=[np.asarray(b,float) for b in data['biases']]
        self.support=np.asarray(data['support'],float)
        shapes=[(22,32),(32,16),(16,2)]
        if (self.mean.shape!=(22,) or self.scale.shape!=(22,) or np.any(self.scale<=0)
                or self.center.shape!=(2,) or self.extent.shape!=(2,) or np.any(self.extent<=0)
                or len(self.weights)!=3 or len(self.biases)!=3
                or any(w.shape!=s or b.shape!=(s[1],) for w,b,s in zip(self.weights,self.biases,shapes))
                or self.support.ndim!=2 or self.support.shape[1]!=22 or not 1<=len(self.support)<=700
                or not all(np.isfinite(a).all() for a in [self.mean,self.scale,self.center,self.extent,self.support,*self.weights,*self.biases])):
            raise ValueError('Invalid personal gaze network')

    def predict(self, features, guarded=True):
        x=np.asarray(features,float)
        if x.shape!=(22,) or not np.isfinite(x).all():return None
        z=(x-self.mean)/self.scale
        if guarded and np.sqrt(np.mean((self.support-z)**2,axis=1)).min()>.9:return None
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=z@w+b
            if i<2:z=np.tanh(z)
        result=z*self.extent+self.center
        return result if np.isfinite(result).all() else None


def fit_network(x,y,bounds,sample_quality=None):
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import ConvergenceWarning
    mean=x.mean(0);scale=np.maximum(x.std(0),.005)
    center=np.asarray(bounds[:2])+np.asarray(bounds[2:])/2
    extent=np.asarray(bounds[2:])/2
    groups=cells(y,bounds)
    weights=np.array([1/np.count_nonzero(groups==g) for g in groups]);weights*=len(weights)/weights.sum()
    if sample_quality is not None:weights*=np.asarray(sample_quality)
    net=MLPRegressor(hidden_layer_sizes=(32,16),activation='tanh',solver='lbfgs',alpha=.03,
                     max_iter=220,max_fun=10000,tol=1e-5,random_state=17)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        net.fit((x-mean)/scale,(y-center)/extent,sample_weight=weights)
    # Deterministic, bounded support set; no sklearn object or pickle at inference.
    support=(x-mean)/scale
    support=support[np.linspace(0,len(support)-1,min(700,len(support)),dtype=int)]
    return DirectModel(dict(mean=mean.tolist(),scale=scale.tolist(),center=center.tolist(),extent=extent.tolist(),
                           weights=[w.tolist() for w in net.coefs_],biases=[b.tolist() for b in net.intercepts_],
                           support=support.tolist()))


def train_candidate(rows,seeds,bounds,incumbent=None,field=None):
    from threadpoolctl import threadpool_limits
    # A long stay on one button must not dominate either fitting or evaluation.
    groups=cells([r['target'] for r in rows],bounds)
    selected=[]
    for cell in np.unique(groups):
        ids=np.flatnonzero(groups==cell)
        selected.extend(ids[np.linspace(0,len(ids)-1,min(40,len(ids)),dtype=int)])
    rows=[rows[i] for i in sorted(selected)]
    x=np.array([r['features'] for r in rows]);y=np.array([r['target'] for r in rows])
    quality=np.array([float(r.get('weight',1.)) for r in rows])
    baseline=np.array([r['baseline'] for r in rows]);groups=cells(y,bounds)
    unique=np.unique(groups)
    if len(unique)<6 or len(rows)<48:return None
    sx,sy=seeds
    sg=cells(sy,bounds) if len(sy) else np.array([])
    predictions=np.empty_like(y);accepted_support=np.zeros(len(y),bool)
    with threadpool_limits(limits=1):
        for fold in range(3):
            held=unique[fold::3];test=np.isin(groups,held);train=~test;seed_mask=~np.isin(sg,held)
            comparison=field_without_cells(field,held,bounds,rows)
            for index in np.flatnonzero(test):
                row=rows[index]
                if comparison is not None:
                    # An incumbent network may already contain these rows;
                    # compare against only the held-out, non-network field.
                    # The live incumbent is compared later on future clicks.
                    prior,_=adaptive_prediction(row['features'],row['base'],comparison,row.get('supported',True),None)
                else:prior=None
                if prior is not None:baseline[index]=prior
            model=fit_network(np.r_[x[train],sx[seed_mask]],np.r_[y[train],sy[seed_mask]],bounds,
                              np.r_[quality[train],np.ones(seed_mask.sum())])
            for index in np.flatnonzero(test):
                predicted=model.predict(x[index])
                accepted_support[index]=predicted is not None
                predictions[index]=predicted if predicted is not None else baseline[index]
        err=np.linalg.norm(predictions-y,axis=1);base=np.linalg.norm(baseline-y,axis=1)
        errors=np.array([np.median(err[groups==g]) for g in unique])
        old=np.array([np.median(base[groups==g]) for g in unique])
        median=float(np.median(errors));p90=float(np.percentile(errors,90))
        base_median=float(np.median(old));base_p90=float(np.percentile(old,90))
        accepted=bool(median<base_median*.94 and p90<=base_p90*1.03 and accepted_support.mean()>=.5)
        final=fit_network(np.r_[x,sx],np.r_[y,sy],bounds,np.r_[quality,np.ones(len(sy))]) if accepted else None
    return dict(model=final.data if final else None,accepted=accepted,targets=len(unique),samples=len(rows),
                median_px=median,p90_px=p90,baseline_median_px=base_median,baseline_p90_px=base_p90,
                support_fraction=float(accepted_support.mean()))


class PersonalTrainer:
    def __init__(self):
        self.epoch=0;self.worker=None;self.results=queue.SimpleQueue();self.attempted=0
        self.model=None;self.report=None;self.seeds=(np.empty((0,22)),np.empty((0,2)))
        self.pending=None;self.pending_clicks=[];self.shadow=[];self.promoted=False;self.promoted_clicks=[];self.error=''

    def invalidate(self):
        self.epoch+=1;self.model=None;self.pending=None;self.pending_clicks=[];self.shadow=[]
        self.promoted=False;self.promoted_clicks=[]
        self.report=None;self.attempted=0;self.error=''

    def set_seed(self,profile):
        self.seeds=(np.array(profile.arrays['training_x'],copy=True),np.array(profile.arrays['training_y'],copy=True))

    def restore(self,data):
        self.invalidate()
        if data:
            if data.get('model'):self.model=DirectModel(data['model'])
            if data.get('pending'):self.pending=DirectModel(data['pending'])
            self.pending_clicks=copy.deepcopy(data.get('pending_clicks',[]))
            self.shadow=list(data.get('shadow',[]))[-12:]
            self.report=data.get('report');self.attempted=int(data.get('attempted',0))

    def state(self):
        return dict(model=self.model.data if self.model else None,pending=self.pending.data if self.pending else None,
                    pending_clicks=self.pending_clicks,
                    shadow=self.shadow,report=self.report,attempted=self.attempted)

    def request(self,rows,bounds,count,field=None):
        if self.pending or self.worker and self.worker.is_alive():return
        if len(rows)<48 or count-self.attempted<48 or len(np.unique(cells([r['target'] for r in rows],bounds)))<6:return
        self.attempted=count;epoch=self.epoch;data=copy.deepcopy(rows);seeds=self.seeds;model=self.model;field=copy.deepcopy(field)
        def run():
            try:
                result=train_candidate(data,seeds,bounds,model,field)
                if result and result['model']:
                    result['model']['through_count']=count
                    result['trained_clicks']=field.get('clicks',[]) if field else []
                self.results.put((epoch,result,''))
            except Exception as exc:self.results.put((epoch,None,type(exc).__name__))
        self.worker=threading.Thread(target=run,daemon=True,name='personal-gaze-training');self.worker.start()

    def poll(self):
        changed=False
        while not self.results.empty():
            epoch,result,error=self.results.get()
            if epoch!=self.epoch:continue
            self.error=error
            if result:
                self.report={k:v for k,v in result.items() if k not in ('model','trained_clicks')}
                if result['model']:
                    self.pending=DirectModel(result['model']);self.pending_clicks=copy.deepcopy(result.get('trained_clicks',[]));self.shadow=[]
            changed=True
        return changed

    def observe_candidate(self,features,target,baseline,bounds):
        """Evaluate a staged network on future corroborated labels only."""
        self.promoted=False;self.promoted_clicks=[]
        if self.pending is None:return False
        prediction=self.pending.predict(features)
        if prediction is None:return False
        target=np.asarray(target,float);baseline=np.asarray(baseline,float)
        if not np.isfinite(np.r_[target,baseline]).all():return False
        cell=int(cells([target],bounds)[0])
        if sum(item['cell']==cell for item in self.shadow)>=4:return False
        self.shadow.append(dict(cell=cell,candidate=float(np.linalg.norm(prediction-target)),
                                baseline=float(np.linalg.norm(baseline-target))))
        if len(self.shadow)<12 or len({item['cell'] for item in self.shadow})<3:return True
        errors=np.asarray([item['candidate'] for item in self.shadow])
        prior=np.asarray([item['baseline'] for item in self.shadow])
        good=bool(np.median(errors)<np.median(prior)*.94 and
                  np.percentile(errors,90)<=np.percentile(prior,90)*1.03)
        self.report=dict(self.report or {},shadow_accepted=good,shadow_samples=len(self.shadow),
                         shadow_cells=len({item['cell'] for item in self.shadow}),
                         shadow_median_px=float(np.median(errors)),shadow_baseline_median_px=float(np.median(prior)))
        if good:
            self.model=self.pending;self.promoted=True
            self.promoted_clicks=self.pending_clicks
        self.pending=None;self.pending_clicks=[];self.shadow=[]
        return True

    def status(self,rows,bounds):
        if self.worker and self.worker.is_alive():return 'Mira is learning a more accurate gaze model in the background'
        if self.error:return 'Gaze learning didn’t finish · '+self.error
        if self.pending:
            n=len({item['cell'] for item in self.shadow})
            return f'Checking the new gaze model on upcoming clicks · {len(self.shadow)}/12 clicks, {n}/3 screen areas'
        if self.report:
            r=self.report
            if r.get('shadow_accepted'):return f'Gaze model checked on new clicks · {r["shadow_median_px"]:.0f} px, previously {r["shadow_baseline_median_px"]:.0f} px'
            if r.get('shadow_accepted') is False:return 'Mira kept the current gaze model and continues learning small corrections'
            if r['accepted']:return f'Gaze model checked against saved clicks · {r["median_px"]:.0f} px, previously {r["baseline_median_px"]:.0f} px'
            return ('Using the more accurate existing gaze model' if self.model
                    else 'Mira is gathering more clicks to improve the gaze model')
        n=len(np.unique(cells([r['target'] for r in rows],bounds))) if rows and bounds else 0
        return f'Gaze model progress ·{n}/6 screen areas · {len(rows)}/48 clicks'
