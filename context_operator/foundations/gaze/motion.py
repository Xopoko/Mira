"""TrueDepth calibration from fixed targets observed while the head moves."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .profile import Profile
from .pose import _fit,_predict

class MotionProfile(Profile):
    def __init__(self,arrays,metadata,settings):
        super().__init__(arrays,metadata,settings)
        if (settings.get('kind')!='motion-v1' or self.training_x.shape[1]!=22
                or arrays['coverage'].ndim!=2 or arrays['coverage'].shape[1]!=22
                or not 12<=len(arrays['coverage'])<=700):
            raise ValueError('Invalid motion calibration')

    def predict_base(self,features):
        result=_predict(features,self.arrays,self.settings['model'])
        return result[0] if np.asarray(features).ndim==1 else result

    def supports(self,features):
        x=np.asarray(features)
        if x.shape!=(22,) or not np.isfinite(x).all():return False
        z=(x-self.arrays['mean'])/self.arrays['scale']
        delta=self.arrays['coverage']-z
        # Head pose and eye direction must both be near observed states.
        distance=np.sqrt(np.maximum(np.mean(delta[:,4:13]**2,axis=1),
                                    np.mean(delta[:,np.r_[0:4,13:22]]**2,axis=1))).min()
        return bool(distance<=self.settings['support_limit'])

    @classmethod
    def fit_records(cls,records,bounds):
        if len(records)<9:raise ValueError('Setup needs points across the whole screen')
        expanded=[];groups=[];all_x=[]
        for index,(target,features) in enumerate(records):
            x=np.asarray(features,float);target=np.asarray(target,float)
            if (x.ndim!=2 or x.shape[1]!=22 or len(x)<48 or target.shape!=(2,)
                    or not np.isfinite(x).all() or not np.isfinite(target).all()):
                raise ValueError('Head movement data is incomplete')
            # Short temporal segments preserve distinct poses. Penalizing variance
            # within each segment reduces jitter without averaging away all motion.
            for chunk in np.array_split(x,6):expanded.append((target,chunk,index));groups.append(index)
            all_x.extend(x)
        folds=list(GroupKFold(n_splits=3).split(expanded,groups=groups));scores=[]
        metadata=dict(display=list(bounds),engine='arkit-truedepth-motion-v1')
        candidates=[('linear',.03,2.,0.),('linear',.3,10.,0.),('quadratic',3.,10.,0.),
                    ('rbf_affine',.03,2.,.04),('rbf_affine',.03,10.,.2)]
        with threadpool_limits(limits=1):
            for kind,alpha,noise,gamma in candidates:
                errors=[];jitters=[]
                for train,test in folds:
                    a=_fit([expanded[i] for i in train],(alpha,noise),metadata,kind,gamma)
                    for group in np.unique(np.asarray(groups)[test]):
                        ids=[i for i in test if groups[i]==group]
                        x=np.concatenate([expanded[i][1] for i in ids]);target=expanded[ids[0]][0]
                        pred=_predict(x,a,(kind,alpha,gamma))
                        errors.append(float(np.median(np.linalg.norm(pred-target,axis=1))))
                        jitters.append(float(np.median(np.linalg.norm(pred-np.median(pred,axis=0),axis=1))))
                scores.append(dict(model=[kind,alpha,gamma],noise=noise,
                                   score=float(np.median(errors)+.25*np.percentile(errors,90)+np.median(jitters)),
                                   median_px=float(np.median(errors)),p90_px=float(np.percentile(errors,90))))
            best=min(scores,key=lambda r:r['score']);kind,alpha,gamma=best['model']
            arrays=_fit(expanded,(alpha,best['noise']),metadata,kind,gamma)
        x=np.asarray(all_x);ids=np.linspace(0,len(x)-1,min(700,len(x)),dtype=int)
        arrays['coverage']=(x[ids]-arrays['mean'])/arrays['scale']
        settings=dict(version=2,kind='motion-v1',model=best['model'],noise_penalty=best['noise'],
                      support_limit=1.2,distance_range_m=[float((-x[:,6]).min()),float((-x[:,6]).max())],cv_by_target=scores)
        return cls(arrays,metadata,settings)

    @classmethod
    def load(cls,path,metadata):
        with np.load(path,allow_pickle=False) as f:
            actual=json.loads(str(f['metadata']));settings=json.loads(str(f['settings']))
            if actual!=metadata or settings.get('version')!=2:raise ValueError('Incompatible motion profile')
            arrays={k:f[k] for k in f.files if k not in ('metadata','settings')}
        return cls(arrays,actual,settings)
