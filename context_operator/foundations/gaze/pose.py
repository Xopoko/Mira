import hashlib, json
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .profile import Profile

def distance_m(features):
    return -float(np.asarray(features)[6])

def _design(z,kind,support,gamma):
    return np.c_[z,rbf_kernel(z,support,gamma=gamma)] if kind=='rbf_affine' else Profile.design(z,kind)

def _predict(x,arrays,model):
    z=(np.atleast_2d(x)-arrays['mean'])/arrays['scale']
    design=_design(z,model[0],arrays['support'],model[2])
    return (design@arrays['coef']+arrays['intercept'])*arrays['target_scale']+arrays['target_mean']

def _fit(records, spec, metadata, kind='linear', gamma=0.):
    """Equal fixation weight; penalize within-fixation variance explicitly."""
    centers=np.array([np.mean(r[1],axis=0) for r in records])
    y=np.asarray([r[0] for r in records],float)
    mean=centers.mean(0);scale=np.maximum(centers.std(0),.001)
    z=(centers-mean)/scale
    target_mean=y.mean(0);target_scale=np.maximum(y.std(0),1.)
    bases=[_design((x-mean)/scale,kind,z,gamma) for _,x,_ in records]
    design=np.c_[np.array([b.mean(0) for b in bases]),np.ones(len(z))]
    covariance=np.zeros((design.shape[1],design.shape[1]))
    for b in bases:
        dev=np.c_[b-b.mean(0),np.zeros(len(b))]
        covariance+=dev.T@dev/len(dev)
    alpha,noise=spec
    penalty=np.diag(np.r_[np.full(design.shape[1]-1,alpha),0.])
    coef=np.linalg.solve(design.T@design+penalty+noise*covariance,
                         design.T@((y-target_mean)/target_scale))
    return dict(mean=mean,scale=scale,coef=coef[:-1],intercept=coef[-1],
                target_mean=target_mean,target_scale=target_scale,support=z,
                training_x=centers,training_y=y)

class PoseProfile(Profile):
    def __init__(self,arrays,metadata,settings):
        super().__init__(arrays,metadata,settings)
        bounds=np.asarray(settings.get('distance_range_m'),float)
        depths=np.asarray(settings.get('pose_distances_m'),float)
        if (self.training_x.shape[1]!=22 or settings.get('kind')!='pose-v1' or
                bounds.shape!=(2,) or depths.shape!=(3,) or
                not np.isfinite(bounds).all() or not np.isfinite(depths).all() or
                not 0.15<bounds[0]<bounds[1]<1.5 or
                not (bounds[0]<=depths.min()<=depths.max()<=bounds[1]) or
                depths[1]-depths[0]<.045 or depths[0]-depths[2]<.045):
            raise ValueError('Invalid three-position calibration')
        kind,alpha,gamma=settings['model']
        if kind not in ('linear','quadratic','rbf_affine') or not np.isfinite([alpha,gamma]).all() or alpha<=0 or (kind=='rbf_affine' and gamma<=0):
            raise ValueError('Invalid pose model')
        columns=_design(np.zeros((1,22)),kind,arrays['support'],gamma).shape[1]
        if arrays['coef'].shape!=(columns,2):raise ValueError('Invalid pose coefficients')
        self.point_reference=None
        if 'point_blend' in settings:
            blend=settings['point_blend']
            if not 0<=blend['strength']<=.5 or not .01<=blend['width_m']<=.1 or not .2<blend['zero_at_m']<1.5:
                raise ValueError('Invalid point blend')
            refs={k[len('reference_'):]:v for k,v in arrays.items() if k.startswith('reference_')}
            self.point_reference=Profile(refs,metadata,blend['settings'])
            if self.point_reference.arrays['mean'].shape!=(22,):raise ValueError('Invalid point reference')

    def predict_base(self,features):
        result=_predict(features,self.arrays,self.settings['model'])
        if self.point_reference is not None:
            x=np.atleast_2d(features);p=self.point_reference;b=self.settings['point_blend']
            z=(x-p.arrays['mean'])/p.arrays['scale']
            distance=np.sqrt(np.mean((z[:,None]-p.arrays['support'][None])**2,axis=2)).min(1)/p.settings['support_limit']
            support=np.clip((1-distance)/.2,0,1);support=support*support*(3-2*support)
            depth=np.clip((b['zero_at_m']+x[:,6])/b['width_m'],0,1);depth=depth*depth*(3-2*depth)
            weight=b['strength']*support*depth
            result+=weight[:,None]*(np.atleast_2d(p.predict(x))-result)
        return result[0] if np.asarray(features).ndim==1 else result

    @classmethod
    def load(cls,path,metadata):
        with np.load(path,allow_pickle=False) as f:
            actual=json.loads(str(f['metadata']));settings=json.loads(str(f['settings']))
            if actual!=metadata or settings.get('version')!=2:raise ValueError('Incompatible pose profile')
            arrays={k:f[k] for k in f.files if k not in ('metadata','settings')}
        return cls(arrays,actual,settings)

    def supports(self, features):
        x=np.asarray(features)
        if x.shape!=(22,) or not np.isfinite(x).all():return False
        lo,hi=self.settings['distance_range_m']
        return lo<=distance_m(x)<=hi and super().supports(x)

    def fingerprint(self):
        digest=hashlib.sha256(json.dumps(self.settings,sort_keys=True).encode())
        digest.update(json.dumps(self.metadata,sort_keys=True).encode())
        for key in sorted(self.arrays):
            a=np.ascontiguousarray(self.arrays[key]);digest.update(key.encode())
            digest.update(str(a.shape).encode());digest.update(a.tobytes())
        return digest.hexdigest()

    @classmethod
    def fit_records(cls,records,original,base_id,extra_records=()):
        if len(records)!=27 or [int(r[2]) for r in records]!=[p for p in range(3) for _ in range(9)]:
            raise ValueError('Look at nine points in each of the three positions.')
        observations=list(records)+list(extra_records)
        for target,x,pose in observations:
            if pose not in (0,1,2) or np.asarray(x).ndim!=2 or x.shape[1]!=22 or len(x)<28 or not np.isfinite(x).all() or np.asarray(target).shape!=(2,) or not np.isfinite(target).all():
                raise ValueError('Position measurements are incomplete.')
        depths=np.array([np.median(np.concatenate([-x[:,6] for _,x,p in observations if p==pose])) for pose in range(3)])
        if depths[1]-depths[0]<.045 or depths[0]-depths[2]<.045:
            raise ValueError('The positions are too close together. Move at least 5 cm nearer and farther away.')
        # Original point observations act as replay, never as edited files.
        replay=[(target,original.training_x[(original.training_y==target).all(1)],-1)
                for target in np.unique(original.training_y,axis=0)]
        training=observations+replay
        targets=np.asarray([r[0] for r in training]);ox,oy,w,h=original.metadata['display']
        # Hold out whole spatial cells across ALL poses and replay records.
        uv=(targets-[ox,oy])/[w,h]
        groups=np.clip(np.rint(uv[:,0]*2),0,2)+3*np.clip(np.rint(uv[:,1]*2),0,2)
        folds=list(GroupKFold(n_splits=3).split(targets,groups=groups))
        scores=[]
        with threadpool_limits(limits=2):
            specs=[(k,a,n,0.) for k in ('linear','quadratic') for a in (.03,.3,3.) for n in (2.,10.)]
            specs += [('rbf_affine',a,2.,g) for a in (.001,.03) for g in (.04,.2)]
            for kind,alpha,noise,gamma in specs:
                    errors=[];jitters=[]
                    for train,test in folds:
                        arrays=_fit([training[i] for i in train],(alpha,noise),original.metadata,kind,gamma)
                        for i in test:
                            target,x,_=training[i]
                            pred=_predict(x,arrays,(kind,alpha,gamma))
                            errors.append(np.median(np.linalg.norm(pred-target,axis=1)))
                            jitters.append(np.median(np.linalg.norm(pred-np.median(pred,axis=0),axis=1)))
                    score=float(np.median(errors)+.25*np.percentile(errors,90)+np.median(jitters))
                    scores.append(dict(model=kind,alpha=alpha,gamma=gamma,noise=noise,score=score))
            best=min(scores,key=lambda s:s['score'])
            arrays=_fit(training,(best['alpha'],best['noise']),original.metadata,best['model'],best['gamma'])
        settings=dict(version=2,kind='pose-v1',model=[best['model'],best['alpha'],best['gamma']],
                      support_limit=1.2,base_fingerprint=base_id,cv_by_cell=scores,
                      replayed_check_fixations=len(extra_records),
                      noise_penalty=best['noise'],pose_distances_m=depths.tolist(),
                      distance_range_m=[float(depths.min()-.035),float(depths.max()+.035)])
        return cls(arrays,original.metadata,settings)
