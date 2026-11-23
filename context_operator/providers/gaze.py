from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import time
import numpy as np
from ..settings import LOCAL
from ..core.contracts import Gaze, GazeMeasurement
from ..foundations.gaze.profile import Profile
from ..foundations.gaze.smoothing import Smoother
from ..foundations.gaze.pose import PoseProfile
from ..foundations.gaze.arkit import ARKitWorker
from ..foundations.gaze.motion import MotionProfile
from .phone_connection import PhoneConnection

MOTION_PROFILE=LOCAL/'calibration/calibration-arkit-motion-v1.npz'
SPATIAL_PROFILE=LOCAL/'calibration/calibration-arkit-spatial-v1.npz'

def learning_path(method,key,folder=None):
    folder=LOCAL if folder is None else Path(folder)
    return folder/(f'gaze-adaptation-{method}-{key[:16]}.json' if method in ('motion','spatial','personal') else 'gaze-adaptation.json')

@dataclass
class Sample:
    seq:int=0
    captured:float=0.
    features:object=None
    blink:bool=False
    error:str=''
    fps:float=0.
    pose:tuple=(0.,0.,0.)
    reason:str=''
    source:str='arkit'
    depth_available:bool=False
    depth_recent:bool=False
    source_age_ms:float=0.
    read_wait_ms:float=0.
    eye_origins:tuple=()

class PhoneGaze:
    def __init__(self,bounds,calibration='stable',connection=None):
        self.connection=connection or PhoneConnection()
        folder=LOCAL/'calibration'
        if calibration not in ('stable','motion','spatial','personal'):raise ValueError('Unknown calibration')
        if calibration=='personal':
            from ..personalization import ProfileStore
            self.profile,status=ProfileStore(LOCAL).load(bounds)
            self.profile_key=status['profile_key']
            self.profile_quality=status['report'].get('quality','verified')
            # Held-out error informs only a search-region heuristic. It is not
            # a calibrated probability or a claim about attended objects.
            floor=140. if self.profile_quality=='starter' else 80.
            self.validation_radius=max(floor,min(650.,float(status['report']['p90_px'])*1.25))
            method='personal'
        else:
            selection=json.loads((folder/'calibration-arkit-v1-prediction.json').read_text(encoding='utf-8')) if calibration=='stable' else {}
            method=calibration if calibration in ('motion','spatial') else selection.get('method') or 'points'
            path=SPATIAL_PROFILE if method=='spatial' else MOTION_PROFILE if method=='motion' else folder/f'calibration-arkit-v1-{method}.npz'
            self.profile_key=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as data:metadata=json.loads(str(data['metadata']))
            if metadata['display']!=bounds.as_list():
                raise RuntimeError('Your gaze profile uses a different screen layout. Restore that layout or run setup again.')
            if method=='spatial':
                from ..foundations.gaze.spatial import SpatialProfile
                self.profile=SpatialProfile.load(path,metadata)
                self.profile_key=self.profile.fingerprint()
            else:self.profile=(MotionProfile if method=='motion' else PoseProfile if method=='pose' else Profile).load(path,metadata)
            if method=='pose' and self.profile.fingerprint()!=selection.get('pose_fingerprint'):
                raise RuntimeError('The selected setup doesn’t match the saved gaze profile')
        self.method=method;self.bounds=bounds
        self.worker=None;self.smoother=Smoother(.065,scale=bounds.h);self.last_seq=0
        self.measurement=None;self.invalid_captured=None
        self.feature_scale=np.maximum(self.profile.arrays['scale'],.005)

    def start(self):
        if self.worker and self.worker.is_alive():return
        self.worker=ARKitWorker(Sample,connection=self.connection);self.worker.start()

    def poll(self,radius):
        self.measurement=None;self.invalid_captured=None
        if not self.worker:return None,'Off',0
        sample=self.worker.latest()
        now=time.monotonic()
        missing_stamp=sample.captured if 0<=now-sample.captured<.2 else now
        if sample.error:self.invalid_captured=missing_stamp;return None,sample.error,0
        if sample.features is None or sample.blink or not 0<=now-sample.captured<.20:
            self.invalid_captured=missing_stamp
            self.smoother.reset()
            reason='iPhone data is out of date' if sample.features is not None and not 0<=now-sample.captured<.20 else sample.reason or 'Waiting for iPhone'
            return None,reason,sample.fps
        features=sample.features
        if getattr(self,'method','')=='spatial':
            try:features=self.profile.canonicalize(features)
            except ValueError:
                self.invalid_captured=missing_stamp;self.smoother.reset()
                return None,'Can’t locate gaze on the screen. Pair iPhone again.',sample.fps
        try:
            supported=self.profile.supports(features)
            prediction=self.profile.predict(features)
        except ValueError:
            self.invalid_captured=missing_stamp;self.smoother.reset()
            return None,'Gaze is outside the screen. Check your iPhone pairing.',sample.fps
        if np.asarray(prediction).shape!=(2,) or not np.isfinite(prediction).all():
            self.invalid_captured=missing_stamp
            self.smoother.reset();return None,'Can’t detect gaze',sample.fps
        self.measurement=GazeMeasurement(sample.captured,tuple(features),tuple(prediction),supported,
            raw_features=tuple(sample.features),eye_origins=sample.eye_origins)
        if not supported:
            self.smoother.reset()
            cm=-float(sample.features[6])*100
            span=self.profile.settings.get('distance_range_m') if isinstance(self.profile.settings,dict) else None
            detail=f' · setup distance {span[0]*100:.0f}–{span[1]*100:.0f} cm' if span else ''
            return None,f'Face visible · {cm:.0f} cm · position outside setup range'+detail,sample.fps
        if sample.seq!=self.last_seq or self.smoother.point is None:
            self.last_seq=sample.seq
            # Preserve the raw measurement for learning; constrain only output,
            # before smoothing, so an overshoot cannot hide the marker offscreen.
            b=np.asarray(self.bounds.as_list())
            self.point=self.smoother.update(np.clip(prediction,b[:2],b[:2]+b[2:]-1),sample.captured)
        point=getattr(self,'point',None)
        if point is None:return None,'Waiting for gaze',sample.fps
        name=('Gaze setup' if getattr(self,'profile_quality','')=='starter' else 'Personal gaze') if self.method=='personal' else 'Spatial gaze' if self.method=='spatial' else 'Head tracking' if self.method=='motion' else 'Three-position setup' if self.method=='pose' else 'TrueDepth'
        link='Wi-Fi' if self.connection.transport=='wifi' else 'USB'
        effective_radius=self.validation_radius if self.method=='personal' else radius
        return Gaze(*point,sample.captured,effective_radius),f'{name} · {-float(sample.features[6])*100:.0f} cm · {link} · data age ≤ {sample.source_age_ms:.0f} ms · response {sample.read_wait_ms:.0f} ms',sample.fps

    def stop(self):
        self.measurement=None
        if self.worker:
            self.worker.stop_event.set()
            self.worker.join(timeout=.7)
            self.worker=None
        self.smoother.reset()
