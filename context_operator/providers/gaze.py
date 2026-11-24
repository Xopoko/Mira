from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from ..settings import LOCAL
from ..core.contracts import Gaze, GazeMeasurement
from ..foundations.gaze.smoothing import Smoother
from ..foundations.gaze.arkit import ARKitWorker
from .phone_connection import PhoneConnection

def learning_path(key,folder=None):
    folder=LOCAL if folder is None else Path(folder)
    return folder/f'gaze-adaptation-personal-{key[:16]}.json'

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
    def __init__(self,bounds,connection=None):
        from ..personalization import ProfileStore
        self.connection=connection or PhoneConnection()
        self.profile,status=ProfileStore(LOCAL).load(bounds)
        self.profile_key=status['profile_key']
        # Held-out error informs the search region.
        self.validation_radius=max(140.,min(650.,float(status['report']['p90_px'])*1.25))
        self.bounds=bounds
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
            return None,f'Face visible · {cm:.0f} cm · position outside setup range',sample.fps
        if sample.seq!=self.last_seq or self.smoother.point is None:
            self.last_seq=sample.seq
            # Preserve the raw measurement for learning; constrain only output,
            # before smoothing, so an overshoot cannot hide the marker offscreen.
            b=np.asarray(self.bounds.as_list())
            self.point=self.smoother.update(np.clip(prediction,b[:2],b[:2]+b[2:]-1),sample.captured)
        point=getattr(self,'point',None)
        if point is None:return None,'Waiting for gaze',sample.fps
        link='Wi-Fi' if self.connection.transport=='wifi' else 'USB'
        return Gaze(*point,sample.captured,self.validation_radius),f'Gaze setup · {-float(sample.features[6])*100:.0f} cm · {link} · data age ≤ {sample.source_age_ms:.0f} ms · response {sample.read_wait_ms:.0f} ms',sample.fps

    def stop(self):
        self.measurement=None
        if self.worker:
            self.worker.stop_event.set()
            self.worker.join(timeout=.7)
            self.worker=None
        self.smoother.reset()
