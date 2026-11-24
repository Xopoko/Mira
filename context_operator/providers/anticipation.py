"""Numeric-only persistence for a separate personalized intention model."""
import json
import os
from pathlib import Path
from ..core.anticipation import IntentPredictor,OnlineRanker,GazeRanker


class IntentLearning(IntentPredictor):
    def __init__(self,path):
        self.path=Path(path);self.error='';self.writable=True
        super().__init__(own_pid=os.getpid())
        try:
            if self.path.exists():
                data=json.loads(self.path.read_text(encoding='utf-8'))
                if isinstance(data,dict) and data.get('schema')==2:
                    self.model=OnlineRanker.restore(data['mouse']);self.gaze_model=GazeRanker.restore(data['gaze'])
                else:raise ValueError('Unknown intention state')
        except (OSError,ValueError,TypeError,KeyError):
            self.error='Prediction learning stopped because Mira couldn’t read the saved model';self.writable=False

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state(),allow_nan=False),encoding='utf-8');os.replace(temp,self.path)

    def feedback(self,click,now):
        if not self.writable:return False
        previous=self.state()
        if not super().feedback(click,now):return False
        try:self.save();self.error=''
        except OSError as exc:
            self.model=OnlineRanker.restore(previous['mouse']);self.gaze_model=GazeRanker.restore(previous['gaze'])
            self.error='Couldn’t save prediction data: '+str(exc);return False
        return True

    @property
    def status(self):
        if self.error:return self.error
        mouse=self.model;gaze=self.gaze_model
        if not mouse.evaluated:mouse_text=f'Mouse · waiting for enough data · {mouse.learned} training clicks'
        else:mouse_text=f'Mouse · {mouse.hits}/{mouse.evaluated} early matches · initial model {mouse.prior_hits}/{mouse.evaluated}'
        if not gaze.evaluated:gaze_text=f'Gaze · waiting for enough data · {gaze.learned} training clicks'
        else:gaze_text=f'Gaze and click · {gaze.hits}/{gaze.evaluated} matches · initial model {gaze.prior_hits}/{gaze.evaluated}'
        return mouse_text+' · '+gaze_text
