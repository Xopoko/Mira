import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from types import SimpleNamespace
import numpy as np
from context_operator.core.contracts import GazeMeasurement,Rect
from context_operator.foundations.gaze.motion import MotionProfile
from context_operator.foundations.gaze.profile import Smoother
from context_operator.providers.gaze import PhoneGaze,Sample
from context_operator.providers.learning import GazeLearning
from context_operator.providers.personal_model import adaptive_prediction

BOUNDS=Rect(0,0,1000,700)
TRAIN_TARGETS=tuple((u,v) for v in (.08,.5,.92) for u in (.08,.36,.64,.92))
def features(u,v,z):
    x=np.zeros(22);x[:4]=[u+(z-.65)*.7,v-.2*(z-.65),u+(z-.65)*.7,v-.2*(z-.65)]
    x[4]=.06*np.sin(z*8);x[5]=.035*np.cos(z*8);x[6]=-z
    x[7:13]=[1,0,0,1,0,0];x[13:]=np.tile([u*.1,v*.1,1.],3);return x
def records():
    return [(np.asarray([u,v])*[1000,700],np.array([features(u,v,z) for z in np.linspace(.38,.95,96)])) for u,v in TRAIN_TARGETS]
def m(t,u=.5,v=.5,z=.4):return GazeMeasurement(t,tuple(features(u,v,z)),(3000,-500),False)

class Baseline:
    signature='baseline'
    def predict(self,x):
        if not .61<=-x[6]<=.89:return None
        return np.array([(x[0]-.7*(-x[6]-.65))*1000+40,(x[1]+.2*(-x[6]-.65))*700+20])

class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.profile=MotionProfile.fit_records(records(),BOUNDS.as_list())
    def test_real_fit_generalizes_targets_and_distance_without_old_gate(self):
        for z in (.4,.65,.9):
            for u,v in ((u,v) for v in (.15,.5,.85) for u in (.15,.5,.85)):
                f=features(u,v,z);self.assertTrue(self.profile.supports(f))
                self.assertLess(np.linalg.norm(self.profile.predict(f)-np.array([u*1000,v*700])),12)
        self.assertFalse(self.profile.supports(np.full(22,8)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'profile.npz';self.profile.save(path)
            restored=MotionProfile.load(path,self.profile.metadata)
            np.testing.assert_allclose(restored.predict(features(.6,.2,.9)),self.profile.predict(features(.6,.2,.9)))
    def test_base_output_clips_but_training_keeps_full_extrapolation(self):
        from context_operator.providers.phone_connection import PhoneConnection
        phone=PhoneGaze.__new__(PhoneGaze);phone.bounds=BOUNDS;phone.worker=Mock();phone.profile=Mock()
        phone.connection=PhoneConnection()
        phone.profile.supports.return_value=True;phone.profile.predict.return_value=np.array([2000,-500])
        phone.smoother=Smoother(.065);phone.last_seq=0;phone.method='pose'
        phone.worker.latest.return_value=Sample(seq=1,captured=100,features=features(.6,.2,.65))
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):gaze,_,_=phone.poll(230)
        self.assertEqual((gaze.x,gaze.y),(999,0));self.assertEqual(phone.measurement.point,(2000,-500))
        phone.worker.latest.return_value.captured=99
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):
            gaze,reason,_=phone.poll(230);self.assertIsNone(gaze);self.assertIsNone(phone.measurement);self.assertIn('out of date',reason)
    def test_diagnostic_separates_tracking_from_coverage(self):
        phone=PhoneGaze.__new__(PhoneGaze);phone.bounds=BOUNDS;phone.worker=Mock();phone.profile=Mock()
        phone.profile.settings={'distance_range_m':[.612,.887]};phone.profile.supports.return_value=False
        phone.profile.predict.return_value=np.array([2000,-500]);phone.smoother=Smoother(.065)
        phone.worker.latest.return_value=Sample(seq=1,captured=100,features=features(.5,.5,.4))
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):gaze,reason,_=phone.poll(230)
        self.assertIsNone(gaze);self.assertIn('40 cm',reason);self.assertIn('61–89 cm',reason)
        self.assertIsNotNone(phone.measurement)
    def test_legacy_corrections_compare_same_clipping_order_as_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            learning=GazeLearning(Path(folder)/'learning.json');learning.bind('baseline',BOUNDS)
            learning.anchors=[[-100,100,200,0,1]]
            profile=SimpleProfile()
            measurement=GazeMeasurement(100,tuple(features(.5,.5,.65)),(-100,100),True)
            from context_operator.core.contracts import Gaze
            point=learning.correct(Gaze(0,100,100),measurement,100)
            expected,_=adaptive_prediction(measurement.features,measurement.point,learning.field_state(),True)
            np.testing.assert_allclose([point.x,point.y],expected)

class SimpleProfile:
    def predict(self,x):return np.array([-100,100])
    def supports(self,x):return True

if __name__=='__main__':unittest.main()
