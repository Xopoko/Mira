import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from PySide6.QtWidgets import QApplication
from context_operator.core.contracts import Rect,Element,Scene,Focus,Gaze,GazeMeasurement
from context_operator.controller import Controller
from context_operator.providers.learning import GazeLearning
from context_operator.providers.personal_model import DirectModel
from context_operator.providers.gaze import PhoneGaze
from context_operator.foundations.gaze.arkit import Sample
from context_operator.foundations.gaze.smoothing import Smoother
from context_operator.settings import DEFAULTS

APP=QApplication.instance() or QApplication([])

def sample(now,distance=.5,x=350,y=350,supported=True):
    features=np.zeros(22);features[6]=-distance
    return GazeMeasurement(now,tuple(features),(x,y),supported)

class SavedLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'learning.json'
        self.bounds=Rect(0,0,1000,700);self.scale=np.full(22,.05)
        self.m=GazeLearning(self.path);self.m.bind('original',self.bounds,self.scale)
    def tearDown(self):self.temp.cleanup()
    def test_binding_another_profile_invalidates_late_training_result(self):
        epoch=self.m.trainer.epoch;self.m.bind('new',self.bounds,self.scale)
        self.m.trainer.results.put((epoch,{'model':None,'accepted':True},''));self.m.poll_training()
        self.assertIsNone(self.m.trainer.report);self.assertEqual(self.m.rows,[])
    def test_training_report_save_failure_retries_without_losing_result(self):
        self.m.trainer.results.put((self.m.trainer.epoch,{'model':None,'accepted':False},''))
        with patch.object(self.m,'save',side_effect=OSError('disk')):
            with patch('context_operator.providers.learning.time.monotonic',return_value=100):
                with self.assertRaises(OSError):self.m.poll_training()
        self.assertFalse(self.m.trainer.report['accepted'])
        with patch.object(self.m,'save',wraps=self.m.save) as save:
            with patch('context_operator.providers.learning.time.monotonic',return_value=101):self.m.poll_training()
            save.assert_not_called()
            with patch('context_operator.providers.learning.time.monotonic',return_value=106):self.m.poll_training()
            save.assert_called_once()
        self.assertFalse(json.loads(self.path.read_text())['personal_network']['report']['accepted'])
    def test_promotion_retains_newer_click_residual_and_reload(self):
        old_features=list(sample(0,.4).features)
        new_features=list(sample(0,.8,x=350,y=100).features)
        trained=dict(features=new_features,offset=[20.,0.],target=[450.,100.],count=1,revision=8)
        self.m.click_anchors=[dict(features=old_features,offset=[40.,0.],target=[390.,100.],count=3,revision=7),
                              dict(features=new_features,offset=[35.,0.],target=[450.,100.],count=2,revision=12)]
        self.m.count=12
        data=dict(mean=[0.]*22,scale=[1.]*22,center=[400.,100.],extent=[500.,350.],
                  weights=[np.zeros((22,32)).tolist(),np.zeros((32,16)).tolist(),np.zeros((16,2)).tolist()],
                  biases=[np.zeros(32).tolist(),np.zeros(16).tolist(),np.zeros(2).tolist()],
                  support=[np.zeros(22).tolist()],through_count=8)
        self.m.trainer.pending=DirectModel(data);self.m.trainer.pending_clicks=[trained]
        self.m.trainer.shadow=[dict(cell=cell,candidate=10.,baseline=100.)
                               for cell,count in ((0,4),(1,4),(2,3)) for _ in range(count)]
        self.assertTrue(self.m.learn_click(sample(13,.8,x=350,y=100),(450,100)))
        self.assertTrue(self.m.trainer.report['shadow_accepted'])
        self.assertEqual(len(self.m.click_anchors),1)
        self.assertEqual(self.m.click_anchors[0]['revision'],13)
        self.assertEqual(self.m.click_anchors[0]['offset'],[55.,0.])
        restored=GazeLearning(self.path);restored.bind('original',self.bounds,self.scale)
        self.assertEqual(len(restored.click_anchors),1)
        self.assertEqual(restored.click_anchors[0]['offset'],[55.,0.])
        self.assertAlmostEqual(restored.correct(None,sample(14,.8,x=350,y=100),14).x,455.)

class MouseControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False));self.c.timer.stop()
        self.c.running=True;self.c.win=Mock();self.c.win.emergency.return_value=False
        self.c.win.shift_down.return_value=False
        self.bounds=Rect(0,0,1000,700);self.c.win.rect.return_value=self.bounds;self.c.win.at.return_value=1
        self.a=Element('a','A','Button',Rect(100,100,60,40),1,pid=7)
        self.b=Element('b','B','Button',Rect(800,100,60,40),1,pid=7)
        self.c.scene=Scene(1,100,self.bounds,(self.a,self.b),1,window_rect=self.bounds);self.c.focus=Focus(self.a)
        self.c.learning=GazeLearning(Path(self.temp.name)/'learn.json');self.c.learning.bind('profile',self.bounds,np.full(22,.05))
        self.c.phone=Mock();self.c.phone.measurement=None;self.c.phone.poll.return_value=(Gaze(130,120,100,80),'ready',60)
        self.c.win.cursor.return_value=(830,120);self.c.output=Mock()
    def tearDown(self):self.c.close();self.temp.cleanup()
    def tick(self,now,pressed=False):
        self.c.win.shift_down.return_value=pressed
        self.c.phone.poll.return_value=(Gaze(130,120,now,80),'ready',60)
        with patch('context_operator.controller.time.monotonic',return_value=now):self.c.tick()
    def test_shift_follows_mouse_and_release_returns_to_eyes(self):
        self.tick(100,True);self.assertEqual(self.c.focus.target,self.b);self.assertEqual(self.c.gaze.x,130)
        self.c.win.cursor.return_value=(130,120);self.tick(100.1,True);self.assertEqual(self.c.focus.target,self.a)
        self.c.win.cursor.return_value=(830,120);self.tick(100.2,True);self.assertEqual(self.c.focus.target,self.b)
        self.tick(100.3,False);self.assertEqual(self.c.focus.target,self.a);self.assertEqual(self.c.attention,self.c.gaze)
    def test_mouse_on_empty_space_other_window_or_stale_scene_has_no_target(self):
        self.c.win.cursor.return_value=(500,500);self.tick(100,True);self.assertIsNone(self.c.focus.target)
        self.c.win.cursor.return_value=(830,120);self.c.win.at.return_value=2;self.tick(100.1,True);self.assertIsNone(self.c.focus.target)
        self.c.win.at.return_value=1;self.tick(104,True);self.assertIsNone(self.c.focus.target)
    def test_cursor_dwell_does_not_click_and_pause_clears_attention(self):
        self.c.settings['dwell_enabled']=True;self.c.command=Mock();self.tick(100,True);self.tick(102,True)
        self.c.command.assert_not_called();self.c.pause();self.assertIsNone(self.c.attention)
    def test_shift_voice_uses_mouse_at_phrase_onset(self):
        self.tick(100,True);self.c.microphone=True;self.c.voice_generation=9
        with patch('context_operator.controller.time.monotonic',return_value=100):
            self.c._speech({'type':'speech_start','utteranceId':1},9)
        self.tick(100.1,False);self.c.command=Mock()
        self.c._speech({'type':'final','utteranceId':1,'text':'click'},9)
        args=self.c.command.call_args.args;self.assertEqual(args[1].target,self.b);self.assertEqual(args[2].source,'mouse')

class MeasurementTests(unittest.TestCase):
    def test_out_of_base_support_keeps_numeric_sample_but_blink_drops_it(self):
        phone=PhoneGaze.__new__(PhoneGaze);phone.worker=Mock();phone.profile=Mock()
        phone.profile.supports.return_value=False;phone.profile.predict.return_value=np.array([200,300])
        phone.smoother=Smoother();phone.last_seq=0
        phone.worker.latest.return_value=Sample(seq=1,captured=100,features=np.zeros(22))
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):
            gaze,_,_=phone.poll(230);self.assertIsNone(gaze);self.assertIsNotNone(phone.measurement)
            phone.worker.latest.return_value.blink=True;phone.poll(230);self.assertIsNone(phone.measurement)
            self.assertEqual(phone.invalid_captured,100)

if __name__=='__main__':unittest.main()
