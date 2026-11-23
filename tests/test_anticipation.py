import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage,QPainter
from PySide6.QtCore import QRectF
from context_operator.core.click_learning import PointerMotion,HumanClick
from context_operator.core.contracts import Rect,Element,Scene,Gaze
from context_operator.core.anticipation import Anticipation,IntentPredictor,OnlineRanker,PRIOR,softmax
from context_operator.providers.anticipation import IntentLearning
from context_operator.providers.clicks import ClickFilter
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS
from context_operator.ui.overlay import paint_prediction

APP=QApplication.instance() or QApplication([])
BOUNDS=Rect(0,0,1000,700)
ELEMENTS=(Element('a','Next','Button',Rect(650,200,80,40),10,pid=20),
          Element('b','Other','Button',Rect(650,500,80,40),10,pid=20),
          Element('c','Tab','TabItem',Rect(300,550,90,40),10,pid=20))

def scene(now=10):return Scene(1,now,BOUNDS,ELEMENTS,10,window_rect=BOUNDS)
def move(model,start=10,gaze=True):
    result=None
    for i in range(15):
        now=start+i*.025
        model.observe(PointerMotion(100+i*20,220,now,10,20))
        result=model.predict(scene(now),Gaze(680,220,now,100) if gaze else None,now)
    return result

def fixate(model,start=10,*,supported=True,source='iphone',point=(680,220),scene_factory=scene):
    result=None
    for i in range(12):
        now=start+i*.025
        result=model.predict(scene_factory(now),Gaze(*point,now,100,source),now,gaze_supported=supported)
    return result

class PredictionTests(unittest.TestCase):
    def test_legacy_four_argument_forecast_constructor_gets_mouse_ttl(self):
        forecast=Anticipation(ELEMENTS[0],10.,10.,.7)
        self.assertEqual(forecast.source,'mouse');self.assertEqual(forecast.valid_until,10.18)
        self.assertTrue(forecast.is_fresh(10.17));self.assertFalse(forecast.is_fresh(10.18))

    def test_gaze_dwell_predicts_without_mouse_and_is_timestamped(self):
        p=IntentPredictor();forecast=fixate(p)
        self.assertIsNotNone(forecast);self.assertEqual(forecast.target.id,'a');self.assertEqual(forecast.source,'gaze')
        self.assertEqual(forecast.evidence_at,10.275);self.assertTrue(forecast.is_fresh(10.3))
        self.assertFalse(forecast.is_fresh(forecast.valid_until))
        self.assertEqual(p.model.learned,0);self.assertEqual(p.gaze_model.learned,0)

    def test_duplicate_unsupported_stale_or_mouse_gaze_cannot_create_dwell(self):
        p=IntentPredictor();g=Gaze(680,220,10,100)
        for i in range(12):
            now=10+i*.025;self.assertIsNone(p.predict(scene(now),g,now))
        for supported,source,stamp in ((False,'iphone',11.),(True,'iphone',9.),(True,'mouse',11.)):
            q=IntentPredictor()
            for i in range(12):
                now=11+i*.025;forecast=q.predict(scene(now),Gaze(680,220,stamp,100,source),now,gaze_supported=supported)
                self.assertIsNone(forecast)

    def test_replaced_gaze_target_and_own_process_controls_are_rejected(self):
        p=IntentPredictor();
        for i in range(8):
            now=10+i*.025;p.predict(scene(now),Gaze(680,220,now,100),now)
        replacement=Element('new','New','Edit',ELEMENTS[0].rect,10,pid=20)
        new_scene=lambda now:Scene(2,now,BOUNDS,(replacement,*ELEMENTS[1:]),10)
        self.assertIsNone(p.predict(new_scene(10.175),Gaze(680,220,10.175,100),10.175))
        own=IntentPredictor(own_pid=20)
        self.assertIsNone(fixate(own,scene_factory=lambda now:scene(now)))

    def test_gaze_click_labels_only_gaze_model_and_away_click_trains_mouse_only(self):
        p=IntentPredictor();self.assertIsNotNone(fixate(p))
        p.observe_scene(scene(10.65),10.66)
        self.assertTrue(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
        self.assertEqual((p.model.learned,p.gaze_model.learned,p.gaze_model.hits),(0,1,1))

        q=IntentPredictor();fixate(q)
        move(q,10.3,gaze=False);q.observe_scene(scene(10.65),10.66)
        self.assertTrue(q.feedback(HumanClick(670,520,10.7,10.8,10,20),10.81))
        self.assertEqual(q.model.learned,1);self.assertEqual(q.gaze_model.learned,0)
        self.assertEqual((q.gaze_model.evaluated,q.gaze_model.hits),(1,0))

    def test_gaze_click_after_press_and_duplicate_feedback_do_not_relabel(self):
        p=IntentPredictor();fixate(p);snapshot=p.snapshots[-1]
        self.assertEqual(snapshot['channel'],'gaze')
        p.observe_scene(scene(10.65),10.66)
        p.observe(PointerMotion(670,220,10.7,10,20,'press'))
        self.assertTrue(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
        self.assertFalse(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.82))
        self.assertEqual((p.gaze_model.evaluated,p.gaze_model.hits,p.gaze_model.learned),(1,1,1))

    def test_drag_barrier_requires_three_fresh_legitimate_moves(self):
        p=IntentPredictor();move(p)
        p.observe(PointerMotion(400,220,10.4,10,20,'drag'))
        for i in range(2):
            now=10.5+i*.025;p.observe(PointerMotion(500+i*20,220,now,10,20));self.assertIsNone(p.predict(scene(now),None,now))
        for i in range(5):
            now=10.55+i*.025;p.observe(PointerMotion(540+i*20,220,now,10,20));forecast=p.predict(scene(now),None,now)
        self.assertIsNotNone(forecast);self.assertEqual(forecast.source,'mouse')

    def test_drag_and_injected_motion_block_gaze_until_release_then_new_fixation(self):
        p=IntentPredictor();fixate(p)
        p.observe(PointerMotion(300,220,10.3,10,20,'blocked'))
        for i in range(12):
            now=10.325+i*.025
            self.assertIsNone(p.predict(scene(now),Gaze(680,220,now,100),now))
        p.observe(PointerMotion(300,220,10.65,10,20,'release'))
        self.assertIsNone(p.predict(scene(10.66),Gaze(680,220,10.275,100),10.66))
        forecast=None
        for i in range(12):
            now=10.675+i*.025;forecast=p.predict(scene(now),Gaze(680,220,now,100),now)
        self.assertIsNotNone(forecast);self.assertEqual(forecast.source,'gaze')

    def test_predicts_target_before_arrival_then_learns_only_on_click(self):
        p=IntentPredictor();forecast=move(p)
        self.assertIsNotNone(forecast);self.assertEqual(forecast.target.id,'a');self.assertEqual(p.model.learned,0)
        click=HumanClick(670,220,10.75,10.8,10,20)
        self.assertTrue(p.feedback(click,10.81));self.assertEqual(p.model.learned,1)
        self.assertEqual(p.model.hits,1);self.assertGreaterEqual(p.model.lead_sum_ms,150)
        self.assertIsNone(p.current);self.assertFalse(p.feedback(click,10.82))

    def test_motion_only_works_and_idle_stale_or_future_scene_hides(self):
        p=IntentPredictor();self.assertIsNotNone(move(p,gaze=False))
        self.assertIsNone(p.predict(scene(11),None,11))
        move(p,12);self.assertIsNone(p.predict(scene(10),None,12.36))
        self.assertIsNone(p.predict(scene(13),None,12.36))

    def test_stale_gaze_is_equivalent_to_missing_not_a_target(self):
        p=IntentPredictor();move(p)
        a=p.predict(scene(10.37),Gaze(690,520,8,40),10.37)
        b=p.predict(scene(10.38),None,10.38)
        self.assertEqual(a.target.id,b.target.id)

    def test_other_window_or_pid_cannot_be_the_predicted_or_taught_target(self):
        p=IntentPredictor();move(p)
        self.assertFalse(p.feedback(HumanClick(670,220,10.7,10.8,11,20),10.81))
        self.assertEqual(p.model.learned,0)
        p.reset();p.observe(PointerMotion(100,100,10,11,20));p.observe(PointerMotion(150,100,10.05,11,20));p.observe(PointerMotion(200,100,10.1,11,20))
        self.assertIsNone(p.predict(scene(),None,10.1))

    def test_injected_drag_and_pause_clear_forecasts_press_keeps_prepress_snapshot(self):
        for kind in ('blocked','drag'):
            p=IntentPredictor();move(p);p.observe(PointerMotion(400,220,10.4,10,20,kind))
            self.assertFalse(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
        p=IntentPredictor();move(p);p.observe(PointerMotion(670,220,10.7,10,20,'press'))
        self.assertTrue(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
        move(p,12);p.reset();self.assertFalse(p.snapshots);self.assertEqual(p.model.learned,1)

    def test_late_and_already_arrived_are_not_early_accuracy(self):
        p=IntentPredictor();move(p)
        self.assertFalse(p.feedback(HumanClick(670,220,10.37,10.38,10,20),11))
        p.reset()
        for i in range(8):
            now=20+i*.025;p.observe(PointerMotion(653+i*7,220,now,10,20));p.predict(scene(now),Gaze(680,220,now),now)
        self.assertFalse(p.feedback(HumanClick(705,220,20.4,20.42,10,20),20.43))
        self.assertEqual(p.model.evaluated,0)

    def test_missing_candidate_is_counted_separately_and_not_fabricated(self):
        p=IntentPredictor();move(p)
        self.assertTrue(p.feedback(HumanClick(900,650,10.7,10.8,10,20),10.81))
        self.assertEqual((p.model.evaluated,p.model.unmapped,p.model.learned),(1,1,0))

    def test_prequential_result_is_frozen_before_training(self):
        p=IntentPredictor();move(p);snapshot=p.snapshots[-1]
        label=next(i for i,e in enumerate(snapshot['elements']) if e.id=='b')
        self.assertNotEqual(snapshot['best'],label)
        p.feedback(HumanClick(670,520,10.7,10.8,10,20),10.81)
        self.assertEqual((p.model.evaluated,p.model.hits),(1,0))
        for _ in range(40):p.model.learn(snapshot['x'],label)
        self.assertEqual(np.argmax(p.model.scores(snapshot['x'])),label)
        self.assertEqual(p.model.hits,0)

    def test_replaced_target_before_press_cannot_teach_or_credit_old_object(self):
        p=IntentPredictor();move(p)
        replacement=Element('new','New edit','Edit',ELEMENTS[0].rect,10,pid=20)
        p.observe_scene(Scene(2,10.6,BOUNDS,(replacement,*ELEMENTS[1:]),10),10.61)
        p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81)
        self.assertEqual((p.model.evaluated,p.model.hits,p.model.unmapped,p.model.learned),(1,0,1,0))

    def test_page_change_after_press_and_delayed_arrival_do_not_relabel_click(self):
        for capture in (10.6,10.72):
            p=IntentPredictor();move(p)
            replacement=Element('new','New edit','Edit',ELEMENTS[0].rect,10,pid=20)
            p.observe(PointerMotion(670,220,10.7,10,20,'press'))
            p.predict(Scene(2,capture,BOUNDS,(replacement,*ELEMENTS[1:]),10),None,10.75)
            self.assertTrue(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
            self.assertEqual((p.model.hits,p.model.learned,p.model.last_role),(1,1,'Button'))

    def test_fast_click_trains_without_inflating_early_accuracy(self):
        p=IntentPredictor()
        for i in range(5):
            now=20+i*.025;p.observe(PointerMotion(100+i*50,220,now,10,20));p.predict(scene(now),None,now)
        self.assertTrue(p.feedback(HumanClick(670,220,20.16,20.2,10,20),20.21))
        self.assertEqual((p.model.learned,p.model.evaluated),(1,0))
        p.reset();move(p,22)
        # All forecasts are after this press: no future features may teach the model.
        self.assertFalse(p.feedback(HumanClick(670,220,22.01,22.4,10,20),22.41))
        self.assertEqual(p.model.learned,1)

    def test_click_filter_jitter_freezes_and_real_drag_discards_forecast(self):
        for dx,expected in ((1,True),(7,False)):
            p=IntentPredictor();move(p);f=ClickFilter()
            f.feed(0x201,670,220,10.7);p.observe(PointerMotion(670,220,10.7,10,20,'press'))
            f.feed(0x200,670+dx,220,10.71)
            p.observe(PointerMotion(670+dx,220,10.71,10,20,f.motion_kind({0x201})))
            self.assertIsNone(p.predict(scene(10.72),None,10.72))
            click=f.feed(0x202,670+dx,220,10.8)
            self.assertEqual(click is not None,expected)
            if click:
                self.assertTrue(p.feedback(HumanClick(click.x,click.y,click.pressed,click.released,10,20),10.81))
                self.assertEqual(p.model.learned,1)
            else:self.assertFalse(p.snapshots)

    def test_ambiguous_identical_targets_abstain(self):
        p=IntentPredictor()
        twins=(Element('a','A','Button',Rect(650,190,80,20),10,pid=20),Element('b','B','Button',Rect(650,230,80,20),10,pid=20))
        for i in range(10):
            now=10+i*.025;p.observe(PointerMotion(100+i*20,220,now,10,20))
            self.assertIsNone(p.predict(Scene(1,now,BOUNDS,twins),None,now))

    def test_negative_monitor_origin_preserves_same_target(self):
        p=IntentPredictor();dx=-1800
        elements=tuple(Element(e.id,e.label,e.role,Rect(e.rect.x+dx,e.rect.y,e.rect.w,e.rect.h),10,pid=20) for e in ELEMENTS)
        for i in range(15):
            now=10+i*.025;p.observe(PointerMotion(100+i*20+dx,220,now,10,20))
            result=p.predict(Scene(1,now,Rect(dx,0,1000,700),elements),Gaze(680+dx,220,now,100),now)
        self.assertEqual(result.target.id,'a')

class PersistenceTests(unittest.TestCase):
    def test_replay_numeric_only_reload_and_failure_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';p=IntentLearning(path);move(p)
            p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81)
            data=json.loads(path.read_text());self.assertNotIn('Next',path.read_text());self.assertNotIn('elements',data)
            restored=IntentLearning(path);np.testing.assert_allclose(restored.model.weights,p.model.weights)
            self.assertEqual(restored.model.learned,1);before=path.read_bytes();move(p,12)
            with patch.object(p,'save',side_effect=OSError('disk')):
                self.assertFalse(p.feedback(HumanClick(670,520,12.7,12.8,10,20),12.81))
            self.assertEqual(p.model.learned,1);self.assertEqual(path.read_bytes(),before)
            self.assertIn('Couldn’t save prediction data',p.status)
    def test_invalid_saved_model_does_not_get_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';path.write_text('{bad');p=IntentLearning(path);move(p)
            self.assertFalse(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81));self.assertEqual(path.read_text(),'{bad')

    def test_schema_two_persists_independent_gaze_weights_and_reads_legacy_mouse_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'intent.json';p=IntentLearning(path);fixate(p);p.observe_scene(scene(10.65),10.66)
            self.assertTrue(p.feedback(HumanClick(670,220,10.7,10.8,10,20),10.81))
            data=json.loads(path.read_text());self.assertEqual(data['schema'],2)
            self.assertEqual(data['gaze']['learned'],1);self.assertNotIn('Next',path.read_text())
            restored=IntentLearning(path)
            self.assertEqual(restored.gaze_model.learned,1);np.testing.assert_allclose(restored.gaze_model.weights,p.gaze_model.weights)
            legacy=IntentLearning(Path(tmp)/'legacy.json');legacy.path.write_text(json.dumps(p.model.state()))
            migrated=IntentLearning(legacy.path)
            self.assertEqual(migrated.model.learned,p.model.learned);self.assertEqual(migrated.gaze_model.learned,0)

class FakeSource:
    def __init__(self):self.events=[];self.motions=[];self.started=False;self.error=''
    def start(self):self.started=True
    def stop(self):self.started=False;self.events=[];self.motions=[]
    def drain(self):v,self.events=self.events,[];return v
    def drain_motion(self):v,self.motions=self.motions,[];return v

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.source=FakeSource()
        self.c=Controller(dict(DEFAULTS,vision=False,ranker=False,click_learning=False),click_source=self.source);self.c.timer.stop()
        self.c.anticipation=IntentLearning(Path(self.tmp.name)/'intent.json')
        self.c.running=True;self.c.win=Mock();self.c.win.emergency.return_value=False;self.c.win.shift_down.return_value=False
        self.c.output=Mock()
    def tearDown(self):self.c.close();self.tmp.cleanup()
    def test_mouse_prediction_runs_without_phone_or_gaze_click_learning(self):
        self.c._start_clicks();self.assertTrue(self.source.started)
        for i in range(15):
            now=10+i*.025;self.c.scene=scene(now);self.source.motions=[PointerMotion(100+i*20,220,now,10,20)]
            with patch('context_operator.controller.time.monotonic',return_value=now):self.c.tick()
        self.assertIsNotNone(self.c.prediction);self.c.output.execute.assert_not_called()
        self.source.events=[HumanClick(670,220,10.7,10.8,10,20)]
        with patch('context_operator.controller.time.monotonic',return_value=10.81):self.c.tick()
        self.assertEqual(self.c.anticipation.model.learned,1);self.assertEqual(self.c.learning.count,0)
        self.c.pause();self.assertFalse(self.source.started);self.assertIsNone(self.c.prediction)
    def test_disable_is_persisted_and_does_not_change_gaze_learning(self):
        self.c.learning.count=154
        with patch('context_operator.settings.save') as save:self.c.set_anticipation(False);save.assert_called_once()
        self.assertFalse(self.source.started);self.assertEqual(self.c.learning.count,154)
        with patch('context_operator.settings.save'):self.c.set_anticipation(True)
        self.assertTrue(self.source.started)
    def test_gaze_save_failure_keeps_shared_mouse_source_and_intent_learning(self):
        from context_operator.core.contracts import GazeMeasurement
        self.c.settings['click_learning']=True;self.c.phone=Mock();self.c.learning=Mock()
        self.c.learning.key='profile';self.c.learning.learn_click.side_effect=OSError('disk')
        self.c._start_clicks()
        for stamp in (9.99,10.02,10.05):
            measurement=GazeMeasurement(stamp,tuple([0.]*22),(670,220),True)
            self.c.click_history.observe(measurement,10.06)
        self.c._learn_clicks(None,10.11,[HumanClick(670,220,10.06,10.1,10,20)])
        self.assertFalse(self.c.settings['click_learning']);self.assertTrue(self.source.started)
        for i in range(15):
            now=12+i*.025;self.c.scene=scene(now);self.source.motions=[PointerMotion(100+i*20,220,now,10,20)]
            self.c._predict_intention([],now)
        self.c._predict_intention([HumanClick(670,220,12.7,12.8,10,20)],12.81)
        self.assertEqual(self.c.anticipation.model.learned,1)
    def test_violet_is_weaker_and_does_not_paint_empty_space(self):
        image=QImage(320,180,QImage.Format_ARGB32_Premultiplied);image.fill(0)
        p=QPainter(image);paint_prediction(p,QRectF(80,60,100,50));p.end()
        color=image.pixelColor(130,57)
        self.assertGreater(color.blue(),color.green());self.assertLess(color.alpha(),250)
        self.assertGreater(color.alpha(),150)
        self.assertEqual(image.pixelColor(300,170).alpha(),0)

    def test_click_filter_marks_physical_release_and_rejects_injected_release(self):
        f=ClickFilter()
        self.assertEqual(f.event_motion_kind(0x202,set()),'release')
        self.assertEqual(f.event_motion_kind(0x202,set(),injected=True),'blocked')
        self.assertEqual(f.event_motion_kind(0x204,{0x204}), 'blocked')

if __name__=='__main__':unittest.main()
