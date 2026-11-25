import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from PySide6.QtWidgets import QApplication
from context_operator.core.click_learning import ClickHistory,HumanClick
from context_operator.core.contracts import GazeMeasurement,Gaze,Rect,Element,Focus
from context_operator.providers.clicks import ClickFilter
from context_operator.providers.learning import GazeLearning
from context_operator.providers.personal_model import adaptive_prediction,train_candidate,PersonalTrainer
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS

APP=QApplication.instance() or QApplication([])

def sample(stamp,head=0,point=(25,12),supported=True):
    x=np.zeros(22);x[4:13]=head
    return GazeMeasurement(stamp,tuple(x),point,supported)

class FakeClicks:
    def __init__(self):self.events=[];self.error='';self.started=False
    def start(self):self.started=True
    def stop(self):self.started=False;self.events=[]
    def drain(self):events,self.events=self.events,[];return events
    def drain_motion(self):return []

class ClickInputTests(unittest.TestCase):
    def test_physical_click_preserves_down_point_and_time(self):
        f=ClickFilter();f.feed(0x201,21,72,10)
        self.assertEqual(f.feed(0x202,23,73,10.1),HumanClick(21,72,10,10.1))
        self.assertIsNone(f.feed(0x202,23,73,10.2))

    def test_injected_own_window_drag_and_long_hold_are_not_labels(self):
        for case in ('injected-down','injected-up','own-down','own-up','drag','long'):
            with self.subTest(case=case):
                f=ClickFilter();f.feed(0x201,20,70,10,injected=case=='injected-down',own_window=case=='own-down')
                if case=='drag':f.feed(0x200,80,70,10.1);f.feed(0x200,20,70,10.15)
                self.assertIsNone(f.feed(0x202,20,70,12 if case=='long' else 10.2,
                                        injected=case=='injected-up',own_window=case=='own-up'))

    def test_match_uses_pre_press_features_not_post_click_look(self):
        h=ClickHistory();before=sample(9.94);middle=sample(9.98);latest=sample(10.00);after=sample(10.03,head=5)
        for item in (before,middle,latest,after):h.observe(item,10.04)
        self.assertIs(h.match(HumanClick(30,18,10.01,10.12),10.13).sample,latest)

    def test_blink_stale_future_and_delayed_events_do_not_teach(self):
        h=ClickHistory();h.observe(sample(9.94),10);h.observe(sample(9.98),10);h.observe(None,10.04)
        self.assertIsNone(h.match(HumanClick(25,15,10.06,10.12),10.13))
        h.clear();h.observe(sample(11),10)
        self.assertIsNone(h.match(HumanClick(25,15,10,10.1),10.12))
        h.clear();h.observe(sample(10),10)
        self.assertIsNone(h.match(HumanClick(25,15,10.2,10.3),10.31))
        self.assertIsNone(h.match(HumanClick(25,15,10.03,10.1),11))

    def test_source_timed_blink_before_press_blocks_but_after_press_does_not(self):
        h=ClickHistory();s=sample(9.98);h.observe(sample(9.94),10);h.observe(s,10)
        h.observe(None,10.05,missing_at=10)
        self.assertIsNone(h.match(HumanClick(25,15,10.02,10.08),10.10))
        h.clear();h.observe(sample(9.90),10);h.observe(sample(9.94),10);h.observe(s,10);h.observe(None,10.05,missing_at=10.03)
        self.assertIs(h.match(HumanClick(25,15,10.02,10.08),10.10).sample,s)

    def test_far_click_is_not_treated_as_gaze_ground_truth(self):
        h=ClickHistory()
        for stamp in (9.92,9.96,10.00):h.observe(sample(stamp),10.02)
        self.assertIsNone(h.match(HumanClick(25,600,10.01,10.1),10.12,Rect(0,0,1000,700)))
        self.assertIn('far from your gaze',h.last_rejection)

    def test_duplicate_samples_unsupported_gaze_and_moving_fixation_are_rejected(self):
        h=ClickHistory();s=sample(9.94);h.observe(s,10);h.observe(s,10);h.observe(sample(10),10)
        self.assertIsNone(h.match(HumanClick(25,15,10.01,10.1),10.12))
        h.clear()
        for stamp in (9.92,9.96,10.00):h.observe(sample(stamp,supported=False),10.02)
        self.assertIsNone(h.match(HumanClick(25,15,10.01,10.1),10.12))
        h.clear()
        for stamp,point in ((9.92,(0,0)),(9.96,(100,100)),(10.00,(0,0))):h.observe(sample(stamp,point=point),10.02)
        self.assertIsNone(h.match(HumanClick(25,15,10.01,10.1),10.12))

    def test_short_prepress_fixation_survives_eye_departure_but_not_opposite_departure(self):
        h=ClickHistory()
        for stamp in (9.70,9.75,9.80):h.observe(sample(stamp,point=(200,12)),stamp+.01)
        h.observe(sample(9.90,point=(280,12)),9.91)
        match=h.match(HumanClick(325,12,10,10.08),10.10,Rect(0,0,1000,700),150)
        self.assertIsNotNone(match);self.assertEqual(match.sample.captured,9.80)
        self.assertEqual(match.limit,300)
        self.assertIsNone(h.match(HumanClick(120,12,10,10.08),10.10,Rect(0,0,1000,700),150))
        self.assertEqual(h.last_rejection_code,'away')

    def test_source_radius_and_short_screen_edge_bound_label_distance(self):
        h=ClickHistory()
        for stamp in (9.92,9.96,10.00):h.observe(sample(stamp),stamp+.01)
        self.assertIsNotNone(h.match(HumanClick(275,12,10.01,10.10),10.12,Rect(0,0,1000,700),150))
        self.assertIsNone(h.match(HumanClick(350,12,10.01,10.10),10.12,Rect(0,0,1000,700),150))
        self.assertEqual(h.last_rejection_code,'away')

class HardCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'learning.json'
        self.m=GazeLearning(self.path);self.m.bind('original',Rect(0,0,1000,700),np.ones(22))
    def tearDown(self):self.temp.cleanup()

    def test_click_advances_toward_target_by_bounded_step(self):
        before=self.m.correct(Gaze(25,12,10),sample(10),10)
        self.assertTrue(self.m.learn_click(sample(10),(25,72)))
        result=self.m.correct(Gaze(35,2,10),sample(10),10)
        self.assertGreater(result.y,before.y)
        self.assertLessEqual(np.linalg.norm(np.asarray([result.x-before.x,result.y-before.y])),40.01)
        self.assertNotEqual((result.x,result.y),(25,72))
        self.assertFalse(self.m.rows)
        # A network trained through this click does not apply it twice.
        model=SimpleNamespace(predict=lambda x:np.array([35,2]),data={'through_count':10000})
        predicted,route=adaptive_prediction(sample(10).features,sample(10).point,self.m.field_state(),True,model)
        self.assertEqual(route,'network');np.testing.assert_allclose(predicted,[35,2])

    def test_repeat_grows_vote_undo_is_one_click_and_reload_keeps_it(self):
        self.m.learn_click(sample(10),(25,72));self.m.learn_click(sample(11),(30,90))
        self.assertEqual(len(self.m.click_anchors),1)
        self.assertGreater(adaptive_prediction(sample(12).features,(25,12),self.m.field_state())[0][1],52)
        self.m.undo_last();self.assertEqual(self.m.count,1)
        other=GazeLearning(self.path);other.bind('original',self.m.bounds,np.ones(22))
        np.testing.assert_allclose(other.correct(None,sample(12),12).x,25)
        self.assertEqual(other.correct(None,sample(12),12).y,52)
        self.assertEqual(len(other.rows),0)

    def test_conflicting_vote_is_rejected_and_three_agreeing_votes_enter_network_rows(self):
        s=sample(10)
        self.assertTrue(self.m.learn_click(s,(25,72)))
        before=json.dumps(self.m.field_state())
        self.assertFalse(self.m.learn_click(s,(25,0)))
        self.assertEqual(self.m.last_click_result,'conflict')
        self.assertEqual(json.dumps(self.m.field_state()),before)
        self.assertTrue(self.m.learn_click(s,(25,72)))
        self.assertTrue(self.m.learn_click(s,(25,72)))
        self.assertEqual(self.m.click_anchors[0]['count'],3)
        self.assertEqual(len(self.m.rows),1)

    def test_unsupported_and_far_clicks_do_not_learn_or_shift_unknown_pose(self):
        s=sample(10,point=(2500,-200),supported=False)
        self.assertFalse(self.m.learn_click(s,(25,72)))
        self.assertFalse(self.m.learn_click(sample(10),(900,600)))
        self.assertIsNone(self.m.correct(None,s,10))
        other=sample(11,head=3,point=(500,400))
        predicted,route=adaptive_prediction(other.features,other.point,self.m.field_state())
        np.testing.assert_allclose(predicted,[500,400]);self.assertEqual(route,'base')
        self.assertIsNone(self.m.correct(None,sample(11,head=3,supported=False),11))

    def test_save_failure_does_not_change_live_model_or_undo(self):
        self.m.learn_click(sample(10),(25,72));before=self.path.read_bytes();field=json.dumps(self.m.field_state())
        with patch.object(self.m,'save',side_effect=OSError('disk')):
            with self.assertRaises(OSError):self.m.learn_click(sample(11),(80,80))
        self.assertEqual(self.path.read_bytes(),before);self.assertEqual(json.dumps(self.m.field_state()),field)
        self.assertEqual(self.m.count,1);self.assertEqual(len(self.m.undo),1)

        damaged = b'{"version":'
        self.path.write_bytes(damaged)
        with self.assertRaisesRegex(OSError, 'Saved gaze corrections are damaged'):
            self.m.learn_click(sample(11), (80, 80))
        self.assertEqual(self.path.read_bytes(), damaged)
        self.assertEqual(json.dumps(self.m.field_state()), field)
        self.assertEqual(self.m.count, 1)
        self.assertEqual(len(self.m.undo), 1)

    def test_invalid_saved_network_does_not_apply_other_corrections(self):
        self.m.learn_click(sample(10),(25,72));data=json.loads(self.path.read_text())
        data['feature_scale']=[2.]*22
        data['personal_network']['pending']={'mean': [0.]}
        self.path.write_text(json.dumps(data))
        self.m.bind('original',self.m.bounds,np.ones(22))
        self.assertEqual((self.m.count,self.m.click_accepted),(0,0))
        self.assertFalse(self.m.click_anchors)
        self.assertIsNone(self.m.trainer.pending)
        np.testing.assert_array_equal(self.m.feature_scale,np.ones(22))

    def test_failed_undo_preserves_disk_live_model_and_retry(self):
        self.m.learn_click(sample(10),(25,72));before=self.path.read_bytes()
        with patch.object(self.m,'save',side_effect=OSError('disk')):
            with self.assertRaises(OSError):self.m.undo_last()
        self.assertEqual(self.m.count,1);self.assertEqual(len(self.m.undo),1)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(self.m.correct(None,sample(11),11).y,52)
        self.m.undo_last();self.assertEqual(self.m.count,0);self.assertFalse(self.m.click_anchors)

class NetworkAcceptanceTests(unittest.TestCase):
    def test_heldout_click_anchors_do_not_make_cv_baseline_zero(self):
        bounds=[0,0,1000,700];rows=[];clicks=[]
        for area in range(6):
            target=[100+200*(area%5),100+230*(area//5)]
            features=[0.]*22;features[4]=area
            base=[target[0]+100,target[1]]
            clicks.append(dict(features=features,target=target,offset=[-100,0],count=3,revision=area+1))
            for _ in range(8):
                rows.append(dict(features=features,target=target,base=base,baseline=base,weight=1.))
        field=dict(clicks=clicks,scale=[1.]*22,bounds=bounds)
        fake=SimpleNamespace(predict=lambda x:np.zeros(2),data={})
        with patch('context_operator.providers.personal_model.fit_network',return_value=fake):
            report=train_candidate(rows,(np.empty((0,22)),np.empty((0,2))),bounds,field=field)
        self.assertEqual(report['baseline_median_px'],100)
        self.assertFalse(report['accepted'])

    def test_candidate_waits_for_future_labels_in_three_cells(self):
        trainer=PersonalTrainer();trainer.pending=SimpleNamespace(predict=lambda x:np.asarray(x[:2])+10)
        bounds=[0,0,1000,700]
        for area in range(3):
            target=[100+200*area,100]
            for _ in range(4):
                features=target+[0.]*20
                trainer.observe_candidate(features,target,[target[0]+100,target[1]+100],bounds)
                if area<2:self.assertIsNone(trainer.model)
        self.assertIsNotNone(trainer.model)
        self.assertIsNone(trainer.pending)
        self.assertTrue(trainer.report['shadow_accepted'])

class ClickControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.source=FakeClicks()
        self.c=Controller(dict(DEFAULTS,vision=False,ranker=False,anticipation=False),click_source=self.source);self.c.timer.stop()
        self.c.running=True;self.c.win=Mock();self.c.win.emergency.return_value=False;self.c.win.shift_down.return_value=False
        self.c.learning=GazeLearning(Path(self.temp.name)/'learning.json')
        self.c.learning.bind('profile',Rect(0,0,1000,700),np.ones(22))
        self.c.phone=Mock(invalid_captured=None,validation_radius=230.);self.c.scene=None;self.c.output=Mock()
    def tearDown(self):self.c.close();self.temp.cleanup()
    def tick(self,now,s):
        self.c.phone.measurement=s;self.c.phone.poll.return_value=(Gaze(25,12,now),'ready',60)
        with patch('context_operator.controller.time.monotonic',return_value=now):self.c.tick()
    def test_normal_mode_teaches_near_click_after_stable_fixation(self):
        self.tick(9.94,sample(9.92));self.tick(9.98,sample(9.96));self.tick(10.01,sample(9.99))
        self.source.events=[HumanClick(30,18,10.02,10.1)]
        self.tick(10.12,sample(10.11))
        self.assertEqual(self.c.learning.count,1);self.assertEqual(self.c.gaze.y,18)
        self.assertFalse(self.c.mouse_control);self.c.output.execute.assert_not_called()
        self.tick(10.15,sample(10.14));self.assertEqual(self.c.learning.count,1)

    def test_covert_attention_far_click_does_not_poison_gaze_learning(self):
        for now,stamp in ((9.94,9.92),(9.98,9.96),(10.01,9.99)):self.tick(now,sample(stamp))
        self.source.events=[HumanClick(30,600,10.02,10.1)]
        self.tick(10.12,sample(10.11))
        self.assertEqual(self.c.learning.count,0)
        self.assertIn('far from your gaze',self.c.click_status)
    def test_rejected_correction_reports_learning_reason_and_counts(self):
        for now,stamp in ((9.94,9.92),(9.98,9.96),(10.01,9.99)):self.tick(now,sample(stamp))
        click=HumanClick(30,18,10.02,10.1)
        with patch.object(self.c.learning,'learn_click',return_value=False):
            for result,reason,counts in (
                ('conflict','it disagrees with earlier clicks here',(0,1)),
                ('away','too far from the corrected gaze',(1,1)),
                ('invalid','the gaze reading wasn’t usable',(1,2)),
            ):
                with self.subTest(result=result):
                    self.c.learning.last_click_result=result
                    self.c._learn_clicks(sample(10.11),10.12,[click])
                    self.assertEqual(self.c.click_status,'Click skipped · '+reason)
                    self.assertEqual((self.c.click_counts['away'],self.c.click_counts['quality']),counts)
        self.assertEqual(self.c.click_counts['accepted'],0)

    def test_disabled_and_pause_discard_events_and_pre_click_history(self):
        self.tick(10,sample(9.98));self.source.events=[HumanClick(25,72,10.01,10.1)]
        self.c.settings['click_learning']=False;self.tick(10.12,sample(10.11));self.assertEqual(self.c.learning.count,0)
        self.c.settings['click_learning']=True;self.c.pause();self.assertEqual(len(self.c.click_history.samples),0)
        self.assertFalse(self.source.started);self.assertEqual(self.c.learning.count,0)
    def test_captured_blink_barrier_reaches_controller(self):
        for start, blink in ((10., True), (11., False)):
            with self.subTest(blink=blink):
                self.c.click_history.clear()
                for offset in (.08, .04, .01):
                    captured = start - offset
                    self.tick(captured + .005, sample(captured))
                self.c.phone.invalid_captured = start if blink else None
                # The missing observation arrives after the click was pressed.
                self.tick(start + .05, None)
                self.c.phone.invalid_captured = None
                self.source.events = [HumanClick(25, 72, start + .02, start + .08)]
                self.tick(start + .10, sample(start + .09))
                self.assertEqual(self.c.learning.count, 0 if blink else 1)
    def test_learning_store_undo_keeps_selected_store(self):
        self.c.learning.learn_click(sample(10),(25,72));self.c.undo_learning()
        self.assertEqual(self.c.learning.count,0);self.assertFalse(self.c.learning.click_anchors)
        self.assertEqual(self.c.click_status,'Last correction undone')

if __name__=='__main__':unittest.main()
