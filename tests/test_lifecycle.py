import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import json
import tempfile
import unittest
import time
import threading
from pathlib import Path
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from context_operator.controller import Controller
from context_operator.settings import DEFAULTS
from context_operator.core.contracts import *

APP=QApplication.instance() or QApplication([])

def drain():
    for _ in range(20):APP.processEvents();time.sleep(.005)

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(dict(DEFAULTS,vision=False,click_learning=False,anticipation=False));self.c.timer.stop()
        self.c.running=True;self.c.generation=4
        self.el=Element('button','Plus','Button',Rect(0,0,80,40),1,pid=99)
        self.scene=Scene(2,time.monotonic(),Rect(0,0,1000,800),(self.el,),1,'Test scene',1,Rect(0,0,900,700))
        self.c.scene=self.scene;self.c.output=Mock()
    def tearDown(self):self.c.close()
    def test_old_rgb_settings_normalize_in_memory_without_private_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw={'gaze_mode':'mobilegaze','truedepth_profile':'motion','hybrid_enabled':True,'hybrid_use':True}
            target=root/'settings.json';target.write_text(json.dumps(raw),encoding='utf-8')
            with patch('context_operator.settings.LOCAL',root):
                from context_operator.settings import load
                loaded=load()
            self.assertEqual(loaded['gaze_mode'],'truedepth');self.assertEqual(loaded['truedepth_profile'],'motion')
            self.assertNotIn('hybrid_enabled',loaded);self.assertNotIn('hybrid_use',loaded)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')),raw)
    def test_late_model_result_after_stop_does_not_act(self):
        self.c.pause();self.c._decision(Intent('click',target_id='button'),self.scene,4,time.monotonic())
        self.c.output.execute.assert_not_called()
    def test_personal_profile_binds_its_own_learning_store(self):
        self.c.learning=Mock(path=self.c.learning.path);self.c.settings['truedepth_profile']='personal'
        path=Path('personal-learning.json')
        with patch('context_operator.controller.PhoneGaze') as provider, \
             patch('context_operator.controller.phone_learning_path',return_value=path):
            provider.return_value.profile_key='personal-profile-key'
            provider.return_value.feature_scale=[1]*22
            self.c.toggle_phone()
            provider.assert_called_once()
        self.assertEqual(self.c.learning.path,path)
        self.assertEqual(self.c.learning.key,'personal-profile-key')
        self.assertEqual(self.c.learning.feature_scale.tolist(),[1]*22)
    def test_failed_phone_binding_stops_the_acquired_worker(self):
        path=Path('existing-learning.json');self.c.learning=Mock(path=path)
        self.c.learning.bind.side_effect=OSError('disk')
        self.c.start=Mock();self.c._start_clicks=Mock()
        with patch('context_operator.controller.PhoneGaze') as provider, \
             patch('context_operator.controller.phone_learning_path',return_value=path):
            provider.return_value.profile_key='profile-key';provider.return_value.feature_scale=[1]*22
            self.c.toggle_phone()
        provider.return_value.start.assert_called_once();provider.return_value.stop.assert_called_once()
        self.assertIsNone(self.c.phone)
    def test_missing_profile_gives_setup_guidance_without_opening_it(self):
        with patch('context_operator.controller.PhoneGaze',side_effect=FileNotFoundError('private calibration file')):
            self.c.toggle_phone()
        self.assertIsNone(self.c.phone);self.assertIsNone(self.c.setup_close)
        self.assertEqual(self.c.status,'Set up your gaze in Profile first.')
    def test_profile_save_failure_restores_setup_hook_and_success_closes_once(self):
        import numpy as np
        close=Mock();self.c.setup_close=close
        path=Path('accepted-personal-adaptation.json')
        for save_effect in (OSError('disk'),None):
            with patch('context_operator.controller.PhoneGaze') as provider, \
                 patch('context_operator.controller.phone_learning_path',return_value=path), \
                 patch('context_operator.settings.save',side_effect=save_effect):
                provider.return_value.profile_key='personal-profile-key'
                provider.return_value.feature_scale=[1]*22
                provider.return_value.profile.arrays={'training_x':np.empty((0,22)),
                                                       'training_y':np.empty((0,2))}
                if save_effect:
                    with self.assertRaises(OSError):self.c.set_truedepth_profile('personal')
                    self.assertIs(self.c.setup_close,close);close.assert_not_called()
                else:
                    self.c.set_truedepth_profile('personal')
        close.assert_called_once();self.assertIsNone(self.c.setup_close)
        self.assertEqual(self.c.settings['truedepth_profile'],'personal')
    def test_pause_closes_guided_setup_exactly_once(self):
        close=Mock();self.c.setup_close=close
        self.c.pause();self.c.pause()
        close.assert_called_once();self.assertIsNone(self.c.setup_close)
    def test_fast_named_command_skips_context_model(self):
        self.c.model_factory=Mock();self.c.output.execute.return_value={'target':'Plus'}
        self.c.command('please open Plus');drain()
        self.c.model_factory.assert_not_called();self.c.output.execute.assert_called_once()
        self.assertIsNone(self.c.ranker)

    def test_complex_command_uses_scores_only_after_background_preparation(self):
        self.c.settings.update(fast_voice=False, ranker=True)
        release = threading.Event()
        ranker = Mock(is_ready=False)
        def warm():
            if release.wait(3):
                ranker.is_ready = True
        ranker.warm.side_effect = warm
        ranker.rank.return_value = {'button': 2.5}
        ranker.close.return_value = threading.Event()
        interpreter = Mock(return_value=Intent('clarify', reply='Say the element name'))
        self.c.model_factory = Mock(return_value=Mock(interpret=interpreter))
        with patch('context_operator.controller.Ranker', return_value=ranker) as create:
            try:
                self.c.command('Find the right element')
                drain()
                self.assertEqual(interpreter.call_args.args[-1], {})
                ranker.rank.assert_not_called()
            finally:
                release.set()
                self.c._ranker_warmup.join(1)
            self.c.command('Find the right element')
            drain()
            self.assertEqual(interpreter.call_args.args[-1], {'button': 2.5})
            ranker.warm.assert_called_once()
            self.c.pause()
            ranker.close.assert_called_once()
            # A new session can interpret commands while the old worker exits.
            self.c.running = True
            self.c.scene = self.scene
            self.c.command('Find the right element')
            drain()
            self.assertEqual(interpreter.call_args.args[-1], {})
            create.assert_called_once()
    def test_missing_target_does_not_act(self):
        self.c._decision(Intent('click',target_id='gone'),self.scene,4,time.monotonic())
        self.c.output.execute.assert_not_called()
    def test_moved_window_invalidates_command(self):
        from dataclasses import replace
        self.c.scene=replace(self.scene,window_rect=Rect(10,0,900,700))
        self.c._decision(Intent('click',target_id='button'),self.scene,4,time.monotonic())
        self.c.output.execute.assert_not_called()
    def test_dispatch_once_invalidates_scene(self):
        self.c.output.execute.return_value={'target':'Plus'}
        self.c._decision(Intent('click',target_id='button'),self.scene,4,time.monotonic())
        drain();self.c.output.execute.assert_called_once();self.assertIsNone(self.c.scene)
    def test_inflight_output_survives_pause_and_serializes_restart(self):
        entered=threading.Event();release=threading.Event();messages=[]
        self.c.message.connect(lambda title,body:messages.append(title))
        def execute(*args):
            entered.set();release.wait(2);return {'target':'Plus'}
        self.c.output.execute.side_effect=execute
        self.c._decision(Intent('click',target_id='button'),self.scene,4,time.monotonic())
        self.assertTrue(entered.wait(1))
        try:
            self.c.pause();self.assertTrue(self.c.output_pending)
            # Restart state without launching real screen/model providers.
            self.c.running=True;self.c.generation+=1;self.c.scene=self.scene
            epoch=self.c.action_epoch
            self.c._decision(Intent('click',target_id='button'),self.scene,self.c.generation,time.monotonic())
            self.c.output.execute.assert_called_once()
        finally:release.set()
        drain()
        self.assertFalse(self.c.output_pending);self.assertFalse(self.c.busy)
        self.assertIsNone(self.c.scene);self.assertGreater(self.c.action_epoch,epoch)
        self.assertIn('Windows reported the action after control stopped',messages)
    def test_pause_closes_all_input_sources(self):
        phone=Mock();self.c.phone=phone;self.c.speech=Mock();self.c.microphone=True
        self.c.pause();phone.stop.assert_called_once();self.c.speech.stop.assert_called_once()
        self.assertIsNone(self.c.gaze);self.assertFalse(self.c.microphone);self.assertFalse(self.c.running)

    def test_phone_disconnect_revokes_latched_and_queued_commands(self):
        c=self.c
        phone=Mock();phone.worker.is_alive.return_value=True;c.phone=phone
        now=time.monotonic()
        c.gaze=c.attention=Gaze(35,20,now)
        c._select_gaze_source(now)
        c.focus=Focus(self.el)
        c.microphone=True;c.voice_generation=3
        c._speech({'type':'speech_start','utteranceId':1},3)
        c._remember_activation(now)
        source_epoch,action_epoch=c.gaze_source_epoch,c.action_epoch

        c.webcam=Mock()
        c.toggle_webcam()
        self.assertEqual(c.gaze_source_mode,'iphone')
        c.gaze=None
        c._select_gaze_source(now+.05)
        self.assertIn(1,c.speech_focus)
        self.assertEqual(c.gaze_source_epoch,source_epoch)

        c.toggle_phone()
        phone.stop.assert_called_once()
        self.assertIsNone(c.phone)
        self.assertIsNone(c.gaze)
        self.assertIsNone(c.attention)
        self.assertIsNone(c.focus.target)
        self.assertFalse(c.activation_history)
        self.assertGreater(c.action_epoch,action_epoch)
        with patch.object(c,'command') as command:
            c._speech({'type':'final','utteranceId':1,'text':'click'},3)
            command.assert_not_called()
        with patch('context_operator.controller.threading.Thread') as dispatch:
            c._decision(Intent('click',target_id=self.el.id),self.scene,c.generation,now,source_epoch=source_epoch)
            dispatch.assert_not_called()
    def test_queued_speech_cannot_restart_after_pause(self):
        self.c.voice_generation=3;self.c.microphone=True
        self.c.received_speech.emit({'type':'final','text':'click'},3)
        self.c.pause();self.c._speech({'type':'final','text':'click'},3)
        self.assertFalse(self.c.running)
    def test_speech_without_onset_target_never_uses_later_focus(self):
        self.c.voice_generation=3;self.c.microphone=True
        self.c._speech({'type':'speech_start','utteranceId':1},3)
        self.c.focus=Focus(self.el);self.c.attention=Gaze(35,20,time.monotonic())
        self.c._decision=Mock()
        self.c._speech({'type':'final','utteranceId':1,'text':'click'},3)
        self.c._decision.assert_not_called()
    def test_speech_without_onset_event_never_uses_current_focus(self):
        self.c.voice_generation=3;self.c.microphone=True
        self.c.focus=Focus(self.el);self.c.attention=Gaze(35,20,time.monotonic())
        self.c._decision=Mock()
        self.c._speech({'type':'final','utteranceId':99,'text':'click'},3)
        self.c._decision.assert_not_called()
    def test_speech_does_not_make_stale_onset_gaze_fresh(self):
        self.c.voice_generation=3;self.c.microphone=True
        self.c.focus=Focus(self.el);self.c.attention=Gaze(35,20,time.monotonic()-1)
        self.c._speech({'type':'speech_start','utteranceId':1},3)
        self.c.attention=Gaze(35,20,time.monotonic());self.c._decision=Mock()
        self.c._speech({'type':'final','utteranceId':1,'text':'click'},3)
        self.c._decision.assert_not_called()
    def test_named_speech_still_works_without_onset_gaze(self):
        self.c.voice_generation=3;self.c.microphone=True;self.c._decision=Mock()
        self.c._speech({'type':'speech_start','utteranceId':1},3)
        self.c._speech({'type':'final','utteranceId':1,'text':'open Plus'},3)
        self.assertEqual(self.c._decision.call_args.args[0].target_id,self.el.id)
    def test_speech_keeps_empty_target_even_with_fresh_onset_gaze(self):
        self.c.voice_generation=3;self.c.microphone=True;self.c.attention=Gaze(35,20,time.monotonic())
        self.c._speech({'type':'speech_start','utteranceId':1},3)
        self.c.focus=Focus(self.el);self.c._decision=Mock()
        self.c._speech({'type':'final','utteranceId':1,'text':'click'},3)
        self.c._decision.assert_not_called()
    def test_speech_keeps_onset_target_across_refresh_and_gaze_move(self):
        from dataclasses import replace
        self.c.voice_generation=3;self.c.microphone=True
        self.c.focus=Focus(self.el);self.c.attention=Gaze(35,20,time.monotonic())
        other=replace(self.el,id='other',rect=Rect(200,0,80,40))
        fresh=replace(self.el,captured=time.monotonic())
        self.c.scene=replace(self.scene,elements=(fresh,other))
        self.c._speech({'type':'speech_start','utteranceId':1},3)
        self.c.focus=Focus(other);self.c.attention=Gaze(235,20,time.monotonic());self.c._decision=Mock()
        self.c._speech({'type':'final','utteranceId':1,'text':'click'},3)
        self.assertEqual(self.c._decision.call_args.args[0].target_id,self.el.id)
    def test_pause_revokes_output_before_waiting_for_sources(self):
        from context_operator.providers.windows import WindowsOutput
        output=WindowsOutput();entered=threading.Event();release=threading.Event();finished=threading.Event()
        effects=[]
        def execute(*args):
            entered.set()
            try:
                if not release.wait(2):raise RuntimeError('Fixture release timeout')
                if not output.is_active():raise RuntimeError('Control stopped')
                effects.append('sent')
                return {'target':'Plus'}
            finally:finished.set()
        output.execute=execute;self.c.output=output
        self.c._decision(Intent('click',target_id='button'),self.scene,4,time.monotonic())
        self.assertTrue(entered.wait(1))
        def stop_source():
            release.set();self.assertTrue(finished.wait(1))
        self.c.clicks.stop=stop_source
        try:self.c.pause()
        finally:release.set()
        drain();self.assertEqual(effects,[])
    def test_old_vision_result_cannot_return_after_action(self):
        self.c.action_epoch=2
        self.c._vision((self.scene,time.monotonic(),(self.el,),1),4)
        self.assertEqual(self.c.vision_elements,())
    def test_stopping_a_registered_gaze_source_revokes_only_its_target(self):
        class Source:
            def start(self,emit):self.emit=emit
            def stop(self):self.stopped=True
        c=self.c
        source,other=Source(),Source()
        c.register_source('gaze-fixture',source);c.enable_source('gaze-fixture')
        c.register_source('command-fixture',other);c.enable_source('command-fixture')
        now=time.monotonic()
        gaze=Gaze(35,20,now,100)
        emit=source.emit
        emit(InputEvent('gaze',now,gaze))
        c.win=Mock();c.win.emergency.return_value=False;c.win.shift_down.return_value=False
        c.tick()
        self.assertEqual(c.focus.target.id,'button')
        epoch=c.gaze_source_epoch

        c.disable_source('command-fixture')
        c.tick()
        self.assertTrue(other.stopped)
        self.assertIs(c.gaze,gaze)
        self.assertEqual(c.gaze_source_epoch,epoch)

        c.disable_source('gaze-fixture')
        self.assertTrue(source.stopped)
        self.assertIsNone(c.attention)
        self.assertIsNone(c.focus.target)
        self.assertFalse(c.activation_history)
        self.assertGreater(c.gaze_source_epoch,epoch)

        c.enable_source('gaze-fixture')
        emit(InputEvent('gaze',now,gaze))
        self.assertIsNone(c.source_gaze)
    def test_dwell_does_not_repeat_through_scene_refresh(self):
        from dataclasses import replace
        self.c.settings['dwell_enabled']=True;self.c.settings['dwell_ms']=700
        self.c.win=Mock();self.c.win.emergency.return_value=False;self.c.win.shift_down.return_value=False
        self.c.phone=Mock();self.c.phone.measurement=None
        self.c.phone.poll.side_effect=lambda radius:(Gaze(35,20,time.monotonic()),'ready',40)
        self.c.output.execute.return_value={'target':'Plus'}
        now=time.monotonic()
        with patch('context_operator.controller.time.monotonic',return_value=now):self.c.tick()
        # A new control occupying the same rectangle needs its own hold.
        replacement=replace(self.el,label='Different button')
        self.scene=replace(self.scene,elements=(replacement,))
        self.c.scene=self.scene
        with patch('context_operator.controller.time.monotonic',return_value=now+.75):self.c.tick()
        self.c.output.execute.assert_not_called()
        with patch('context_operator.controller.time.monotonic',return_value=now+1.5):self.c.tick()
        drain()
        with patch('context_operator.controller.time.monotonic',return_value=now+1.6):self.c.tick()
        self.c.scene=replace(self.scene,captured=now+1.7)
        with patch('context_operator.controller.time.monotonic',return_value=now+1.7):self.c.tick()
        with patch('context_operator.controller.time.monotonic',return_value=now+2.5):self.c.tick()
        self.c.output.execute.assert_called_once()

if __name__=='__main__':unittest.main()
