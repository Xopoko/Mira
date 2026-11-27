import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import tempfile
import unittest
import time
import threading
from pathlib import Path
from unittest.mock import Mock,patch
from PySide6.QtCore import Qt
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
    def test_late_model_result_after_restart_does_not_dispatch(self):
        generation=self.c.generation;started=time.monotonic()
        self.c.pause()
        with patch('context_operator.controller.threading.Thread') as worker:
            self.c.start()
            self.c._scene(self.scene,self.c.generation,self.c.action_epoch)
            worker.reset_mock()
            self.c._decision(Intent('click',target_id='button'),self.scene,generation,started)
            worker.assert_not_called()
        self.c.output.execute.assert_not_called()
    def test_personal_profile_binds_its_own_learning_store(self):
        self.c.learning=Mock(path=self.c.learning.path)
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
        with tempfile.TemporaryDirectory() as folder, patch('context_operator.providers.gaze.LOCAL',Path(folder)):
            self.c.toggle_phone()
        self.assertIsNone(self.c.phone);self.assertIsNone(self.c.setup_close)
        self.assertEqual(self.c.status,'Set up your gaze first');self.assertTrue(self.c.phone_needs_setup)
    def test_profile_activation_failure_restores_setup_hook_and_success_closes_once(self):
        import numpy as np
        close=Mock();self.c.setup_close=close
        path=Path('accepted-personal-adaptation.json')
        with patch('context_operator.controller.PhoneGaze') as provider, \
             patch('context_operator.controller.phone_learning_path',return_value=path), \
             patch('context_operator.settings.save') as save:
            provider.return_value.profile_key='personal-profile-key'
            provider.return_value.feature_scale=[1]*22
            provider.return_value.profile.arrays={'training_x':np.empty((0,22)),
                                                   'training_y':np.empty((0,2))}
            with patch.object(self.c,'pause',side_effect=OSError('disk')):
                with self.assertRaises(OSError):self.c.activate_profile()
            self.assertIs(self.c.setup_close,close);close.assert_not_called()
            self.c.activate_profile()
            save.assert_not_called()
        close.assert_called_once();self.assertIsNone(self.c.setup_close)
        self.assertFalse(self.c.running)
        self.assertEqual(self.c.learning.key,'personal-profile-key')
        self.assertEqual(self.c.learning.path,path)
        self.assertEqual(self.c.status,'Gaze profile selected. Connect iPhone.')
    def test_pause_closes_guided_setup_exactly_once(self):
        close=Mock();self.c.setup_close=close
        self.c.pause();self.c.pause()
        close.assert_called_once();self.assertIsNone(self.c.setup_close)
    def test_fast_named_command_skips_context_model(self):
        self.c.model_factory=Mock();self.c.output.execute.return_value={'target':'Plus'}
        self.c.command('please open Plus');drain()
        self.c.model_factory.assert_not_called();self.c.output.execute.assert_called_once()
        self.assertIsNone(self.c.ranker)

    def test_screen_map_failure_keeps_active_command_busy(self):
        c = self.c
        c.running = False
        c.settings.update(fast_voice=False, ranker=False)
        fail_scan = threading.Event()
        finish_command = threading.Event()
        command_started = threading.Event()

        def observe():
            fail_scan.wait(3)
            raise OSError('scan failed')

        def interpret(*args):
            command_started.set()
            finish_command.wait(3)
            return Intent('clarify')

        c.screen_factory = Mock(return_value=Mock(observe=observe))
        interpreter = Mock(side_effect=interpret)
        c.model_factory = Mock(return_value=Mock(interpret=interpreter))
        c.start()
        try:
            c.command('Find the right element')
            self.assertTrue(command_started.wait(1))
            fail_scan.set()
            deadline = time.monotonic() + 2
            while c.status != 'Screen map: scan failed' and time.monotonic() < deadline:
                drain()
            self.assertEqual(c.status, 'Screen map: scan failed')
            self.assertTrue(c.busy)
            c.command('Find another element')
            interpreter.assert_called_once()
        finally:
            fail_scan.set()
            finish_command.set()
            drain()

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
        self.assertGreater(c.action_epoch,action_epoch)
        with patch.object(c,'command') as command:
            c._speech({'type':'final','utteranceId':1,'text':'click'},3)
            command.assert_not_called()
        with patch('context_operator.controller.threading.Thread') as dispatch:
            c._decision(Intent('click',target_id=self.el.id),self.scene,c.generation,now,source_epoch=source_epoch)
            dispatch.assert_not_called()
    def test_queued_speech_cannot_restart_after_pause(self):
        c = self.c
        c.voice_generation = 3
        c.microphone = True
        c.command = Mock(wraps=c.command)
        c.start = Mock()
        delivered = Mock()
        c.received_speech.connect(delivered, Qt.ConnectionType.QueuedConnection)
        event = {'type': 'final', 'utteranceId': 1, 'text': 'open Plus'}
        worker = threading.Thread(target=c.received_speech.emit, args=(event, 3))
        worker.start()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        delivered.assert_not_called()
        c.command.assert_not_called()

        c.pause()
        APP.processEvents()
        delivered.assert_called_once_with(event, 3)
        c.start.assert_not_called()
        c.command.assert_not_called()
        c.output.execute.assert_not_called()
        self.assertFalse(c.running)
    def test_spoken_stop_still_pauses_after_gaze_source_changes(self):
        c = self.c
        c.phone = Mock()
        c.microphone = True
        c.voice_generation = 3
        now = time.monotonic()
        c.gaze = c.attention = Gaze(35, 20, now)
        c._select_gaze_source(now)
        c._speech({'type': 'speech_start', 'utteranceId': 1}, 3)

        c.toggle_phone()
        self.assertTrue(c.running)
        c._speech({'type': 'final', 'utteranceId': 1, 'text': 'Stop.'}, 3)

        self.assertFalse(c.running)
        self.assertFalse(c.microphone)
        c.output.execute.assert_not_called()
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
    def test_screen_recognition_reports_startup_failure_once(self):
        from context_operator.foundations.perception.runtime import PerceptionClient
        stopped=threading.Event();notices=Mock()
        self.c.message.connect(notices)
        client=Mock(spec=PerceptionClient)
        client.ready.side_effect=OSError('runtime missing')
        with patch('context_operator.foundations.perception.runtime.PerceptionClient',return_value=client):
            self.c.scene=None
            try:
                self.c._start_vision(stopped,self.c.generation)
                deadline=time.monotonic()+2
                while not notices.called and time.monotonic()<deadline:drain()
                drain();drain()
                notices.assert_called_once_with('Screen recognition','Unavailable: runtime missing')
            finally:
                stopped.set()
                self.c.message.disconnect(notices)

    def test_emergency_stop_closes_iphone_setup_while_paused(self):
        self.c.running=False
        self.c.win=Mock()
        self.c.win.emergency.return_value=True
        close=Mock()
        self.c.setup_close=close
        notices=Mock()
        self.c.message.connect(notices)
        self.c.tick()
        close.assert_called_once()
        self.assertIsNone(self.c.setup_close)
        notices.assert_called_once_with('Stop','Capture and control stopped')

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
        self.c.output.execute.assert_called_once()
        with patch('context_operator.controller.time.monotonic',return_value=now+1.6):self.c.tick()
        with patch('context_operator.controller.time.monotonic',return_value=now+1.7):
            self.c._scene(replace(self.scene,id=3,captured=now+1.7),self.c.generation,self.c.action_epoch)
            self.c.tick()
        with patch('context_operator.controller.time.monotonic',return_value=now+2.5):self.c.tick()
        drain()
        self.c.output.execute.assert_called_once()

if __name__=='__main__':unittest.main()
