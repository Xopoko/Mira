import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import time
import unittest
from dataclasses import replace
from unittest.mock import Mock,patch
from PySide6.QtCore import Qt,QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication,QDialog
from context_operator.controller import Controller
from context_operator.core.contracts import Element,Rect,Scene,Focus,Gaze,InputEvent
from context_operator.settings import DEFAULTS
from context_operator.ui.button_binding import ButtonBindingDialog
from context_operator.providers.button import DeviceKeyEvent

APP=QApplication.instance() or QApplication([])


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(dict(DEFAULTS,vision=False,ranker=False,click_learning=False,anticipation=False),output=Mock())
        self.c.timer.stop();self.c.running=True;self.c.generation=4
        self.now=time.monotonic()
        self.a=Element('a','First','Button',Rect(0,0,80,40),1,pid=77)
        self.b=Element('b','Second','Button',Rect(100,0,80,40),1,pid=77)
        self.c.scene=Scene(2,self.now,Rect(0,0,1000,800),(self.a,self.b),1,window_rect=Rect(0,0,900,700))
        self.c.focus=Focus(self.a);self.c.attention=Gaze(30,20,self.now)
        self.c.active_sources['fixture']=(4,5);self.c.input_sources['fixture']=Mock()
        self.c._remember_activation(self.now)
        self.press=InputEvent('activate',self.now+.01)
        self.history=self.c.activation_history
        self.c._decision=Mock()

    def tearDown(self):self.c.close()

    def activate(self):
        with patch('context_operator.controller.time.monotonic',return_value=self.now+.02):
            self.c._activate_target('fixture',self.press,self.history,5)

    def test_click_uses_highlight_at_press_even_when_focus_moves_before_delivery(self):
        self.c.focus=Focus(self.b);self.c._remember_activation(self.now+.015)
        self.history=self.c.activation_history;self.activate()
        self.assertEqual(self.c._decision.call_args.args[0].target_id,'a')
        self.assertEqual(self.c._decision.call_args.args[0].action,'click')
        self.assertEqual(self.c._decision.call_args.args[0].click_method,'pointer')

    def test_empty_focus_at_press_cannot_use_a_later_target(self):
        self.c.focus=Focus();self.c._remember_activation(self.now+.005)
        self.c.focus=Focus(self.b);self.c._remember_activation(self.now+.015)
        self.history=self.c.activation_history;self.activate();self.c._decision.assert_not_called()

    def test_moved_or_renamed_target_rejected(self):
        for replacement in (replace(self.a,rect=Rect(2,0,80,40)),replace(self.a,label='Delete'),replace(self.a,pid=999)):
            self.c.scene=replace(self.c.scene,elements=(replacement,self.b));self.activate()
        self.c._decision.assert_not_called()

    def test_stale_scene_and_gaze_and_future_press_rejected(self):
        self.c.scene=replace(self.c.scene,captured=self.now-4);self.activate()
        self.c.scene=replace(self.c.scene,captured=self.now)
        self.c.attention=replace(self.c.attention,captured=self.now-1)
        self.c.activation_history=();self.c._remember_activation(self.now);self.history=self.c.activation_history
        self.activate();self.press=InputEvent('activate',self.now+1);self.activate()
        self.c._decision.assert_not_called()

    def test_pause_reactivation_and_changed_action_epoch_reject_queued_press(self):
        self.c.running=False;self.activate();self.c.running=True
        self.c.action_epoch+=1;self.activate();self.c.action_epoch-=1
        self.c.active_sources['fixture']=(4,6);self.activate()
        self.c._decision.assert_not_called()

    def test_inflight_commands_block_press_and_queued_press(self):
        for attr in ('busy','output_pending'):
            setattr(self.c,attr,True)
            self.activate()
            self.c._remember_activation(self.now);self.history=self.c.activation_history
            setattr(self.c,attr,False)
            self.activate();self.c.activation_history=();self.c._remember_activation(self.now);self.history=self.c.activation_history
        self.c._decision.assert_not_called()

    def test_stale_frame_not_retargeted(self):
        self.press=InputEvent('activate',self.now+.2)
        with patch('context_operator.controller.time.monotonic',return_value=self.now+.21):
            self.c._activate_target('fixture',self.press,self.history,5)
        self.c._decision.assert_not_called()

    def test_button_disable_does_not_disconnect_other_gaze_source(self):
        self.c.input_sources['keyboard-button']=Mock();self.c.source_gaze=('input:fixture',self.c.attention)
        self.c.disable_source('keyboard-button');self.assertIsNotNone(self.c.source_gaze)

    def test_slow_device_reader_stop_cannot_prevent_emergency_stop_of_other_sources(self):
        self.c.input_sources['keyboard-button']=Mock()
        self.c.input_sources['keyboard-button'].stop.side_effect=RuntimeError('cleanup pending')
        self.c.active_sources['keyboard-button']=(4,7)
        phone=self.c.phone=Mock();speech=self.c.speech=Mock()
        self.c.pause()
        phone.stop.assert_called_once();speech.stop.assert_called_once()
        self.assertFalse(self.c.running);self.assertNotIn('keyboard-button',self.c.active_sources)

    def test_binding_is_saved_and_source_reenabled_with_new_token(self):
        source=Mock(vk=124,device_id='a'*64,scan_code=100,extended=0);self.c.button_factory=Mock(return_value=source)
        with patch('context_operator.settings.save') as save:
            self.c.set_button(124,'F13','a'*64,100,0)
        save.assert_called_once();source.start.assert_called_once()
        self.c.button_factory.assert_called_once_with(124,'a'*64,100,0)
        callback=source.start.call_args.args[0]
        with patch('context_operator.settings.save'):self.c.set_button_enabled(False)
        source.stop.assert_called_once();callback(self.press)
        self.c._decision.assert_not_called();self.assertNotIn('keyboard-button',self.c.active_sources)

    def test_legacy_key_only_binding_never_starts_or_enables_source(self):
        self.c.settings.update(button_vk=13,button_enabled=True,button_device_id='')
        self.c.button_factory=Mock();self.c._start_button()
        self.c.button_factory.assert_not_called()
        with self.assertRaises(ValueError):self.c.set_button_enabled(True)

    def test_settings_load_disables_legacy_global_mapping(self):
        import tempfile,json
        from pathlib import Path
        from context_operator.settings import load
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'settings.json').write_text(json.dumps({'button_vk':13,'button_enabled':True}),encoding='utf-8')
            with patch('context_operator.settings.LOCAL',root):settings=load()
        self.assertFalse(settings['button_enabled']);self.assertEqual(settings['button_device_id'],'')

    def test_real_dispatch_receives_only_latched_target_and_never_calls_model(self):
        del self.c._decision
        self.c.model_factory=Mock();self.c.output.execute.return_value={'target':'First'}
        self.activate()
        for _ in range(30):APP.processEvents();time.sleep(.002)
        self.c.output.execute.assert_called_once()
        self.assertEqual(self.c.output.execute.call_args.args[1],self.a)
        self.assertEqual(self.c.output.execute.call_args.args[0].click_method,'pointer')
        self.c.model_factory.assert_not_called();self.assertIsNone(self.c.scene)


class BindingDialogTests(unittest.TestCase):
    def dialog(self):
        d=ButtonBindingDialog(observer=Mock());d._listening=True;d.isActiveWindow=Mock(return_value=True)
        self.addCleanup(d.close);return d

    def press(self,d,device='a'*64,vk=13):
        d._on_raw(DeviceKeyEvent(device,vk,28,0,True,time.monotonic()))
        d._on_raw(DeviceKeyEvent(device,vk,28,0,False,time.monotonic()))

    def test_two_physical_devices_with_same_enter_are_distinguished(self):
        d=self.dialog();self.press(d)
        self.assertEqual(d.keyboard_id,'a'*64);self.assertIsNone(d.binding)
        self.press(d);self.assertIsNone(d.binding)
        self.press(d,'b'*64)
        self.assertEqual(d.binding,(13,'Enter','b'*64,28,0));self.assertEqual(d.result(),QDialog.Accepted)
        d.observer.stop.assert_called_once()

    def test_does_not_bind_on_repeat_or_release_from_other_device(self):
        d=self.dialog();self.press(d)
        down=DeviceKeyEvent('b'*64,124,100,0,True,time.monotonic())
        d._on_raw(down);d._on_raw(down)
        self.press(d,'a'*64);self.assertIsNone(d.binding)
        d._on_raw(replace(down,down=False,captured=time.monotonic()))
        self.assertEqual(d.binding,(124,'F13','b'*64,100,0))

    def test_stale_and_inactive_and_unidentified_input_not_enrolled(self):
        d=self.dialog();self.press(d)
        event=DeviceKeyEvent('b'*64,13,28,0,True,time.monotonic()-1)
        d._on_raw(event);self.assertIsNone(d.pending)
        self.press(d,'');self.assertIsNone(d.pending)
        d.isActiveWindow.return_value=False;self.press(d,'b'*64)
        self.assertIsNone(d.binding)

    def test_qt_key_event_alone_cannot_bind_a_global_key(self):
        d=self.dialog()
        d.keyPressEvent(QKeyEvent(QEvent.KeyPress,Qt.Key_Return,Qt.NoModifier,0,13,0))
        d.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease,Qt.Key_Return,Qt.NoModifier,0,13,0))
        self.assertIsNone(d.binding);self.assertIsNone(d.keyboard_id)

    def test_cancel_stops_reader_and_ignores_late_device_events(self):
        d=self.dialog();d.keyPressEvent(QKeyEvent(QEvent.KeyPress,Qt.Key_Escape,Qt.NoModifier))
        self.press(d);self.press(d,'b'*64)
        self.assertIsNone(d.binding);self.assertEqual(d.result(),QDialog.Rejected)


if __name__=='__main__':unittest.main()
