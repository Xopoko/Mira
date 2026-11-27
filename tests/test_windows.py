import unittest
import time
from unittest.mock import ANY,Mock,patch
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from context_operator.providers.windows import WindowsOutput
from context_operator.providers.uia_reader import WindowReader
from context_operator.core.contracts import MAP_TTL,Rect,Element,Scene,Intent,WindowMap

class OutputTests(unittest.TestCase):
    def test_ascii_punctuation_is_not_sent_as_unrelated_virtual_keys(self):
        for key in ('[','.', '-', '`', '/', '!'):
            with self.subTest(key=key):
                output=WindowsOutput();output.win=Mock();output.win.shift_down.return_value=False
                scene=Scene(1,time.monotonic(),Rect(0,0,800,600),(),12)
                with self.assertRaisesRegex(RuntimeError,'Unknown key'):
                    output.execute(Intent('press_keys',keys=('Ctrl',key)),None,scene)
                output.win.send.assert_not_called()
    def test_supported_chord_preserves_down_and_reverse_up_order(self):
        output=WindowsOutput();win=output.win;output.win=Mock(wraps=win)
        output.win.u=Mock();output.win.shift_down.return_value=False;output.win.activate=Mock()
        output.win.foreground=Mock(return_value=12)
        output.win.send=Mock();scene=Scene(1,time.monotonic(),Rect(0,0,800,600),(),12)
        output.execute(Intent('press_keys',keys=('Ctrl','A')),None,scene)
        events=output.win.send.call_args.args[0]
        self.assertEqual([(e.ki.vk,e.ki.flags) for e in events],[(17,0),(65,0),(65,2),(17,2)])
    def test_shift_does_not_turn_native_command_into_shift_click(self):
        output=WindowsOutput();output.win=Mock();output.win.shift_down.return_value=False;output.win.shift_down.return_value=True
        output.semantic=Mock(return_value=False)
        scene=Scene(1,time.monotonic(),Rect(0,0,800,600),(),12)
        with self.assertRaisesRegex(RuntimeError,'Release Shift'):
            output.execute(Intent('press_keys',keys=('Tab',)),None,scene)
        output.win.activate.assert_not_called();output.win.send.assert_not_called()
        with self.assertRaisesRegex(RuntimeError,'Release Shift'):output._send([object()])
        output.win.send.assert_not_called()
    def semantic_fixture(self,name='Name',pattern=None):
        output=WindowsOutput();output.win=Mock();output.win.shift_down.return_value=False
        output.win.at.return_value=12
        pattern=pattern or Mock(IsReadOnly=False,Value='')
        control=Mock(Name=name,IsPassword=False,IsEnabled=True)
        control.BoundingRectangle=SimpleNamespace(left=60,top=40,right=210,bottom=80)
        control.Exists.return_value=True;control.GetPattern.return_value=pattern
        def construct(**kwargs):
            self.assertNotIn('Name',kwargs)
            self.assertTrue(kwargs['Compare'](control,1))
            return control
        api=SimpleNamespace(UIAutomationInitializerInThread=nullcontext,EditControl=construct,
            ControlFromHandle=Mock(),PatternId=SimpleNamespace(ValuePattern=1,InvokePattern=2))
        target=Element('edit',name[:180] or 'Edit','Edit',Rect(60,40,150,40),12,pid=7)
        return output,pattern,api,target
    def pointer_click_fixture(self):
        output=WindowsOutput();win=output.win
        # Keep Windows' INPUT constructors real; replace every effectful call.
        win.u=Mock();win.u.IsWindow.return_value=True;win.pid=Mock(return_value=7)
        win.at=Mock(return_value=12);win.foreground=Mock(return_value=12);win.shift_down=Mock(return_value=False)
        win.activate=Mock();win.send=Mock();win.bounds=Mock(return_value=Rect(0,0,1920,1080))
        output.validate_visual=Mock();output.validate_control=Mock()
        output.semantic=Mock(return_value=False)
        target=Element('button','Yellow button','Button',Rect(60,40,150,40),12,pid=7)
        scene=Scene(1,time.monotonic(),Rect(0,0,800,600),(target,),12)
        return output,target,scene
    def test_click_falls_back_to_one_native_mouse_sequence(self):
        output,target,scene=self.pointer_click_fixture()
        intent=Intent('click',target_id=target.id)
        result=output.execute(intent,target,scene)
        output.semantic.assert_called_once_with(intent,target,ANY)
        output.validate_visual.assert_called_once_with(target)
        output.validate_control.assert_called_once_with(target)
        output.win.send.assert_called_once()
        events=output.win.send.call_args.args[0]
        self.assertEqual(len(events),3)
        self.assertTrue(all(event.__class__.__name__=='INPUT' for event in events))
        self.assertEqual([event.u.mi.dwFlags for event in events],[0xC001,2,4])
        self.assertEqual(result['route'],'native')
    def test_invalid_pointer_commands_have_no_effect(self):
        for intent, reason in ((Intent('drag'), 'Unsupported action'),
                               (Intent('scroll', direction='left'), 'Choose a scroll direction'),
                               (Intent('scroll'), 'Choose a scroll direction')):
            with self.subTest(intent=intent):
                output, target, scene = self.pointer_click_fixture()
                with self.assertRaisesRegex(RuntimeError, reason):
                    output.execute(intent, target, scene)
                output.semantic.assert_not_called()
                output.win.activate.assert_not_called()
                output.win.send.assert_not_called()
    def test_pointer_click_rejects_changed_owner_before_sending(self):
        output,target,scene=self.pointer_click_fixture();output.win.pid.return_value=8
        with self.assertRaisesRegex(RuntimeError,'The app restarted'):
            output.execute(Intent('click',target_id=target.id),target,scene)
        output.win.send.assert_not_called();output.semantic.assert_not_called()
    def test_pointer_click_rejects_occluded_target_before_sending(self):
        output,target,scene=self.pointer_click_fixture();output.win.at.return_value=99
        with self.assertRaisesRegex(RuntimeError,'Another window covers the element'):
            output.execute(Intent('click',target_id=target.id),target,scene)
        output.win.send.assert_not_called()
    def test_pointer_click_rejects_stale_scene_before_sending(self):
        output,target,scene=self.pointer_click_fixture()
        scene=Scene(scene.id,time.monotonic()-4,scene.bounds,scene.elements,scene.window)
        with self.assertRaisesRegex(RuntimeError,'The screen changed'):
            output.execute(Intent('click',target_id=target.id),target,scene)
        output.win.send.assert_not_called();output.semantic.assert_not_called()
    def test_a_window_map_that_expires_during_validation_prevents_dispatch(self):
        stamp = 10.
        bounds = Rect(0, 0, 800, 600)
        for route in ('native', 'uia'):
            with self.subTest(route=route), patch('context_operator.providers.windows.time.monotonic', return_value=stamp) as clock:
                def expire(*args, **kwargs):
                    clock.return_value = stamp + MAP_TTL

                if route == 'native':
                    output, target, _ = self.pointer_click_fixture()
                    output.validate_control.side_effect = expire
                    intent = Intent('click', target_id=target.id)
                    api = None
                else:
                    output, pattern, api, target = self.semantic_fixture()
                    lookup = api.EditControl
                    def delayed_lookup(**kwargs):
                        expire()
                        return lookup(**kwargs)
                    api.EditControl = delayed_lookup
                    intent = Intent('type_text', text='Hello', target_id=target.id)

                win = output.win
                win.pid = Mock(return_value=target.pid)
                win.rect = Mock(return_value=bounds)
                window = WindowMap(target.window, target.pid, bounds, stamp, (target,))
                scene = Scene(1, stamp, bounds, (target,), target.window, windows=(window,))
                with (patch.dict('sys.modules', uiautomation=api) if api else nullcontext()), self.assertRaisesRegex(RuntimeError, 'Updating the window map'):
                    output.execute(intent, target, scene)
                win.send.assert_not_called()
                if api:
                    pattern.SetValue.assert_not_called()
    def test_pointer_click_cancelled_before_send(self):
        output,target,scene=self.pointer_click_fixture();output.set_guard(lambda:False)
        with self.assertRaisesRegex(RuntimeError,'Control stopped'):
            output.execute(Intent('click',target_id=target.id),target,scene)
        output.win.send.assert_not_called()
    def test_failed_native_click_is_not_retried(self):
        output,target,scene=self.pointer_click_fixture()
        intent=Intent('click',target_id=target.id)
        output.win.send.side_effect=RuntimeError('Windows rejected the input')
        with self.assertRaisesRegex(RuntimeError,'rejected the input'):
            output.execute(intent,target,scene)
        output.win.send.assert_called_once()
        output.semantic.assert_called_once_with(intent,target,ANY)
    def test_semantic_identity_handles_unnamed_and_long_labels(self):
        for name in ('','Long label '*25):
            with self.subTest(name_length=len(name)):
                output,pattern,api,target=self.semantic_fixture(name)
                with patch.dict('sys.modules',uiautomation=api):
                    self.assertTrue(output.semantic(Intent('type_text',text='Hello'),target))
                pattern.SetValue.assert_called_once_with('Hello',waitTime=0)
    def test_cancel_during_value_read_prevents_write(self):
        output,_,api,target=self.semantic_fixture()
        class Pattern:
            IsReadOnly=False
            SetValue=Mock()
            @property
            def Value(self):
                output.set_guard(lambda:False);return ''
        pattern=Pattern()
        original=api.EditControl
        def construct(**kwargs):
            control=original(**kwargs);control.GetPattern.return_value=pattern;return control
        api.EditControl=construct
        with patch.dict('sys.modules',uiautomation=api),self.assertRaisesRegex(RuntimeError,'Control stopped'):
            output.semantic(Intent('type_text',text='Hello'),target)
        pattern.SetValue.assert_not_called()
    def test_semantic_nonempty_field_is_not_replaced(self):
        output,pattern,api,target=self.semantic_fixture(pattern=Mock(IsReadOnly=False,Value='Existing text'))
        with patch.dict('sys.modules',uiautomation=api):
            self.assertFalse(output.semantic(Intent('type_text',text='Hello'),target))
        pattern.SetValue.assert_not_called()
    def test_semantic_refuses_new_window_covering_target(self):
        output,pattern,api,target=self.semantic_fixture()
        output.win.at.return_value=99
        with patch.dict('sys.modules',uiautomation=api),self.assertRaisesRegex(RuntimeError,'Another window covers the element'):
            output.semantic(Intent('type_text',text='Hello'),target)
        pattern.SetValue.assert_not_called()
    def test_cancelled_output_does_not_activate_window(self):
        output=WindowsOutput();output.win=Mock();output.win.shift_down.return_value=False;output.semantic=Mock(return_value=False)
        output.set_guard(lambda:False)
        scene=Scene(1,time.monotonic(),Rect(0,0,800,600),(),12)
        with self.assertRaisesRegex(RuntimeError,'Control stopped'):
            output.execute(Intent('type_text',text='Hello'),None,scene)
        output.win.activate.assert_not_called();output.win.send.assert_not_called()
    def test_named_dictation_focuses_field_before_unicode(self):
        output, button, scene = self.pointer_click_fixture()
        target = replace(button, id='edit', label='Name', role='Edit')
        scene = replace(scene, elements=(target,))
        win = output.win
        calls = Mock()
        calls.attach_mock(win.send, 'send')
        calls.attach_mock(output.validate_control, 'validate')
        with patch('context_operator.providers.windows.time.sleep'):
            output.execute(Intent('type_text', text='Hi 🙂', target_id=target.id), target, scene)
        self.assertEqual([entry[0] for entry in calls.mock_calls], ['validate', 'send', 'validate', 'send'])
        self.assertEqual(calls.mock_calls[2].args, (target,))
        self.assertEqual(calls.mock_calls[2].kwargs, {'focused': True})
        click, keys = [entry.args[0] for entry in win.send.call_args_list]
        self.assertEqual([(e.type, e.mi.dwFlags) for e in click], [(0, 0xC001), (0, 2), (0, 4)])
        self.assertEqual((click[0].mi.dx, click[0].mi.dy), (4610, 3644))
        self.assertEqual([(e.type, e.ki.vk, e.ki.scan, e.ki.flags) for e in keys],
                         [(1, 0, unit, flags) for unit in (0x48, 0x69, 0x20, 0xD83D, 0xDE42)
                          for flags in (4, 6)])
    def test_dictation_past_a_button_is_reported_for_the_window_not_the_button(self):
        output, button, scene = self.pointer_click_fixture()
        result = output.execute(Intent('type_text', text='Hi', target_id=button.id), button, replace(scene, title='Notes'))
        output.win.send.assert_called_once()
        self.assertEqual(result['target'], 'Notes')
    def test_failed_field_focus_never_types(self):
        output=WindowsOutput.__new__(WindowsOutput);output.win=Mock();output.win.shift_down.return_value=False;output.validate_visual=Mock()
        output.semantic=Mock(return_value=False);output.is_active=lambda:True
        output.validate_control=Mock(side_effect=[None,RuntimeError('focus lost')])
        w=output.win;w.u.IsWindow.return_value=True;w.pid.return_value=7;w.rect.return_value=Rect(0,0,800,600);w.at.return_value=12;w.foreground.return_value=12
        e=Element('edit','Name','Edit',Rect(60,40,150,40),12,pid=7);s=Scene(1,time.monotonic(),Rect(0,0,800,600),(e,),12,window_rect=Rect(0,0,800,600))
        with self.assertRaises(RuntimeError),patch('context_operator.providers.windows.time.sleep'):
            output.execute(Intent('type_text',text='Hello',target_id='edit'),e,s)
        self.assertEqual(w.send.call_count,1)
    def test_uia_scan_rejects_window_move_mid_scan(self):
        provider = WindowReader.__new__(WindowReader)
        provider.win = Mock()
        provider.uia = Mock()
        provider.cache = provider.condition = provider.secret_condition = Mock()
        win = provider.win
        win.pid.return_value = 7
        win.u.IsWindowVisible.return_value = True
        win.u.IsIconic.return_value = False
        root = provider.uia.ElementFromHandle.return_value
        root.FindFirst.return_value = None
        root.FindAllBuildCache.return_value.Length = 0
        bounds = Rect(0, 0, 800, 600)
        surface = WindowMap(12, 7, bounds, 0., title='Fixture')
        win.rect.side_effect = [bounds, bounds, bounds, Rect(20, 0, 800, 600)]
        self.assertEqual(provider.scan(surface).rect, bounds)
        with self.assertRaisesRegex(RuntimeError, 'Window changed during scan'):
            provider.scan(surface)

if __name__=='__main__':unittest.main()
