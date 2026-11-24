import unittest
from context_operator.core.contracts import *
from context_operator.core.fusion import Fusion
from context_operator.core.commands import quick_intent

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.now=100.
        self.plus=Element('plus','Plus','Button',Rect(400,200,100,70),1)
        self.minus=Element('minus','Minus','Button',Rect(550,200,100,70),1)
        self.clock=Element('clock','Clock','Button',Rect(970,750,60,35),1)
        self.scene=Scene(1,100,Rect(0,0,1040,800),(self.plus,self.minus,self.clock),1)
        self.fusion=Fusion(0)
    def test_corner_gaze_is_object_context_not_pixel(self):
        focus=self.fusion.choose(self.scene,Gaze(910,710,100,180),100)
        self.assertEqual(focus.target.id,'clock')
        self.assertNotEqual(focus.target.rect.center,(910,710))
    def test_no_target_when_gaze_lost(self):
        self.fusion.choose(self.scene,Gaze(440,230,100),100)
        self.assertIsNone(self.fusion.choose(self.scene,None,100.1).target)
        self.assertIsNone(self.fusion.choose(self.scene,Gaze(440,230,99),100).target)
    def test_expired_scene_and_future_sample(self):
        for magnetic in (False,True):
            for now in (99.,100.+SCENE_TTL,104.):
                with self.subTest(magnetic=magnetic,now=now):
                    self.fusion.choose(self.scene,Gaze(440,230,100),100,magnetic=magnetic)
                    focus=self.fusion.choose(self.scene,Gaze(440,230,now),now,magnetic=magnetic)
                    self.assertEqual(focus.reason,'Updating the screen map')
                    self.assertIsNone(focus.target)
                    self.assertIsNone(self.fusion.current)
        self.assertIsNone(self.fusion.choose(self.scene,Gaze(440,230,101),100).target)
    def test_text_preselection_keeps_a_named_target_away_from_gaze(self):
        focus=self.fusion.choose(self.scene,Gaze(440,230,100),100,'Clock')
        self.assertEqual([r.element.id for r in focus.alternatives[:2]],['clock','plus'])
    def test_direction_correction_is_held(self):
        focus=self.fusion.choose(self.scene,Gaze(440,230,100),100)
        self.assertEqual(self.fusion.cycle(focus,100,'right').id,'minus')
        self.assertEqual(self.fusion.choose(self.scene,Gaze(440,230,100.1),100.1).target.id,'minus')
    def test_correction_never_survives_disconnect(self):
        focus=self.fusion.choose(self.scene,Gaze(440,230,100),100)
        self.fusion.cycle(focus,100,'right')
        self.fusion.choose(self.scene,None,100.1)
        self.assertEqual(self.fusion.choose(self.scene,Gaze(440,230,100.2),100.2).target.id,'plus')
    def test_selection_hysteresis(self):
        f=Fusion(.12)
        self.assertEqual(f.choose(self.scene,Gaze(440,230,100,90),100).target.id,'plus')
        self.assertEqual(f.choose(self.scene,Gaze(610,230,100.02,90),100.02).target.id,'plus')
        self.assertEqual(f.choose(self.scene,Gaze(610,230,100.20,90),100.20).target.id,'minus')
    def test_named_request_works_without_gaze(self):
        self.assertEqual(self.fusion.choose(self.scene,None,100,'Plus').target.id,'plus')

class CommandsTests(unittest.TestCase):
    def test_stop_is_instant(self):self.assertEqual(quick_intent('Stop!').action,'pause')
    def test_dictation_preserves_case_and_punctuation(self):
        self.assertEqual(quick_intent('Type: Hello, Windows!').text,'Hello, Windows!')
    def test_natural_commands_go_to_model(self):
        self.assertIsNone(quick_intent('Do not click Plus, click the one to its right'))
        self.assertIsNone(quick_intent('Make the font a little larger'))
    def test_chord(self):
        self.assertEqual(quick_intent('Ctrl+A').keys, ('Ctrl', 'A'))
        self.assertEqual(quick_intent('Press Ctrl+A').keys, ('Ctrl', 'A'))

    def test_directional_corrections(self):
        for direction in ('left', 'right', 'up', 'down'):
            with self.subTest(direction=direction):
                intent = quick_intent(f'Move {direction}')
                self.assertEqual((intent.action, intent.direction), ('select', direction))
        self.assertEqual(quick_intent('\u043f\u0440\u0430\u0432\u0435\u0435').direction, 'right')

if __name__=='__main__':unittest.main()
