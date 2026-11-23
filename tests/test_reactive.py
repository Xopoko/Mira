import json
import unittest,time,tempfile
from pathlib import Path
from unittest.mock import Mock
from context_operator.providers.fast_intent import FastIntent
from context_operator.providers.learning import GazeLearning
from context_operator.core.contracts import *

class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.router=FastIntent()
    def setUp(self):
        self.now=time.monotonic()
        self.chat=Element('chat','ChatGPT - Google Chrome','TabItem',Rect(0,0,220,40),1)
        self.new=Element('new','New Tab','Button',Rect(260,0,70,40),1)
        self.plus=Element('plus','Plus','Button',Rect(700,400,90,50),1)
        self.scene=Scene(1,self.now,Rect(0,0,1000,700),(self.chat,self.new,self.plus),1)
        self.focus=Focus(self.chat);self.gaze=Gaze(100,20,self.now,100)
    def infer(self,text,**kwargs):return self.router.interpret(text,self.scene,self.focus,self.gaze,self.now,**kwargs)
    def test_fast_phrases_without_generative_model(self):
        cases = (
            ('Click here', 'click', 'chat'),
            ('Open ChatGPT', 'click', 'chat'),
            ('Switch to ChatGPT', 'click', 'chat'),
            ('Click New Tab', 'click', 'new'),
            ('Highlight Plus', 'select', 'plus'),
            ('New Tab', 'select', 'new'),
            ('Click ChatGPT twice', 'double_click', 'chat'),
            ('Go to New Tab', 'click', 'new'),
            ('Button New Tab', 'select', 'new'),
            ('Right click ChatGPT', 'right_click', 'chat'),
            # Russian transcripts still use the same classifier and normalized target labels.
            ('\u043e\u0442\u043a\u0440\u043e\u0439 \u043d\u043e\u0432\u0443\u044e ' '\u0432\u043a\u043b\u0430\u0434\u043a\u0443', 'click', 'new'),
            ('\u043d\u0430\u0436\u043c\u0438 \u043a\u043d\u043e\u043f\u043a\u0443 ' '\u043f\u043b\u044e\u0441', 'click', 'plus'),
            ('\u0427\u0430\u0442 \u0434\u0436\u0438 \u043f\u0438 \u0442\u0438', 'select', 'chat'),
        )
        for text, action, target in cases:
            with self.subTest(text=text):
                result = self.infer(text)
                self.assertIsNotNone(result)
                self.assertEqual((result.action, result.target_id), (action, target))

    def test_complex_negated_narration_stays_with_context(self):
        cases = (
            'Do not click ChatGPT',
            'He said click Plus',
            'If ready, open ChatGPT',
            'Click Plus then open New Tab',
            'Why would I click Plus?',
            'Make the font a little larger',
            'Close ChatGPT',
            '\u043d\u0435 \u043d\u0430\u0436\u0438\u043c\u0430\u0439 ChatGPT',
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(self.infer(text))

    def test_ambiguous_name_requires_gaze(self):
        duplicate=Element('other','New Tab','Button',Rect(600,0,70,40),1)
        self.scene=Scene(1,self.now,self.scene.bounds,(self.new,duplicate),1)
        self.gaze=None
        self.assertEqual(self.infer('New Tab').action,'clarify')
        self.gaze=Gaze(290,20,self.now,90)
        self.assertEqual(self.infer('New Tab').target_id,'new')
    def test_missing_label_cannot_click_gaze_as_substitute(self):self.assertIsNone(self.infer('open YouTube'))
    def test_deictic_command_requires_valid_gaze_or_explicit_selection(self):
        for gaze in (None,Gaze(100,20,self.now-30),Gaze(100,20,self.now+30),Gaze(100,20,self.now,valid=False)):
            self.gaze=gaze
            self.assertEqual(self.infer('click here').action,'clarify')
        self.assertEqual(self.infer('click here',allow_selected=True).action,'click')
class LegacyLearningTests(unittest.TestCase):
    def test_legacy_screen_anchor_survives_click_and_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'learn.json';bounds=Rect(0,0,1000,700)
            old=dict(version=1,profile='profile',display=bounds.as_list(),
                     anchors=[[350,350,122.5,-98,1]],count=1)
            path.write_text(json.dumps(old))
            model=GazeLearning(path);model.bind('profile',bounds)
            original=model.correct(Gaze(350,350,100))
            self.assertGreater(original.x,350)
            sample=GazeMeasurement(100,(0.,)*22,(350,350),True)
            self.assertTrue(model.learn_click(sample,(400,350)))
            model.undo_last()
            restored=GazeLearning(path);restored.bind('profile',bounds)
            self.assertEqual(restored.count,1)
            self.assertEqual(restored.anchors,old['anchors'])
            self.assertEqual(restored.correct(Gaze(350,350,100)),original)

if __name__=='__main__':unittest.main()
