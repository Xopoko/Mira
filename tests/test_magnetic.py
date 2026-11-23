import unittest
from dataclasses import replace

from context_operator.core.contracts import Element, Gaze, Rect, Scene
from context_operator.core.fusion import Fusion


class MagneticSelectionTests(unittest.TestCase):
    def setUp(self):
        self.plus = Element('plus', 'Plus', 'Button', Rect(400, 200, 100, 70), 1)
        self.minus = Element('minus', 'Minus', 'Button', Rect(550, 200, 100, 70), 1)
        self.clock = Element('clock', 'Clock', 'Button', Rect(970, 750, 60, 35), 1)
        self.scene = Scene(1, 100, Rect(0, 0, 1040, 800),
                           (self.plus, self.minus, self.clock), 1)
        self.fusion = Fusion(.12)

    def gaze(self, x, captured, y=230, radius=90):
        return Gaze(x, y, captured, radius)

    def choose(self, x, captured, now=None, scene=None, query='', y=230):
        now = captured if now is None else now
        scene = self.scene if scene is None else scene
        return self.fusion.choose(scene, self.gaze(x, captured, y), now, query,
                                 magnetic=True)

    def test_small_jitter_keeps_the_actual_focus_target(self):
        self.assertEqual(self.choose(450, 100).target.id, 'plus')
        for timestamp, x in ((100.03, 505), (100.06, 518), (100.09, 508),
                             (100.12, 520), (100.15, 510)):
            self.assertEqual(self.choose(x, timestamp).target.id, 'plus')

    def test_directional_shift_switches_after_unique_frame_dwell(self):
        self.choose(450, 100)
        self.assertEqual(self.choose(560, 100.02).target.id, 'plus')
        # Reusing one captured frame cannot complete the dwell, even when now advances.
        self.assertEqual(self.choose(560, 100.02, now=100.19).target.id, 'plus')
        self.assertEqual(self.choose(560, 100.20).target.id, 'minus')

    def test_nearby_buttons_switch_with_broad_gaze_radii(self):
        for radius in (230, 650):
            fusion = Fusion(.12)
            scene = self.scene
            plus = Gaze(450, 230, 100, radius)
            minus = Gaze(560, 230, 100.02, radius)
            settled = Gaze(560, 230, 100.20, radius)
            self.assertEqual(fusion.choose(scene, plus, 100, magnetic=True).target.id,
                             'plus')
            self.assertEqual(fusion.choose(scene, minus, 100.02, magnetic=True).target.id,
                             'plus')
            self.assertEqual(fusion.choose(scene, settled, 100.20, magnetic=True).target.id,
                             'minus')

    def test_nested_button_can_replace_locked_pane(self):
        pane = Element('pane', 'Panel', 'Pane', Rect(350, 160, 360, 260), 1)
        button = Element('child', 'Send', 'Button', Rect(400, 200, 90, 45), 1)
        scene = Scene(2, 100, self.scene.bounds, (pane, button), 1)
        fusion = Fusion(.12)
        self.assertEqual(fusion.choose(scene, self.gaze(420, 100), 100,
                                       'Panel', magnetic=True).target.id, 'pane')
        self.assertEqual(fusion.choose(scene, self.gaze(420, 102.01), 102.01,
                                       magnetic=True).target.id, 'pane')
        self.assertEqual(fusion.choose(scene, self.gaze(420, 102.20), 102.20,
                                       magnetic=True).target.id, 'child')

    def test_far_jump_switches_without_dwell(self):
        self.choose(450, 100)
        self.assertEqual(self.choose(1000, 100.02, y=770).target.id, 'clock')

    def test_stale_or_future_gaze_clears_lock(self):
        self.choose(450, 100)
        self.assertIsNone(self.fusion.choose(
            self.scene, self.gaze(450, 99), 100, magnetic=True).target)
        self.assertIsNone(self.fusion._magnetic_identity)
        self.assertIsNone(self.fusion.choose(
            self.scene, self.gaze(450, 101), 100, magnetic=True).target)
        self.assertIsNone(self.fusion._magnetic_identity)

    def test_future_gaze_does_not_preserve_lock_during_voice_override(self):
        self.choose(450, 100)
        focus = self.fusion.choose(self.scene, self.gaze(450, 101), 100,
                                   'Clock', magnetic=True)
        self.assertEqual(focus.target.id, 'clock')
        self.assertEqual(self.fusion._magnetic_identity, self.clock.identity)
        self.assertIsNone(self.fusion._magnetic_anchor)

    def test_future_scene_clears_lock(self):
        self.choose(450, 100)
        future = Scene(2, 101, self.scene.bounds, self.scene.elements, 1)
        self.assertIsNone(self.fusion.choose(
            future, self.gaze(450, 100), 100, magnetic=True).target)
        self.assertIsNone(self.fusion._magnetic_identity)

    def test_removed_or_replaced_target_does_not_keep_old_identity(self):
        self.choose(450, 100)
        other_source = replace(self.plus, source='vision')
        source_scene = Scene(2, 100.01, self.scene.bounds,
                             (other_source, self.minus, self.clock), 1)
        self.assertEqual(self.choose(450, 100.01, scene=source_scene).target.id, 'plus')
        self.assertEqual(self.fusion._magnetic_identity, other_source.identity)
        changed = Element('plus', 'Other', 'Button', Rect(410, 200, 100, 70),
                          1, pid=99)
        scene = Scene(2, 100.02, self.scene.bounds,
                      (changed, self.minus, self.clock), 1)
        self.assertEqual(self.choose(450, 100.02, scene=scene).target.label, 'Other')
        self.assertEqual(self.fusion._magnetic_identity, changed.identity)
        removed = Scene(3, 100.04, self.scene.bounds, (self.minus, self.clock), 1)
        self.assertEqual(self.choose(600, 100.04, scene=removed).target.id, 'minus')
        self.assertEqual(self.fusion._magnetic_identity, self.minus.identity)

    def test_voice_query_and_cycle_are_immediate_overrides(self):
        focus = self.choose(450, 100)
        self.assertEqual(self.fusion.cycle(focus, 100.01, 'right').id, 'minus')
        self.assertEqual(self.choose(450, 100.02).target.id, 'minus')
        self.assertEqual(self.choose(450, 100.03, query='Clock').target.id, 'clock')

    def test_explicit_choice_keeps_the_exact_element_without_gaze(self):
        duplicate = replace(self.minus, label=self.plus.label)
        scene = replace(self.scene, elements=(self.plus, duplicate))
        moved = replace(duplicate, rect=Rect(600, 200, 100, 70))
        changed_scene = replace(scene, elements=(self.plus, moved))
        for magnetic in (False, True):
            with self.subTest(magnetic=magnetic):
                fusion = Fusion()
                fusion.select(duplicate, 100, 3.)
                self.assertEqual(fusion.choose(scene, self.gaze(450, 100), 100,
                                               magnetic=magnetic).target, duplicate)
                focus = fusion.choose(scene, None, 100.1, magnetic=magnetic)
                self.assertEqual(focus.target, duplicate)
                self.assertEqual(fusion.selection_until(focus.target, 100.1), 103.)
                self.assertIsNone(fusion.cycle(focus, 100.1, 'right'))
                self.assertIsNone(fusion.choose(changed_scene, None, 100.2,
                                                magnetic=magnetic).target)
                self.assertEqual(fusion.selection_until(duplicate, 100.2), 0.)
                self.assertIsNone(fusion.choose(scene, None, 100.3,
                                                magnetic=magnetic).target)
                fusion.select(duplicate, 100.3, .2)
                self.assertEqual(fusion.choose(scene, self.gaze(450, 100.4), 100.4,
                                               magnetic=magnetic).target, duplicate)
                self.assertEqual(fusion.choose(scene, self.gaze(450, 100.6), 100.6,
                                               magnetic=magnetic).target, self.plus)

    def test_no_candidates_clears_lock_and_default_mode_is_unchanged(self):
        self.choose(450, 100)
        empty = Scene(2, 100.02, self.scene.bounds, (), 1)
        self.assertIsNone(self.fusion.choose(empty, self.gaze(450, 100.02),
                                             100.02, magnetic=True).target)
        self.assertIsNone(self.fusion._magnetic_identity)

        legacy = Fusion(.12)
        self.assertEqual(legacy.choose(self.scene, self.gaze(450, 100), 100)
                         .target.id, 'plus')
        self.assertEqual(legacy.choose(self.scene, self.gaze(610, 100.02), 100.02)
                         .target.id, 'plus')
        self.assertIsNone(legacy._magnetic_identity)


if __name__ == '__main__':
    unittest.main()
