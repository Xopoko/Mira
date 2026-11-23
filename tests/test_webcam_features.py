import math
import unittest

import numpy as np

from context_operator.providers.webcam_runtime import (
    FEATURE_COUNT,
    extract_features,
    extract_single_face,
)


def _face(*, iris_shift=(0.0, 0.0), roll=0.0, blink_gap=0.15):
    """Create normalized 478-point geometry for deterministic feature tests."""
    points = np.zeros((478, 3), dtype=float)
    points[:, :2] = (0.5, 0.5)
    frame_scale = np.array([640.0, 480.0])
    points[1, :2] = np.array([0.47, 0.60])  # nose tip

    # Two eyes use the same corner order as the runtime contract. A slight
    # shared roll keeps the projected eye axes coherent.
    unit = np.array([math.cos(roll), math.sin(roll)])
    for a, b, upper, lower, iris_start, center_x in (
        (33, 133, 159, 145, 468, 0.30),
        (362, 263, 386, 374, 473, 0.70),
    ):
        center = np.array([center_x * 640.0, 0.40 * 480.0])
        half = unit * 32.0
        points[a, :2] = (center - half) / frame_scale
        points[b, :2] = (center + half) / frame_scale
        vertical = np.array([-unit[1], unit[0]])
        points[upper, :2] = (center - vertical * (blink_gap * 64.0 / 2.0)) / frame_scale
        points[lower, :2] = (center + vertical * (blink_gap * 64.0 / 2.0)) / frame_scale
        iris = center + unit * iris_shift[0] + vertical * iris_shift[1]
        points[iris_start:iris_start + 5, :2] = iris / frame_scale
    points[:, 0] = np.clip(points[:, 0], 0.02, 0.98)
    points[:, 1] = np.clip(points[:, 1], 0.02, 0.98)
    return points


class WebcamFeatureTests(unittest.TestCase):
    def test_returns_fixed_finite_eight_feature_vector(self):
        result = extract_features(_face(), 640, 480)
        self.assertEqual(result.reason, "")
        self.assertEqual(len(result.features), FEATURE_COUNT)
        self.assertTrue(np.isfinite(result.features).all())
        self.assertAlmostEqual(result.features[2], -0.30)
        self.assertAlmostEqual(result.features[3], 1.50)
        self.assertAlmostEqual(result.features[4], 0.0)
        self.assertAlmostEqual(result.features[5], 0.50)
        self.assertAlmostEqual(result.features[6], 0.495)
        self.assertAlmostEqual(result.features[7], 0.50)

    def test_iris_offsets_are_normalized_and_averaged(self):
        result = extract_features(_face(iris_shift=(6.4, -3.2)), 640, 480)
        self.assertEqual(result.reason, "")
        self.assertAlmostEqual(result.features[0], 0.10, places=6)
        self.assertAlmostEqual(result.features[1], -0.05, places=6)

    def test_eye_line_roll_is_in_radians(self):
        roll = math.radians(12)
        result = extract_features(_face(roll=roll), 640, 480)
        self.assertEqual(result.reason, "")
        self.assertAlmostEqual(result.features[4], roll, places=4)

    def test_rejects_closed_eyes_and_blink_blendshape(self):
        closed = extract_features(_face(blink_gap=0.002), 640, 480)
        explicit = extract_features(_face(), 640, 480, {"eyeBlinkLeft": 0.7})
        self.assertEqual(closed.reason, "blink")
        self.assertEqual(explicit.reason, "blink")
        self.assertIsNone(closed.features)

    def test_rejects_missing_or_ambiguous_faces(self):
        missing = extract_single_face([], [], 640, 480)
        ambiguous = extract_single_face([_face(), _face()], [], 640, 480)
        self.assertEqual(missing.reason, "face_not_found")
        self.assertEqual(ambiguous.reason, "ambiguous_faces")

    def test_rejects_small_eyes_bad_shape_nonfinite_and_oversize_frame(self):
        small = _face()
        small[133, :2] = small[33, :2] + (0.001, 0.0)
        malformed = extract_features(np.zeros((477, 3)), 640, 480)
        nonfinite = _face()
        nonfinite[1, 0] = np.nan
        too_large = extract_features(_face(), 1281, 480)
        self.assertEqual(extract_features(small, 640, 480).reason, "eyes_too_small")
        self.assertEqual(malformed.reason, "invalid_landmarks")
        self.assertEqual(extract_features(nonfinite, 640, 480).reason, "invalid_landmarks")
        self.assertEqual(too_large.reason, "invalid_frame_size")

    def test_accepts_maximum_capture_envelope(self):
        result = extract_features(_face(), 1280, 720)
        self.assertEqual(result.reason, "")
        self.assertEqual(len(result.features), FEATURE_COUNT)
        self.assertEqual(extract_features(_face(), 1281, 720).reason,
                         "invalid_frame_size")
        self.assertEqual(extract_features(_face(), 1280, 721).reason,
                         "invalid_frame_size")


if __name__ == "__main__":
    unittest.main()
