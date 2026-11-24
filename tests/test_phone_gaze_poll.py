import unittest
from unittest.mock import Mock,patch

import numpy as np

from context_operator.core.contracts import Rect
from context_operator.foundations.gaze.smoothing import Smoother
from context_operator.providers.gaze import PhoneGaze
from context_operator.foundations.gaze.arkit import Sample
from context_operator.providers.phone_connection import PhoneConnection


BOUNDS=Rect(0,0,1000,700)


class PhoneGazePollTests(unittest.TestCase):
    def test_base_output_clips_but_training_keeps_full_extrapolation(self):
        phone=PhoneGaze.__new__(PhoneGaze);phone.bounds=BOUNDS;phone.worker=Mock();phone.profile=Mock()
        phone.connection=PhoneConnection()
        phone.profile.supports.return_value=True;phone.profile.predict.return_value=np.array([2000,-500])
        phone.smoother=Smoother(.065,scale=BOUNDS.h);phone.last_seq=0;phone.validation_radius=140.
        phone.worker.latest.return_value=Sample(seq=1,captured=100,features=np.zeros(22))
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):gaze,_,_=phone.poll(230)
        self.assertEqual((gaze.x,gaze.y),(999,0));self.assertEqual(phone.measurement.point,(2000,-500));self.assertEqual(gaze.radius,140.)
        phone.worker.latest.return_value.captured=99
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):
            gaze,reason,_=phone.poll(230);self.assertIsNone(gaze);self.assertIsNone(phone.measurement);self.assertIn('out of date',reason)
    def test_diagnostic_separates_tracking_from_coverage(self):
        phone=PhoneGaze.__new__(PhoneGaze);phone.bounds=BOUNDS;phone.worker=Mock();phone.profile=Mock()
        phone.profile.supports.return_value=False
        phone.profile.predict.return_value=np.array([2000,-500]);phone.smoother=Smoother(.065)
        x=np.zeros(22);x[6]=-.4
        phone.worker.latest.return_value=Sample(seq=1,captured=100,features=x)
        with patch('context_operator.providers.gaze.time.monotonic',return_value=100):gaze,reason,_=phone.poll(230)
        self.assertIsNone(gaze);self.assertIn('40 cm',reason);self.assertIn('position outside setup range',reason)
        self.assertIsNotNone(phone.measurement);self.assertFalse(phone.measurement.supported)


if __name__=='__main__':unittest.main()
