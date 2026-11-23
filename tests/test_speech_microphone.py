import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from context_operator import settings
from context_operator.foundations import local_models
from context_operator.foundations.speech.runtime import VoiceError, check_policy, select_microphone
from context_operator.providers.models import Speech


class SpeechMicrophoneTest(unittest.TestCase):
    def test_standalone_inference_stops_if_an_installation_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = Path(directory) / 'LocalModels'
            with patch.object(local_models, 'OWNER', owner), patch.object(local_models, 'STANDALONE', True):
                check_policy()
                owner.mkdir()
                with self.assertRaisesRegex(VoiceError, 'LocalModels was detected'):
                    check_policy()

    def test_system_default_and_missing_named_device_use_default_capture(self):
        sd = Mock()
        sd.default = SimpleNamespace(device=(0, 1))
        sd.query_hostapis.return_value = [{'name': 'Windows WASAPI'}, {'name': 'MME'}]
        sd.query_devices.return_value = [
            dict(name=name, hostapi=api, max_input_channels=channels, default_samplerate=48000)
            for name, api, channels in [('Built-in mic', 0, 1), ('Speakers', 0, 0),
                                        ('Headset', 1, 1), ('Headset', 0, 1)]
        ]
        self.assertEqual(select_microphone(sd, None, None, system_default=True)['index'], 0)
        self.assertEqual(select_microphone(sd, None, None)['index'], 0)
        sd.check_input_settings.assert_called_with(device=0, channels=1, dtype='float32', samplerate=16000)
        selected = select_microphone(sd, None, 'Unplugged mic', default_if_missing=True)
        self.assertEqual((selected['index'], selected['fallbackFrom']), (0, 'Unplugged mic'))
        self.assertEqual(select_microphone(sd, None, 'Headset', default_if_missing=True)['index'], 3)
        self.assertEqual(select_microphone(sd, 7, 'Headset', host_api='MME', default_if_missing=True)['index'], 2)
        with self.assertRaises(VoiceError):
            select_microphone(sd, None, 'Unplugged mic')
        sd.default.device = (-1, 1)
        with self.assertRaisesRegex(VoiceError, 'Unplugged mic isn’t connected.*No system default'):
            select_microphone(sd, None, 'Unplugged mic', default_if_missing=True)

    def test_fresh_settings_launch_default_and_legacy_choice_survives_save(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, 'LOCAL', Path(directory)):
            fresh = settings.load()
            self.assertEqual(fresh['microphone'], '')
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'microphone': 'Saved mic'}), encoding='utf-8')
            legacy = settings.load()
            settings.save(dict(legacy, first_run_done=True))
            self.assertEqual(settings.load()['microphone'], 'Saved mic')

        with patch('context_operator.providers.models.subprocess.Popen') as launch, \
             patch('context_operator.providers.models.threading.Thread'):
            Speech(Mock()).start(fresh['microphone'])
            self.assertIn('--system-default', launch.call_args.args[0])
            Speech(Mock()).start(legacy['microphone'])
            self.assertIn('--device-name=Saved mic', launch.call_args.args[0])
            self.assertIn('--default-if-missing', launch.call_args.args[0])
