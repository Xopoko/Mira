"""Synthetic LAN/USB contracts only: no physical phone or private settings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from ipaddress import IPv4Address
import socket
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
from context_operator.providers.phone_connection import (PhoneConnection, WIFI_PORT, WIFI_PHONE_NEXT_STEP,
                                                         WifiPhoneNotFound, find_phones)
from context_operator.providers.gaze import PhoneGaze, Sample
from context_operator.foundations.gaze.arkit import ARKitWorker, PacketDecoder


def packet(seq=1, age=5):
    face=np.eye(4);face[2,3]=-.7
    left=face.copy();left[0,3]=-.032
    right=face.copy();right[0,3]=.032
    return dict(version=1,session='synthetic',seq=seq,timestamp=20+seq/60,ageMs=age,
                trueDepth=True,depthFrame=True,tracked=True,blink=[0,0],lookAt=[0,0,0],
                **{k:v.flatten(order='F').tolist() for k,v in [('face',face),('leftEye',left),('rightEye',right)]})


class PhoneTransportTests(unittest.TestCase):
    def test_manual_wifi_uses_private_address_and_fixed_port(self):
        for host in ('192.168.2.42','10.1.2.3','172.20.2.4'):
            with patch('context_operator.providers.phone_connection.socket.create_connection') as connect:
                PhoneConnection('wifi',host).open()
                connect.assert_called_once_with((host,WIFI_PORT),timeout=2)
        for host in ('localhost','127.0.0.1','0.0.0.0','8.8.8.8','169.254.1.1','224.0.0.1',
                     '192.168.1.1:39472','https://192.168.1.1','192.168.001.2','::1'):
            with self.subTest(host=host),self.assertRaises(ValueError):PhoneConnection('wifi',host)

    def test_usb_default_retains_exact_binding_owner_and_wifi_never_falls_back(self):
        with patch('context_operator.foundations.gaze.arkit.open_usb_stream') as usb:
            self.assertIs(PhoneConnection.from_settings({}).open(),usb.return_value)
            usb.assert_called_once()
        with patch('context_operator.foundations.gaze.arkit.open_usb_stream') as usb, \
             patch('context_operator.providers.phone_connection.find_phones',return_value=[]) as discover, \
             patch('context_operator.providers.phone_connection.socket.create_connection',side_effect=OSError('private details')):
            with self.assertRaises(WifiPhoneNotFound) as error:PhoneConnection('wifi','192.168.2.42').open()
            self.assertEqual(str(error.exception),f'Can’t find iPhone on your Wi‑Fi. {WIFI_PHONE_NEXT_STEP}')
            discover.assert_called_once_with();usb.assert_not_called()

    def test_usb_uses_the_explicit_adapter_path(self):
        variable = 'MIRA_IPHONE_USB_BRIDGE'
        with TemporaryDirectory() as folder, patch.dict(os.environ):
            os.environ.pop(variable, None)
            with self.assertRaisesRegex(ConnectionError, variable):
                PhoneConnection('usb').open()
            adapter = Path(folder) / 'adapter.py'
            adapter.write_text("def open_stream():\n    return 'test stream'\n", encoding='utf-8')
            os.environ[variable] = str(adapter)
            self.assertEqual(PhoneConnection('usb').open(), 'test stream')

    def test_discovery_checks_the_response_on_a_loopback_port(self):
        for response, expected in (
            (b'{"version":1,"ageMs":3}\n', ['127.0.0.1']),
            (b'{"version":1,"ageMs":"3"}\n', []),
            (b'{"version":2,"ageMs":3}\n', []),
            (b'not a Mira response\n', []),
        ):
            with self.subTest(response=response), socket.socket() as listener:
                listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(1)
                port = listener.getsockname()[1]
                requests = []

                def serve():
                    connection, _ = listener.accept()
                    with connection, connection.makefile('rb') as incoming:
                        connection.settimeout(1)
                        requests.append(incoming.readline(16))
                        connection.sendall(response)

                thread = threading.Thread(target=serve,daemon=True)
                thread.start()
                self.assertEqual(find_phones(['127.0.0.1'],port=port),expected)
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(requests,[b'GET\n'])

    def test_discovery_prioritizes_primary_subnet_and_excludes_local_addresses(self):
        with patch('context_operator.providers.phone_connection._primary_address',
                   return_value=IPv4Address('192.168.2.50')), \
             patch('context_operator.providers.phone_connection._interface_addresses',
                   return_value=['10.1.0.5','192.168.2.50','192.168.2.51']), \
             patch('context_operator.providers.phone_connection._responds',
                   side_effect=lambda host, **_:host=='192.168.2.42') as probe:
            self.assertEqual(find_phones(),['192.168.2.42'])
            probed = {call.args[0] for call in probe.call_args_list}
            self.assertEqual(probed,{f'192.168.2.{i}' for i in range(1,255)}-{'192.168.2.50','192.168.2.51'})

    def test_empty_wifi_host_skips_direct_connect_and_records_discovered_host(self):
        connection = PhoneConnection('wifi',' ')
        with patch('context_operator.providers.phone_connection.find_phones',return_value=['192.168.2.42']) as discover, \
             patch('context_operator.providers.phone_connection.socket.create_connection') as connect:
            self.assertIs(connection.open(),connect.return_value)
            discover.assert_called_once_with()
            connect.assert_called_once_with(('192.168.2.42',WIFI_PORT),timeout=2)
        self.assertEqual(connection.found_host,'192.168.2.42')

    def run_fixture(self,serve):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        self.addCleanup(listener.close)
        worker=ARKitWorker(Sample,port=listener.getsockname()[1])
        errors=[]
        def handle():
            try:
                connection,_=listener.accept()
                with connection:serve(connection,worker)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=handle,daemon=True);thread.start();worker.start()
        worker.join(2);worker.stop_event.set();thread.join(1);worker.join(1)
        self.assertFalse(worker.is_alive());self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        return worker

    def test_fragmented_latest_frame_and_disconnect_clear_gaze(self):
        observations=[]
        def serve(sock,worker):
            self.assertEqual(sock.recv(32),b'GET\n')
            raw=json.dumps(packet()).encode()+b'\n'
            for chunk in (raw[:13],raw[13:111],raw[111:]):sock.sendall(chunk)
            self.assertEqual(sock.recv(32),b'GET\n')
            observations.append(worker.latest())
        worker=self.run_fixture(serve)
        self.assertEqual(observations[0].features.shape,(22,))
        self.assertGreaterEqual(observations[0].source_age_ms,5)
        self.assertTrue(worker.latest().error);self.assertIsNone(worker.latest().features)

    def test_delayed_network_reply_is_not_current_even_if_phone_says_fresh(self):
        observations=[]
        def serve(sock,worker):
            sock.recv(32);time.sleep(.23);sock.sendall(json.dumps(packet()).encode()+b'\n')
            sock.recv(32);observations.append(worker.latest())
        self.run_fixture(serve)
        sample=observations[0];self.assertGreaterEqual(sample.source_age_ms,230)
        phone=PhoneGaze.__new__(PhoneGaze);phone.worker=Mock();phone.worker.latest.return_value=sample
        phone.smoother=Mock();phone.profile=Mock()
        gaze,status,_=phone.poll(230)
        self.assertIsNone(gaze);self.assertIn('out of date',status);phone.profile.predict.assert_not_called()

    def test_dripping_partial_reply_hits_total_request_deadline(self):
        def serve(sock,worker):
            sock.recv(32)
            for _ in range(6):
                sock.sendall(b' ');time.sleep(.12)
        before=time.monotonic();worker=self.run_fixture(serve)
        self.assertLess(time.monotonic()-before,1.2)
        self.assertTrue(worker.latest().error);self.assertIsNone(worker.latest().features)

    def test_unrequested_queue_is_rejected_instead_of_becoming_fresh(self):
        def serve(sock,worker):
            sock.recv(32)
            sock.sendall((json.dumps(packet())+'\n'+json.dumps(packet(2))+'\n').encode())
        worker=self.run_fixture(serve)
        self.assertEqual(worker.frames,0);self.assertTrue(worker.latest().error)

    def test_duplicates_stale_frames_and_changed_session_cannot_refresh(self):
        decoder=PacketDecoder();decoder.decode(packet())
        self.assertIsNone(decoder.decode(packet()))
        self.assertIsNone(decoder.decode(packet(2,age=151)).features)
        changed=packet(3);changed['session']='other-phone'
        with self.assertRaises(ValueError):decoder.decode(changed)


class ConnectionSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_connection_settings_validate_before_stopping(self):
        from context_operator.controller import Controller
        from context_operator.settings import DEFAULTS
        c=Controller(dict(DEFAULTS,phone_transport='usb',vision=False,ranker=False,anticipation=False));c.timer.stop()
        try:
            with patch.object(c,'pause') as pause,patch('context_operator.settings.save') as save:
                with self.assertRaises(ValueError):c.set_phone_connection('wifi','http://wrong')
                pause.assert_not_called();save.assert_not_called()
                save.side_effect=OSError('disk failure')
                with self.assertRaises(OSError):c.set_phone_connection('wifi','192.168.2.42')
                pause.assert_not_called()
                self.assertEqual(c.settings['phone_transport'],'usb')
                self.assertEqual(c.settings['phone_wifi_host'],'')
                save.reset_mock(side_effect=True)
                c.set_phone_connection('wifi','192.168.2.42')
                pause.assert_called_once();save.assert_called_once()
                self.assertEqual(c.settings['phone_wifi_host'],'192.168.2.42')
                self.assertEqual(c.settings['phone_transport'],'wifi')
        finally:c.close()

    def test_fresh_settings_default_to_wifi_but_keep_saved_usb(self):
        from context_operator.settings import load
        with TemporaryDirectory() as directory, patch('context_operator.settings.LOCAL', Path(directory)):
            path=Path(directory)/'settings.json'
            fresh=load()
            self.assertEqual((fresh['phone_transport'],fresh['phone_wifi_host']),('wifi',''))
            self.assertFalse(fresh['first_run_done'])
            self.assertFalse(fresh['show_scene_map'])
            self.assertFalse(path.exists())
            saved='{"phone_transport":"usb","phone_wifi_host":"192.168.2.42"}'
            path.write_text(saved,encoding='utf-8')
            existing=load()
            self.assertEqual((existing['phone_transport'],existing['phone_wifi_host']),('usb','192.168.2.42'))
            self.assertEqual(path.read_text(encoding='utf-8'),saved)

    def test_discovered_wifi_address_is_saved_without_pausing_or_stopping_phone(self):
        from context_operator.controller import Controller
        from context_operator.settings import DEFAULTS
        saved_host, found_host = '192.168.2.41', '192.168.2.42'
        connection = PhoneConnection('wifi', saved_host)
        with patch('context_operator.providers.phone_connection.find_phones',return_value=[found_host]), \
             patch('context_operator.providers.phone_connection.socket.create_connection',side_effect=[OSError(),Mock()]):
            connection.open()
        c=Controller(dict(DEFAULTS,phone_transport='wifi',phone_wifi_host=saved_host,
                          vision=False,ranker=False,anticipation=False))
        c.timer.stop()
        phone=Mock(connection=connection,measurement=None)
        phone.poll.return_value=(None,'Waiting for iPhone',0)
        c.phone=phone;c.running=True;c.win=Mock()
        c.win.emergency.return_value=False;c.win.shift_down.return_value=False
        try:
            with patch.object(c,'pause') as pause, patch('context_operator.settings.save') as save:
                c.tick();c.tick()
                save.assert_called_once()
                self.assertEqual(save.call_args.args[0]['phone_wifi_host'],found_host)
                self.assertEqual(c.settings['phone_wifi_host'],found_host)
                self.assertIs(c.phone,phone);self.assertTrue(c.running)
                pause.assert_not_called();phone.stop.assert_not_called()
        finally:c.close()

    def test_wifi_connect_uses_true_depth_phone_provider(self):
        from context_operator.controller import Controller
        from context_operator.settings import DEFAULTS
        c=Controller(dict(DEFAULTS,phone_transport='wifi',phone_wifi_host='192.168.2.42'))
        c.timer.stop()
        try:
            c.start=Mock();c._start_clicks=Mock()
            with patch('context_operator.controller.PhoneGaze') as provider:
                provider.return_value.profile_key='fixture-profile'
                provider.return_value.feature_scale=np.ones(22)
                provider.return_value.profile.arrays={'training_x':np.empty((0,22)),'training_y':np.empty((0,2))}
                c.toggle_phone()
                provider.assert_called_once()
                self.assertEqual(provider.call_args.args,(c.win.bounds(),))
                connection=provider.call_args.kwargs['connection']
                self.assertEqual((connection.transport,connection.host),('wifi','192.168.2.42'))
                provider.return_value.start.assert_called_once()
            self.assertIs(c.phone,provider.return_value)
        finally:c.close()
