"""Read numeric ARKit snapshots from the iPhone over an open phone connection."""
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
import time
import numpy as np

ENGINE_VERSION = 'arkit-truedepth-v1'


def open_usb_stream():
    configured = os.environ.get('MIRA_IPHONE_USB_BRIDGE')
    adapter = Path(configured).expanduser() if configured else None
    if adapter is None or not adapter.is_file():
        raise ConnectionError('Use Wi-Fi or set MIRA_IPHONE_USB_BRIDGE to an installed USB adapter')
    spec = importlib.util.spec_from_file_location('eyetracking_usb_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.open_stream()
    except module.USBError as exc:
        messages = {
            'device_binding_required': 'Pair your iPhone using your USB adapter first',
            'device_binding_read_unavailable': 'Mira couldn’t read the saved iPhone pairing. Close and reopen Mira, then reconnect.',
            'device_binding_protection_failed': 'Windows couldn’t open the iPhone pairing. Run Mira from your usual Windows account.',
            'device_binding_invalid': 'The saved iPhone pairing is damaged. Restore it using the USB bridge.',
            'apple_devices_usb_service_unavailable': 'Connect iPhone to this PC with a USB cable and open Apple Devices',
            'iphone_air_usb_required': 'Connect the paired iPhone to this PC with a USB cable and unlock it',
            'iphone_app_start_required': 'Open Mira on iPhone and start tracking',
        }
        raise ConnectionError(messages.get(str(exc), 'USB connection unavailable: ' + str(exc))) from None


@dataclass
class ARObservation:
    seq: int
    timestamp: float
    age_ms: float
    features: object = None
    blink: bool = False
    reason: str = 'TrueDepth can’t see your face'
    eye_origins: tuple = ()


def matrix(value):
    data = np.asarray(value, dtype=float)
    if data.shape != (16,) or not np.isfinite(data).all():
        raise ValueError('Invalid ARKit matrix')
    m = data.reshape((4,4), order='F')
    if not np.allclose(m[3], [0,0,0,1], atol=.001):
        raise ValueError('Invalid homogeneous transform')
    r = m[:3,:3]
    if not np.allclose(r.T@r, np.eye(3), atol=.03) or not .97 < np.linalg.det(r) < 1.03:
        raise ValueError('Invalid eye rotation')
    return m


class PacketDecoder:
    def __init__(self):
        self.session = None
        self.last_seq = -1
        self.last_timestamp = -1.

    def decode(self, packet):
        if packet.get('version') != 1:
            raise ValueError('Unsupported phone protocol')
        if 'seq' not in packet:  # The user started the listener; ARKit is warming up.
            return None
        session, seq = packet.get('session'), packet.get('seq')
        if not isinstance(session,str) or not 1 <= len(session) <= 64 or type(seq) is not int or seq < 1:
            raise ValueError('Invalid frame identity')
        if self.session is not None and session != self.session:
            raise ValueError('The iPhone session changed. Reconnect to continue.')
        timestamp, age = float(packet['timestamp']), float(packet['ageMs'])
        if not np.isfinite([timestamp,age]).all() or timestamp < 0 or age < 0:
            raise ValueError('Invalid phone timing')
        if seq <= self.last_seq:
            return None
        if timestamp <= self.last_timestamp:
            raise ValueError('Phone clock went backwards')
        self.session, self.last_seq, self.last_timestamp = session, seq, timestamp
        result = ARObservation(seq, timestamp, age)
        if age > 150:
            result.reason = 'iPhone data is out of date'
            return result
        if packet.get('trueDepth') is not True:
            raise ValueError('iPhone didn’t confirm TrueDepth tracking')
        if packet.get('tracked') is not True:
            return result
        face, left, right = (matrix(packet[k]) for k in ('face','leftEye','rightEye'))
        blink = np.asarray(packet['blink'],dtype=float)
        look = np.asarray(packet['lookAt'],dtype=float)
        if blink.shape != (2,) or look.shape != (3,) or not np.isfinite(np.r_[blink,look]).all() or np.any((blink<0)|(blink>1)):
            raise ValueError('Invalid gaze data')
        r = face[:3,:3]
        result.blink = bool(blink.max() > .6)
        if result.blink:
            result.reason = 'Blink detected'
            return result
        origins = [eye[:3,3] for eye in (left,right)]
        result.eye_origins = tuple(tuple(float(v) for v in origin) for origin in origins)
        directions = [eye[:3,2] for eye in (left,right)]  # ARKit +Z points toward the pupil.
        if not .1 < -face[2,3] < 3. or any(direction[2] <= .1 for direction in directions):
            result.reason = 'Looking outside the tracking area'
            return result
        # Intersect both rays with the camera's z=0 plane. Personal calibration
        # learns the offset/tilt between that plane and the actual monitor.
        intersections = [origin[:2]-origin[2]*direction[:2]/direction[2] for origin,direction in zip(origins,directions)]
        look_direction = look-np.mean(origins,axis=0)
        length = np.linalg.norm(look_direction)
        if length < 1e-5:
            result.reason = 'Can’t detect gaze direction'
            return result
        result.features = np.r_[*intersections, face[:3,3], r[:,:2].ravel(), *directions, look_direction/length]
        result.reason = 'TrueDepth can see your eyes and head'
        return result


@dataclass
class Sample:
    seq:int=0
    captured:float=0.
    features:object=None
    blink:bool=False
    error:str=''
    fps:float=0.
    reason:str=''
    source_age_ms:float=0.
    read_wait_ms:float=0.
    eye_origins:tuple=()


class ARKitWorker(threading.Thread):
    def __init__(self, connection):
        super().__init__(daemon=True, name='arkit-numeric')
        self.connection = connection
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.sample = Sample()
        self.failure = None
        self.frames = 0

    def latest(self):
        with self.lock:
            return self.sample

    def run(self):
        decoder = PacketDecoder()
        sock = None
        previous = None
        buffer = b''
        try:
            sock = self.connection.open(stop_event=self.stop_event)
            last_frame_at = time.monotonic()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(.4)
            while not self.stop_event.is_set():
                sent = time.perf_counter()
                sock.settimeout(.4)
                sock.sendall(b'GET\n')
                while b'\n' not in buffer:
                    remaining = .4-(time.perf_counter()-sent)
                    if remaining <= 0:
                        raise TimeoutError('iPhone didn’t respond in time')
                    if self.stop_event.is_set():
                        return
                    sock.settimeout(remaining)
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError('iPhone connection closed')
                    buffer += chunk
                    if len(buffer) > 262144:
                        raise ValueError('Oversized phone packet')
                line,buffer = buffer.split(b'\n',1)
                if buffer:
                    raise ValueError('iPhone sent unexpected data')
                now = time.monotonic()
                round_trip = time.perf_counter()-sent
                observation = decoder.decode(json.loads(line))
                if observation is None:
                    if now - last_frame_at >= 5:
                        raise ConnectionError('No new TrueDepth data. Check the connection.')
                    self.stop_event.wait(.005)
                    continue
                last_frame_at = now
                age = observation.age_ms/1000 + round_trip  # Full RTT, using Windows' precise clock.
                fps = (1/(observation.timestamp-previous) if previous is not None else 0)
                previous = observation.timestamp
                self.frames += 1
                sample = Sample(seq=observation.seq,captured=now-age,features=observation.features,
                    blink=observation.blink,reason=observation.reason,fps=fps,
                    source_age_ms=age*1000,read_wait_ms=round_trip*1000,
                    eye_origins=observation.eye_origins)
                with self.lock:
                    self.sample = sample
                self.stop_event.wait(.005)
        except TimeoutError:
            with self.lock:
                self.sample = Sample(error='iPhone didn’t respond in time. Check the connection and reconnect.')
        except Exception as exc:
            with self.lock:
                self.failure = exc
                self.sample = Sample(error=f'iPhone: {exc}')
        finally:
            if sock is not None:
                sock.close()
