"""Explicit numeric-only phone transport and private LAN discovery."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass, field
from functools import partial
from ipaddress import IPv4Address, IPv4Network
from itertools import chain
import json
import math
import os
import socket
import time

WIFI_PORT = 39472
WIFI_PHONE_NEXT_STEP = 'On iPhone, open Mira, check that Connection says Wi‑Fi, and tap Start Tracking.'
_DISCOVERY_REPLY_LIMIT = 16_384
_LAN = tuple(IPv4Network(value) for value in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))


class WifiPhoneNotFound(ConnectionError):
    def __init__(self):
        super().__init__(f'Can’t find iPhone on your Wi‑Fi. {WIFI_PHONE_NEXT_STEP}')


def _private(address):
    return any(address in network for network in _LAN)


class _IPAddrRow(ctypes.Structure):
    _fields_ = [('address', ctypes.c_ubyte * 4), ('index', ctypes.c_uint32),
                ('mask', ctypes.c_uint32), ('broadcast', ctypes.c_uint32),
                ('reassembly_size', ctypes.c_uint32), ('reserved', ctypes.c_uint16),
                ('type', ctypes.c_uint16)]


class _IPAddrTable(ctypes.Structure):
    _fields_ = [('count', ctypes.c_uint32), ('rows', _IPAddrRow * 1)]


def _interface_addresses():
    if os.name != 'nt':
        try:
            return socket.gethostbyname_ex(socket.gethostname())[2]
        except OSError:
            return []
    get_table = ctypes.windll.iphlpapi.GetIpAddrTable
    get_table.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32]
    get_table.restype = ctypes.c_uint32
    size = ctypes.c_uint32(1024)
    for _ in range(2):
        buffer = ctypes.create_string_buffer(size.value)
        result = get_table(buffer, ctypes.byref(size), 0)
        if result == 0:
            count = ctypes.c_uint32.from_buffer(buffer).value
            rows = (_IPAddrRow * count).from_buffer(buffer, _IPAddrTable.rows.offset)
            return [socket.inet_ntoa(bytes(row.address)) for row in rows]
        if result != 122:  # ERROR_INSUFFICIENT_BUFFER publishes the required size.
            break
    return []


def _primary_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            # UDP connect selects the route without sending a datagram.
            route.connect(('192.0.2.1', 9))
            return IPv4Address(route.getsockname()[0])
    except OSError:
        return None


def _candidate_groups():
    primary = _primary_address()
    addresses = [IPv4Address(host) for host in _interface_addresses()]
    if primary is not None:
        addresses.insert(0, primary)
    own = set(addresses)
    networks = dict.fromkeys(IPv4Network((address, 24), strict=False)
                             for address in addresses if _private(address))
    groups = [[str(host) for host in network.hosts() if host not in own] for network in networks]
    if primary is not None and _private(primary):
        return groups[:1] + [list(chain.from_iterable(groups[1:]))]
    return [list(chain.from_iterable(groups))]


def _responds(host, *, port, timeout, stop_event):
    if stop_event is not None and stop_event.is_set():
        return False
    deadline = time.monotonic() + timeout + .25
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or stop_event is not None and stop_event.is_set():
                return False
            connection.settimeout(remaining)
            connection.sendall(b'GET\n')
            data = bytearray()
            while len(data) <= _DISCOVERY_REPLY_LIMIT:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                connection.settimeout(remaining)
                chunk = connection.recv(min(4096, _DISCOVERY_REPLY_LIMIT + 1 - len(data)))
                if not chunk:
                    return False
                line, newline, _ = chunk.partition(b'\n')
                data.extend(line)
                if newline:
                    reply = json.loads(data)
                    if not isinstance(reply, dict) or type(reply.get('version')) is not int or reply['version'] != 1:
                        return False
                    age = reply.get('ageMs')
                    return type(age) is int or (type(age) is float and math.isfinite(age))
    except (OSError, ValueError, RecursionError):
        return False
    return False


def find_phones(candidates=None, port=WIFI_PORT, timeout=0.35, workers=64, *, stop_event=None):
    """Find Mira replies on the primary /24 before trying other local /24s."""
    groups = (_candidate_groups() if candidates is None else
              [list(dict.fromkeys(str(IPv4Address(host)) for host in candidates))])
    probe = partial(_responds, port=port, timeout=timeout, stop_event=stop_event)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mira-discovery') as pool:
        for group in groups:
            if stop_event is not None and stop_event.is_set():
                break
            found = [host for host, answered in zip(group, pool.map(probe, group)) if answered]
            if found:
                return found
    return []


def wifi_address(value):
    try:
        address = IPv4Address(value.strip())
    except (ValueError, AttributeError):
        raise ValueError('Enter the local IPv4 address shown on iPhone, without http:// or a port.') from None
    if not _private(address):
        raise ValueError('Wi‑Fi requires a private home network address.')
    return str(address)


@dataclass(frozen=True)
class PhoneConnection:
    transport: str
    host: str = ''
    found_host: str = field(default='', init=False, compare=False)

    def __post_init__(self):
        if self.transport not in ('usb', 'wifi'):
            raise ValueError('Unknown iPhone connection type')
        if self.transport == 'wifi':
            host = self.host.strip() if isinstance(self.host, str) else self.host
            object.__setattr__(self, 'host', '' if host == '' else wifi_address(host))

    @classmethod
    def from_settings(cls, settings):
        return cls(settings['phone_transport'], settings.get('phone_wifi_host', ''))

    def open(self, *, stop_event=None):
        if self.transport == 'usb':
            from ..foundations.gaze.arkit import open_usb_stream
            return open_usb_stream()
        object.__setattr__(self, 'found_host', '')
        if stop_event is not None and stop_event.is_set():
            raise ConnectionAbortedError('iPhone connection cancelled')
        if self.host:
            try:
                return socket.create_connection((self.host, WIFI_PORT), timeout=2)
            except OSError:
                pass
        for host in find_phones(stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                break
            try:
                connection = socket.create_connection((host, WIFI_PORT), timeout=2)
            except OSError:
                continue
            object.__setattr__(self, 'found_host', host)
            return connection
        if stop_event is not None and stop_event.is_set():
            raise ConnectionAbortedError('iPhone connection cancelled')
        raise WifiPhoneNotFound()
