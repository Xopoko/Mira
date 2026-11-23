"""Local, final-only Parakeet speech recognition; stdout is flushed JSONL."""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from context_operator.foundations import local_models

SAMPLE_RATE = 16000
WINDOW_SHIFT = 512
MAX_AUDIO_SAMPLES = 12 * SAMPLE_RATE
MAX_FINAL_AGE_SECONDS = 5.0
CAPTURE_QUEUE_BLOCKS = 16
POLICY_SCRIPT = local_models.OWNER / 'scripts/Get-LocalModelStartPolicy.ps1'
POLICY_STATE = local_models.OWNER / 'local/state/model-start-policy.v1.json'
_policy_lock = threading.Lock()
_allowed_policy = None
_allowed_signature = None
SUPPORTED_RATES = (16000, 32000, 48000, 96000)


class VoiceError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


class Control:
    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.stop_lock = threading.RLock()
        self.output_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.stream = None
        self.latest_id = 0
        self.reason = "completed"
        self.error = None
        self.output_closed = False

    def emit(self, event_type, **fields):
        with self.output_lock:
            if self.output_closed:
                return
            try:
                print(json.dumps({"type": event_type, **fields}, ensure_ascii=False,
                                 allow_nan=False), flush=True)
            except (BrokenPipeError, OSError):
                self.output_closed = True
                self.request_stop("stdout_closed")

    def request_stop(self, reason, error=None):
        # Also callable from the PortAudio callback: never write to stdout here.
        with self.stop_lock:
            if not self.stop.is_set():
                self.reason = reason
                self.error = error
                self.stop.set()

    def start_stream(self, stream):
        with self.stream_lock:
            if self.stop.is_set():
                stream.close()
                return False
            self.stream = stream
            try:
                stream.start()
            except BaseException:
                self.stream = None
                stream.close()
                raise
            return True

    def close_stream(self):
        with self.stream_lock:
            stream, self.stream = self.stream, None
            if stream is not None:
                try:
                    stream.abort()
                finally:
                    stream.close()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate policy property")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def policy_signature():
    try:
        script = POLICY_SCRIPT.stat()
        try:
            state = POLICY_STATE.stat()
        except FileNotFoundError:
            state = None
    except OSError:
        return None
    return tuple((file.st_mtime_ns, file.st_ctime_ns, file.st_size, file.st_mode)
                 if file is not None else None for file in (script, state))


def check_policy():
    global _allowed_policy, _allowed_signature
    if local_models.STANDALONE:
        if local_models.owner_is_absent():
            return
        raise VoiceError("local_model_start_not_allowed",
                         "LocalModels was detected. Restart Mira before starting local inference.")
    with _policy_lock:
        before = policy_signature()
        if before is not None and _allowed_policy is not None and before == _allowed_signature:
            return _allowed_policy
        _allowed_policy = _allowed_signature = None
        executable = shutil.which("pwsh")
        if executable is None or not POLICY_SCRIPT.is_file():
            raise VoiceError("policy_unavailable", "Local inference needs the LocalModels policy "
                             "and PowerShell 7, or standalone local inference must be enabled.")
        try:
            result = subprocess.run(
                [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(POLICY_SCRIPT)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8-sig", errors="strict", timeout=10, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                raise ValueError(f"Policy owner exited with code {result.returncode}")
            policy = json.loads(result.stdout, object_pairs_hook=reject_duplicate_keys,
                                parse_constant=reject_nonfinite)
            if (not isinstance(policy, dict)
                    or policy.get("schema") != "localmodels.start-policy.v1"
                    or policy.get("state") != "allowed"
                    or policy.get("blocked") is not False
                    or type(policy.get("generation")) is not int
                    or policy["generation"] < 0
                    or policy.get("source") not in ("default", "persisted")
                    or policy.get("errorCode") not in (None, "")):
                raise ValueError("Policy state is not a valid allowed receipt")
            after = policy_signature()
            if before is None or after is None or before != after:
                raise ValueError("Policy metadata changed or was unreadable during admission")
        except (OSError, subprocess.SubprocessError, ValueError, UnicodeError) as exc:
            raise VoiceError("local_model_start_not_allowed",
                             f"LocalModels policy denied or could not verify admission: {exc}. "
                             "No model will be loaded or decoded; do not retry automatically.") from exc
        _allowed_signature, _allowed_policy = after, policy
        return policy


def start_lifetime_watchers(control):
    parent_pid = os.getppid()
    kernel32 = None
    parent_handle = None
    if os.name == "nt":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        parent_handle = kernel32.OpenProcess(0x00100000, False, parent_pid)
        if not parent_handle:
            raise VoiceError("parent_watch_unavailable",
                             "Cannot monitor the launching process; refusing unattended capture.")

    def supervise():
        try:
            while not control.stop.wait(0.2):
                if kernel32 is not None:
                    status = kernel32.WaitForSingleObject(parent_handle, 0)
                    if status != 0x00000102:  # WAIT_TIMEOUT is the only live-parent result.
                        control.request_stop("parent_exited")
                elif os.getppid() != parent_pid:
                    control.request_stop("parent_exited")
        finally:
            # Close capture even if stdout is blocked or the main thread is in native code.
            try:
                control.close_stream()
            finally:
                if parent_handle:
                    kernel32.CloseHandle(parent_handle)

    threading.Thread(target=supervise, name="voice-lifetime", daemon=True).start()

    if sys.stdin is not None and not sys.stdin.isatty():
        if kernel32 is not None:
            import msvcrt
            stdin_handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            kernel32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            kernel32.ReadFile.restype = wintypes.BOOL
            kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                               ctypes.c_void_p]
            kernel32.PeekNamedPipe.restype = wintypes.BOOL

        def read_stdin():
            if kernel32 is None:
                return os.read(sys.stdin.fileno(), 256)
            # Do not leave a synchronous pipe read pending while native modules
            # initialize. On Windows that can stall NumPy's DLL initialization.
            available = wintypes.DWORD()
            while not control.stop.is_set():
                if not kernel32.PeekNamedPipe(stdin_handle, None, 0, None,
                                              ctypes.byref(available), None):
                    return b""
                if available.value:
                    buffer = ctypes.create_string_buffer(min(256, available.value))
                    count = wintypes.DWORD()
                    if not kernel32.ReadFile(stdin_handle, buffer, len(buffer),
                                            ctypes.byref(count), None):
                        return b""
                    return buffer.raw[:count.value]
                control.stop.wait(0.05)
            return b""

        def watch_stdin():
            pending = b""
            try:
                while not control.stop.is_set():
                    chunk = read_stdin()
                    if not chunk:
                        control.request_stop("stdin_eof")
                        return
                    pending += chunk
                    if len(pending) > 1024:
                        control.request_stop("invalid_stdin_command")
                        return
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line.strip().lower() in (b"stop", b"shutdown", b'{"type":"stop"}'):
                            control.request_stop("stdin_stop")
                            return
            except (OSError, ValueError):
                control.request_stop("stdin_closed")
        threading.Thread(target=watch_stdin, name="voice-stdin", daemon=True).start()


def input_devices(sd):
    host_apis = sd.query_hostapis()
    return [{"index": index, "name": info["name"],
             "hostApi": host_apis[info["hostapi"]]["name"],
             "hostApiIndex": int(info["hostapi"]),
             "maxInputChannels": int(info["max_input_channels"]),
             "defaultSampleRate": float(info["default_samplerate"])}
            for index, info in enumerate(sd.query_devices()) if info["max_input_channels"] > 0]


def microphone_matches(devices, name, host_api=None, index=None):
    if host_api is not None:
        matches = [device for device in devices if device['name'].casefold() == name.casefold()
                   and device['hostApi'].casefold() == host_api.casefold()]
        if len(matches) > 1:
            return [device for device in matches if device['index'] == index] or matches
        return matches
    matches = [device for device in devices if name.casefold() in device['name'].casefold()]
    if not matches:
        return []

    def host_rank(device):
        name = device['hostApi'].casefold()
        return next((rank for rank, marker in enumerate(('wasapi', 'directsound', 'mme', 'wdm'))
                     if marker in name), 4)

    best_rank = min(map(host_rank, matches))
    return [device for device in matches if host_rank(device) == best_rank]


def select_microphone(sd, index, expected_name, *, system_default=False,
                      default_if_missing=False, host_api=None):
    devices = input_devices(sd)

    def fail(message):
        raise VoiceError("microphone_selection_failed", message +
                         " Run --list-devices and supply --device INDEX --device-name EXPECTED_NAME.",
                         devices=devices)

    fallback_from = None
    if system_default or (index is None and expected_name is None):
        device = None
    elif index is not None and host_api is None:
        matches = [device for device in devices if device["index"] == index]
        if not matches:
            fail(f"Input device index {index} does not exist.")
        device = matches[0]
        if expected_name and expected_name.casefold() not in device["name"].casefold():
            fail(f"Device {index} is {device['name']!r}, not expected {expected_name!r}.")
    else:
        desired = expected_name
        matches = microphone_matches(devices, desired, host_api, index)
        if not matches and default_if_missing:
            fallback_from = desired
        elif not matches:
            fail(f"Desired microphone {desired!r} was not found; no default fallback is permitted.")
        elif len(matches) != 1:
            fail(f"Desired microphone {desired!r} is ambiguous within the preferred host API.")
        device = matches[0] if matches else None

    if device is None:
        default = sd.default.device[0]
        device = next((device for device in devices if device['index'] == default), None)
        if device is None:
            message = 'No system default microphone is available. Connect a microphone and try again.'
            if fallback_from:
                message = f'{fallback_from} isn’t connected. ' + message
            raise VoiceError('microphone_selection_failed', message, devices=devices)

    native_rate = int(round(device["defaultSampleRate"]))
    rates = list(dict.fromkeys([SAMPLE_RATE, native_rate, 48000, 32000, 96000]))
    errors = []
    for rate in rates:
        if rate not in SUPPORTED_RATES:
            continue
        try:
            sd.check_input_settings(device=device["index"], channels=1,
                                    dtype="float32", samplerate=rate)
        except (sd.PortAudioError, ValueError) as exc:
            errors.append(f"{rate} Hz: {exc}")
        else:
            selected = {**device, "sampleRate": rate}
            if fallback_from:
                selected['fallbackFrom'] = fallback_from
            return selected
    fail("Selected microphone does not support mono capture at 16/32/48/96 kHz. "
         "Choose its WASAPI endpoint or configure a supported sample rate. " + "; ".join(errors))


class Decimator:
    """Streaming, anti-aliased integer decimation; no arbitrary-rate conversion."""
    def __init__(self, np, rate):
        self.np = np
        self.factor = rate // SAMPLE_RATE
        if rate not in SUPPORTED_RATES:
            raise VoiceError("unsupported_sample_rate",
                             "Use 16, 32, 48 or 96 kHz mono audio; 44.1 kHz is not silently resampled.")
        if self.factor > 1:
            taps = 64 * self.factor + 1
            offset = np.arange(taps) - (taps - 1) / 2
            cutoff = 0.45 / self.factor
            kernel = 2 * cutoff * np.sinc(2 * cutoff * offset) * np.hamming(taps)
            self.kernel = (kernel / kernel.sum()).astype(np.float32)
            self.history = np.zeros(taps - 1, dtype=np.float32)

    def process(self, samples):
        if self.factor == 1:
            return samples
        combined = self.np.concatenate((self.history, samples))
        filtered = self.np.convolve(combined, self.kernel, mode="valid")
        self.history = combined[-len(self.history):].copy()
        return filtered[::self.factor].copy()


class Decoder:
    def __init__(self, control, recognizer):
        self.control = control
        self.recognizer = recognizer
        self.jobs = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self.run, name="voice-asr", daemon=True)
        self.worker.start()

    def speech_started(self):
        with self.control.lock:
            self.control.latest_id += 1
            utterance_id = self.control.latest_id
            self.drop_pending("superseded")
            self.control.emit("speech_start", utteranceId=utterance_id)
            return utterance_id

    def drop_pending(self, reason):
        try:
            job = self.jobs.get_nowait()
        except queue.Empty:
            return
        self.jobs.task_done()
        self.control.emit("ignored", utteranceId=job[0], reason=reason)

    def submit(self, utterance_id, samples, ended_at_ms, ended_monotonic):
        with self.control.lock:
            if self.control.stop.is_set() or utterance_id != self.control.latest_id:
                self.control.emit("ignored", utteranceId=utterance_id, reason="superseded")
                return
            try:
                self.jobs.put_nowait((utterance_id, samples, ended_at_ms, ended_monotonic))
            except queue.Full:
                self.control.emit("ignored", utteranceId=utterance_id, reason="asr_queue_full")

    def current(self, utterance_id, ended_monotonic):
        return (not self.control.stop.is_set() and utterance_id == self.control.latest_id
                and time.monotonic() - ended_monotonic <= MAX_FINAL_AGE_SECONDS)

    def run(self):
        while not self.control.stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            utterance_id, samples, ended_at_ms, ended_monotonic = job
            try:
                if not self.current(utterance_id, ended_monotonic):
                    self.control.emit("ignored", utteranceId=utterance_id, reason="superseded_or_stale")
                    continue
                check_policy()
                if not self.current(utterance_id, ended_monotonic):
                    self.control.emit("ignored", utteranceId=utterance_id, reason="superseded_or_stale")
                    continue
                started = time.monotonic()
                stream = self.recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                asr_ms = round((time.monotonic() - started) * 1000)
                with self.control.lock:
                    if not self.current(utterance_id, ended_monotonic):
                        self.control.emit("ignored", utteranceId=utterance_id, reason="superseded_or_stale")
                    elif not text:
                        self.control.emit("ignored", utteranceId=utterance_id, reason="empty_transcript")
                    else:
                        self.control.emit("final", utteranceId=utterance_id, text=text,
                                          audioMs=round(len(samples) * 1000 / SAMPLE_RATE),
                                          asrMs=asr_ms, endedAtMs=ended_at_ms,
                                          afterSpeechMs=round((time.monotonic()-ended_monotonic)*1000))
            except Exception as exc:
                error = exc if isinstance(exc, VoiceError) else VoiceError("asr_failed", str(exc))
                self.control.request_stop("error", error)
            finally:
                self.jobs.task_done()
                # Do not retain the last utterance while idle.
                job = samples = stream = None

    def drain(self):
        deadline = time.monotonic() + MAX_FINAL_AGE_SECONDS
        with self.jobs.all_tasks_done:
            while self.jobs.unfinished_tasks and not self.control.stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.jobs.all_tasks_done.wait(min(remaining, 0.1))


class Segmenter:
    def __init__(self, np, model, decoder):
        self.np = np
        self.model = model
        self.decoder = decoder
        self.silence_samples = model.min_silence_duration_samples()
        # Unvoiced initial consonants (such as the first sound in "stop") precede VAD onset.
        preroll = round(0.3 * SAMPLE_RATE) + model.min_speech_duration_samples()
        self.history = deque(maxlen=math.ceil(preroll / WINDOW_SHIFT))
        self.overlap = model.window_size() - WINDOW_SHIFT
        if self.overlap not in (0, 64):
            raise VoiceError("unsupported_vad_window", "Expected Silero v4/v5 512-sample windows.")
        self.context = np.zeros(self.overlap, dtype=np.float32)
        self.frames = []
        self.sample_count = 0
        self.active_id = None
        self.discarding = False

    def accept(self, samples, captured_epoch, captured_monotonic):
        if len(samples) != WINDOW_SHIFT:
            raise VoiceError("invalid_audio_block", "Audio block has an unexpected sample count.")
        self.history.append(samples)
        if self.overlap:
            window = self.np.concatenate((self.context, samples))
            self.context = samples[-self.overlap:]
        else:
            window = samples
        check_policy()
        speech = self.model.is_speech(window)
        if speech and self.active_id is None:
            self.active_id = self.decoder.speech_started()
            self.frames = list(self.history)
            self.sample_count = len(self.frames) * WINDOW_SHIFT
        elif self.active_id is not None:
            self.sample_count += WINDOW_SHIFT
            if not self.discarding:
                self.frames.append(samples)

        if self.active_id is None:
            return
        # Keep at most 12 seconds plus endpoint silence. The low-level Silero
        # state continues without forced chunking even after audio is discarded.
        if not self.discarding and self.sample_count - self.silence_samples > MAX_AUDIO_SAMPLES:
            self.discarding = True
            self.frames.clear()
            self.decoder.control.emit("ignored", utteranceId=self.active_id, reason="utterance_too_long")
        if speech:
            return
        if not self.discarding:
            count = self.sample_count - self.silence_samples
            if count > 0:
                audio = self.np.concatenate(self.frames)[:count].copy()
                silence_seconds = self.silence_samples / SAMPLE_RATE
                self.decoder.submit(self.active_id, audio,
                                    round((captured_epoch - silence_seconds) * 1000),
                                    captured_monotonic - silence_seconds)
        self.frames.clear()
        self.active_id = None
        self.sample_count = 0
        self.discarding = False

    def discard_unfinished(self):
        if self.active_id is not None and not self.discarding:
            self.decoder.control.emit("ignored", utteranceId=self.active_id,
                                      reason="capture_ended_during_speech")
        self.frames.clear()
        self.history.clear()
        self.active_id = None


def capture_microphone(args, np, sd, control, segmenter, device):
    rate = device["sampleRate"]
    converter = Decimator(np, rate)
    block_size = WINDOW_SHIFT * converter.factor
    blocks = queue.Queue(maxsize=CAPTURE_QUEUE_BLOCKS)

    def callback(indata, frames, timing, status):
        if control.stop.is_set():
            raise sd.CallbackAbort
        if status or frames != block_size:
            control.request_stop("error", VoiceError(
                "capture_overflow" if status else "invalid_capture_block",
                f"Capture continuity lost ({status or 'unexpected block size'}); all pending speech discarded."))
            raise sd.CallbackAbort
        try:
            blocks.put_nowait((indata[:, 0].copy(), time.time(), time.monotonic()))
        except queue.Full:
            control.request_stop("error", VoiceError(
                "capture_queue_overflow", "VAD could not keep up with capture; all pending speech discarded."))
            raise sd.CallbackAbort

    stream = sd.InputStream(device=device["index"], samplerate=rate, channels=1,
                            dtype="float32", blocksize=block_size, callback=callback)
    if not control.start_stream(stream):
        return
    deadline = time.monotonic() + args.duration if args.duration is not None else None
    try:
        control.emit("ready", source="microphone", device=device,
                     sampleRate=SAMPLE_RATE, silenceMs=args.silence_ms, maxUtteranceMs=12000)
        while not control.stop.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                control.reason = "duration"
                break
            try:
                samples, epoch, monotonic = blocks.get(timeout=0.1)
            except queue.Empty:
                if not stream.active and not control.stop.is_set():
                    raise VoiceError("capture_stopped", "Microphone stream stopped unexpectedly.")
                continue
            if control.stop.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                control.reason = "duration"
                break
            segmenter.accept(converter.process(samples), epoch, monotonic)
    finally:
        control.close_stream()
        segmenter.discard_unfinished()


def replay_wav(args, np, control, segmenter, source):
    rate = source.getframerate()
    converter = Decimator(np, rate)
    block_size = WINDOW_SHIFT * converter.factor
    period = WINDOW_SHIFT / SAMPLE_RATE
    control.emit("ready", source="wav", device=None, inputSampleRate=rate,
                 sampleRate=SAMPLE_RATE, silenceMs=args.silence_ms, maxUtteranceMs=12000)
    started = time.monotonic()
    played_frames = 0

    def deliver(samples):
        nonlocal played_frames
        played_frames += 1
        deadline = started + played_frames * period
        if control.stop.wait(max(0, deadline - time.monotonic())):
            return False
        # If execution stalls, do not accelerate old audio into a command backlog.
        if time.monotonic() - deadline > CAPTURE_QUEUE_BLOCKS * period:
            raise VoiceError("wav_replay_stalled", "Realtime WAV pacing stalled; replay stopped rather than catching up.")
        segmenter.accept(converter.process(samples), time.time(), time.monotonic())
        return True

    try:
        total_frames = source.getnframes()
        read_frames = 0
        limit = int(args.duration * rate) if args.duration is not None else total_frames
        while not control.stop.is_set() and read_frames < min(total_frames, limit):
            frame_count = min(block_size, total_frames - read_frames, limit - read_frames)
            raw = source.readframes(frame_count)
            if len(raw) != frame_count * 2:
                raise VoiceError("wav_truncated", "WAV data ended before its declared frame count.")
            read_frames += frame_count
            if frame_count < block_size and read_frames == limit and read_frames < total_frames:
                control.reason = "duration"
                return  # Never pad a duration-truncated block into an artificial endpoint.
            samples = np.zeros(block_size, dtype=np.float32)
            samples[:frame_count] = np.frombuffer(raw, dtype="<i2") / 32768.0
            if not deliver(samples):
                return
        if control.stop.is_set():
            return
        if read_frames < total_frames:
            control.reason = "duration"
            return  # A duration cutoff is not an utterance endpoint.
        # Real silence, passed through the same resampler/VAD, finalizes the last
        # phrase; never call a forced VAD flush on unfinished microphone speech.
        trailing_blocks = math.ceil((args.silence_ms / 1000 + 0.3) / period)
        silence = np.zeros(block_size, dtype=np.float32)
        for _ in range(trailing_blocks):
            if not deliver(silence):
                return
        control.reason = "wav_eof"
    finally:
        segmenter.discard_unfinished()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=Path, help="Directory with INT8 encoder/decoder/joiner and tokens.txt")
    parser.add_argument("--vad-model", type=Path, help="Local Silero VAD ONNX file")
    parser.add_argument("--device", type=int, help="Explicit input device index; no system defaults are changed")
    parser.add_argument("--device-name", help="Expected microphone name substring (case-insensitive)")
    parser.add_argument("--device-host-api", help="Select the exact device name in this host API")
    parser.add_argument("--system-default", action="store_true", help="Use the system’s default input device")
    parser.add_argument("--default-if-missing", action="store_true", help="Use the system default when the named device is missing")
    parser.add_argument("--list-devices", action="store_true", help="List input device identities as JSON, without capture/models")
    parser.add_argument("--wav", type=Path, help="Replay an existing PCM mono16 WAV at realtime speed, without microphone capture")
    parser.add_argument("--duration", type=float, help="Maximum source capture/replay seconds; unfinished speech is discarded")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--silence-ms", type=int, default=550)
    args = parser.parse_args()
    if args.device is not None and args.device < 0:
        parser.error("--device must be a nonnegative input device index")
    if args.device_name is not None:
        args.device_name = args.device_name.strip()
        if not args.device_name:
            parser.error("--device-name must not be empty")
    if (args.device_host_api or args.default_if_missing) and not args.device_name:
        parser.error("--device-host-api and --default-if-missing require --device-name")
    if args.system_default and (args.device is not None or args.device_name is not None):
        parser.error("--system-default cannot be combined with an explicit device")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and greater than zero")
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    if not 100 <= args.silence_ms <= 2000:
        parser.error("--silence-ms must be between 100 and 2000")
    if args.wav is not None and (args.device is not None or args.device_name is not None or args.system_default):
        parser.error("--wav cannot be combined with microphone selection")
    if not args.list_devices and (args.model_dir is None or args.vad_model is None):
        parser.error("--model-dir and --vad-model are required unless --list-devices is used")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    args = arguments()
    control = Control()
    decoder = None
    wav_source = None
    try:
        if args.list_devices:
            import sounddevice as sd
            print(json.dumps({"devices": input_devices(sd)}, ensure_ascii=False), flush=True)
            return 0
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda _signum, _frame: control.request_stop("signal"))
        start_lifetime_watchers(control)
        required = [args.model_dir / filename for filename in
                    ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")]
        required.append(args.vad_model)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise VoiceError("model_files_missing", "Required local model files are missing.", paths=missing)
        import numpy as np
        import sherpa_onnx
        device = None
        if args.wav is not None:
            wav_source = wave.open(str(args.wav), "rb")
            if (wav_source.getnchannels() != 1 or wav_source.getsampwidth() != 2
                    or wav_source.getcomptype() != "NONE"):
                raise VoiceError("invalid_wav_format", "--wav requires uncompressed PCM mono16 WAV.")
            if wav_source.getframerate() not in SUPPORTED_RATES:
                raise VoiceError("unsupported_sample_rate", "--wav requires 16, 32, 48 or 96 kHz audio.")
        else:
            import sounddevice as sd
            device = select_microphone(sd, args.device, args.device_name, system_default=args.system_default,
                                       default_if_missing=args.default_if_missing, host_api=args.device_host_api)
        if control.stop.is_set():
            return 0
        check_policy()
        if control.stop.is_set():
            return 0
        print("Loading local Parakeet ONNX weights...", file=sys.stderr, flush=True)
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(required[0]), decoder=str(required[1]), joiner=str(required[2]),
            tokens=str(required[3]), model_type="nemo_transducer", provider="cpu",
            num_threads=args.threads, sample_rate=SAMPLE_RATE, decoding_method="greedy_search",
        )
        print("Parakeet loaded; initializing local Silero VAD...", file=sys.stderr, flush=True)
        check_policy()
        if control.stop.is_set():
            return 0
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(args.vad_model)
        config.silero_vad.window_size = WINDOW_SHIFT
        config.silero_vad.min_silence_duration = args.silence_ms / 1000
        config.silero_vad.min_speech_duration = 0.064
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = "cpu"
        vad = sherpa_onnx.VadModel.create(config)
        decoder = Decoder(control, recognizer)
        segmenter = Segmenter(np, vad, decoder)
        if wav_source is not None:
            replay_wav(args, np, control, segmenter, wav_source)
        else:
            capture_microphone(args, np, sd, control, segmenter, device)
        if not control.stop.is_set():
            decoder.drain()
    except KeyboardInterrupt:
        control.request_stop("signal")
    except Exception as exc:
        error = exc if isinstance(exc, VoiceError) else VoiceError("runtime_failed", str(exc))
        control.request_stop("error", error)
    finally:
        control.request_stop(control.reason)
        control.close_stream()
        if wav_source is not None:
            wav_source.close()
        if decoder is not None:
            decoder.drop_pending("stopped")
            decoder.worker.join(timeout=2)
        if control.error is not None:
            print(f"{control.error.code}: {control.error}", file=sys.stderr, flush=True)
            control.emit("error", code=control.error.code, message=str(control.error), **control.error.details)
        if not args.list_devices:
            control.emit("stopped", reason=control.reason)
        if decoder is not None and decoder.worker.is_alive():
            # ONNX decode is not interruptible. Capture is already closed and all
            # admission invalidated; avoid waiting indefinitely during shutdown.
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except OSError:
                pass
            os._exit(1 if control.error else 0)
    return 1 if control.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
