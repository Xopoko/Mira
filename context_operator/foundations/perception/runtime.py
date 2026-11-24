"""OmniParser detector + PaddleOCR, isolated from ASR; masked pixels stay in memory.

The worker receives original-resolution RGB through an anonymous pipe. Only
semantic tile results are cached; snapshot IDs, window ownership and execution
pixel hashes are always assigned afresh by desktop.py. No caption model is loaded.
"""
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from context_operator.foundations.local_models import OWNER, RUNTIME_ROOT
from context_operator.foundations.speech.runtime import VoiceError, check_policy

ROOT = RUNTIME_ROOT / 'jev-perception'
SOURCE = Path(__file__).resolve().parent
STARTUP_SECONDS = 90
INFERENCE_SECONDS = 20
MAX_PIXELS = 80_000_000
MAX_RESPONSE = 4 * 1024 * 1024
TILE_SIZE = 960
DETECTOR_SIZE = 640
TILE_OVERLAP = 64


class PerceptionError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def admit():
    try:
        check_policy()
    except VoiceError as error:
        raise PerceptionError('local_model_start_blocked', str(error)) from error


def read_exact(stream, count):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        received = stream.readinto(view[offset:])
        if not received:
            raise EOFError('Perception pipe closed')
        offset += received
    return data


class PerceptionClient:
    def __init__(self):
        self.child = None
        self.responses = None
        self.metadata = None

    def _wait(self, timeout, stopped=None):
        deadline = time.monotonic() + timeout
        while True:
            if stopped is not None and stopped.is_set():
                self.close()
                raise PerceptionError('stopped', 'Emergency stop interrupted visual perception.')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise PerceptionError('perception_timeout', 'The isolated perception runtime exceeded its deadline. No perception or executable targets were returned.')
            try:
                response = self.responses.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if isinstance(response, Exception):
                self.close()
                raise PerceptionError('perception_unavailable', 'The isolated perception worker exited or returned an invalid response; no visual results are available.') from response
            if 'error' in response:
                error = response['error']
                raise PerceptionError(error['code'], error['message'])
            return response

    def ready(self, stopped=None):
        admit()
        if self.child is not None:
            if self.child.poll() is None:
                return self.metadata
            self.close()
        python = ROOT / '.venv-v5' / 'Scripts' / 'python.exe'
        if not python.is_file():
            raise PerceptionError('perception_unavailable', f'OmniParser/PaddleOCR Python is missing: {python}')
        responses = queue.Queue(maxsize=2)
        self.responses = responses
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        if env.get('MIRA_LOCAL_MODELS_ROOT'):
            env['MIRA_LOCAL_MODELS_ROOT'] = str(OWNER)
        self.child = subprocess.Popen(
            [str(python), '-u', str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(ROOT), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env=env,
        )
        child = self.child

        def receive():
            try:
                while True:
                    line = child.stdout.readline(MAX_RESPONSE + 1)
                    if not line or len(line) > MAX_RESPONSE or not line.endswith(b'\n'):
                        raise EOFError('Perception response missing or oversized')
                    responses.put(json.loads(line))
            except Exception as error:
                try:
                    responses.put_nowait(error)
                except queue.Full:
                    pass

        threading.Thread(target=receive, name='perception-results', daemon=True).start()
        try:
            self.metadata = self._wait(STARTUP_SECONDS, stopped)['ready']
            return self.metadata
        except BaseException:
            self.close()
            raise

    def observe(self, image, stopped=None):
        self.ready(stopped)
        admit()
        started = time.perf_counter()
        payload = image.tobytes('raw', 'RGB')
        child, responses = self.child, self.responses
        # A pipe writer must not prevent stop/deadline handling when a child hangs.
        def send():
            try:
                child.stdin.write(struct.pack('<III', image.width, image.height, len(payload)))
                child.stdin.write(payload)
                child.stdin.flush()
            except (OSError, ValueError) as error:
                try:
                    responses.put_nowait(error)
                except queue.Full:
                    pass
        threading.Thread(target=send, name='perception-frame', daemon=True).start()
        result = self._wait(INFERENCE_SECONDS, stopped)
        result['timings']['perceptionIpcMs'] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def close(self):
        child, self.child = self.child, None
        self.metadata = None
        if child is None:
            return
        # Only the exact child created here is ever terminated; shared models are untouched.
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)
        for stream in (child.stdin, child.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def verified_models():
    manifest = json.loads((SOURCE / 'perception-models.json').read_text(encoding='utf-8'))
    paths = {}
    for asset in manifest['assets']:
        paths.update({ROOT / 'models' / asset['directory'] / name: digest for name, digest in asset['files'].items()}
                     if 'directory' in asset else {ROOT / 'models' / asset['name']: asset['sha256']})
    paths.update({ROOT / 'models' / asset['name']: asset['sha256'] for asset in manifest['derived'].values()})
    for path, expected in paths.items():
        try:
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as error:
            raise PerceptionError('perception_model_missing', f'Cannot read perception model file: {path}') from error
        if actual != expected:
            raise PerceptionError('perception_model_mismatch', 'Perception model integrity failed. Reinstall the pinned owner-local weights; no inference was performed.')
    return manifest


def rect_from_box(box, width, height):
    left = max(0, min(width, math.floor(float(box[0]))))
    top = max(0, min(height, math.floor(float(box[1]))))
    right = max(left, min(width, math.ceil(float(box[2]))))
    bottom = max(top, min(height, math.ceil(float(box[3]))))
    return [left, top, right - left, bottom - top]


def overlap(first, second):
    a, b = first['rect'], second['rect']
    area = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return area / max(1, min(a[2] * a[3], b[2] * b[3]))


class OnnxCpuModel:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, tensor):
        return self.session.run(None, {self.input_name: tensor})


class OmniDetector:
    def __init__(self):
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        self.np, self.torch, self.ops = np, torch, ops
        self.letterbox = LetterBox(new_shape=(DETECTOR_SIZE, DETECTOR_SIZE), auto=False)
        self.model = OnnxCpuModel(ROOT / 'models' / 'omniparser-v2.onnx')

    def predict(self, bgr):
        padded = self.letterbox(image=bgr)
        tensor = self.np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=self.np.float32)
        tensor /= 255
        outputs = self.model.predict(tensor[None])
        boxes = self.ops.non_max_suppression(self.torch.from_numpy(outputs[0]), conf_thres=0.05,
                                            iou_thres=0.5, max_det=300)[0]
        if len(boxes):
            boxes[:, :4] = self.ops.scale_boxes((DETECTOR_SIZE, DETECTOR_SIZE), boxes[:, :4], bgr.shape)
        return boxes[:, :4].tolist(), boxes[:, 4].tolist()


class PaddleTextReader:
    """PaddleOCR v5 detection, original-resolution crops, exact FP32 ONNX recognition."""

    def __init__(self):
        import numpy as np
        import yaml
        from paddleocr import TextDetection
        from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes
        from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg, ToBatch, CTCLabelDecode
        self.np = np
        self.detector = TextDetection(
            model_name='PP-OCRv5_mobile_det', model_dir=str(ROOT / 'models' / 'PP-OCRv5_mobile_det_infer'),
            device='cpu', enable_mkldnn=True, cpu_threads=4, limit_side_len=DETECTOR_SIZE,
            limit_type='max', box_thresh=0.6, unclip_ratio=1.5,
        )
        config = yaml.safe_load((ROOT / 'models' / 'cyrillic_PP-OCRv5_mobile_rec_infer' / 'inference.yml').read_text(encoding='utf-8'))
        shape = next(op['RecResizeImg']['image_shape'] for op in config['PreProcess']['transform_ops'] if 'RecResizeImg' in op)
        self.normalize = OCRReisizeNormImg(rec_image_shape=shape)
        self.batch = ToBatch()
        self.decode = CTCLabelDecode(character_list=config['PostProcess']['character_dict'])
        self.recognizer = OnnxCpuModel(ROOT / 'models' / 'cyrillic_PP-OCRv5_mobile_rec.onnx')
        self.crop = CropByPolys(det_box_type='quad')
        self.sort = SortQuadBoxes()
        self.cache = OrderedDict()

    def predict(self, bgr):
        started = time.perf_counter()
        admit()
        detected = self.detector.predict(bgr)[0]
        polygons = self.sort(detected['dt_polys'])
        crops = list(self.crop(bgr, polygons))
        detection_ms = (time.perf_counter() - started) * 1000
        pending, keys = {}, []
        hits = 0
        for crop in crops:
            crop = self.np.ascontiguousarray(crop)
            key = (crop.shape, hashlib.sha256(memoryview(crop)).digest())
            keys.append(key)
            if key in self.cache:
                hits += 1
                self.cache.move_to_end(key)
            else:
                pending[key] = crop
        started = time.perf_counter()
        ordered = sorted(pending, key=lambda key: pending[key].shape[1] / pending[key].shape[0])
        for start in range(0, len(ordered), 4):
            batch_keys = ordered[start:start + 4]
            tensor = self.batch(self.normalize([pending[key] for key in batch_keys]))[0]
            admit()
            texts, scores = self.decode(self.recognizer.predict(tensor))
            for key, text, score in zip(batch_keys, texts, scores):
                self.cache[key] = (text, float(score))
        items = []
        for polygon, key in zip(polygons, keys):
            text, score = self.cache[key]
            if text.strip() and score >= 0.5:
                points = self.np.asarray(polygon)
                rect = rect_from_box([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
                                     bgr.shape[1], bgr.shape[0])
                if rect[2] and rect[3]:
                    items.append({'source': 'ocr', 'rect': rect, 'label': text.strip(), 'confidence': score})
        # Only text/confidence/hash survives; raw crops and executable identities do not.
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        return items, {
            'textDetectionMs': detection_ms, 'textRecognitionMs': (time.perf_counter() - started) * 1000,
            'textCrops': len(crops), 'cachedTextCrops': hits,
        }


class PerceptionPipeline:
    def __init__(self):
        started = time.perf_counter()
        admit()
        manifest = verified_models()
        os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'ultralytics')
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['PADDLE_PDX_CACHE_HOME'] = str(ROOT / 'paddlex')
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import numpy as np
        import torch
        from ultralytics import settings
        settings.update({'sync': False})
        torch.set_num_threads(4)
        self.np = np
        self.detector = OmniDetector()
        admit()
        self.ocr = PaddleTextReader()
        from concurrent.futures import ThreadPoolExecutor
        self.detector_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix='OmniDetector')
        self.cache = OrderedDict()
        self.metadata = {
            'detector': 'microsoft/OmniParser-v2.0', 'detectorRevision': manifest['detector']['revision'],
            'ocr': 'PaddleOCR 3.3.2 / PP-OCRv5_mobile_det + cyrillic_PP-OCRv5_mobile_rec',
            'languages': ['Cyrillic', 'English'], 'device': 'cpu', 'captioning': False,
            'tileSize': TILE_SIZE, 'tileOverlap': TILE_OVERLAP,
            'detectorInputSize': DETECTOR_SIZE, 'ocrDetectionMaxSide': DETECTOR_SIZE,
            'ocrRecognition': 'original-resolution crops; FP32 ONNX Runtime CPU',
            'cpuThreadsPerModel': 4, 'parallelDetectorOcr': True,
            'detectorRuntime': 'FP32 ONNX Runtime CPU', 'textCropCache': 2048,
            'startupMs': round((time.perf_counter() - started) * 1000, 2),
        }

    def close(self):
        self.detector_thread.shutdown(wait=True, cancel_futures=True)

    def detect(self, bgr):
        started = time.perf_counter()
        result = self.detector.predict(bgr)
        return result, (time.perf_counter() - started) * 1000

    def tile(self, rgb):
        started = time.perf_counter()
        height, width = rgb.shape[:2]
        bgr = self.np.ascontiguousarray(rgb[:, :, ::-1])
        admit()
        detection = self.detector_thread.submit(self.detect, bgr)
        try:
            # Separate models consume the same read-only pixels concurrently;
            # Paddle predictors themselves remain strictly single-threaded here.
            items, stats = self.ocr.predict(bgr)
            stats['ocrMs'] = (time.perf_counter() - started) * 1000
        finally:
            (boxes, confidences), detector_ms = detection.result()
        for box, confidence in zip(boxes, confidences):
            rect = rect_from_box(box, width, height)
            if rect[2] > 2 and rect[3] > 2:
                items.append({'source': 'omniparser', 'rect': rect, 'label': 'Unlabelled icon', 'confidence': float(confidence)})
        stats['detectorMs'] = detector_ms
        return items, stats

    def observe(self, width, height, payload):
        started = time.perf_counter()
        admit()  # Cached tiles never bypass model-use policy.
        rgb = self.np.frombuffer(payload, dtype=self.np.uint8).reshape(height, width, 3)
        items, next_cache = [], OrderedDict()
        hits = tiles = 0
        stats = dict.fromkeys(('ocrMs', 'detectorMs', 'textDetectionMs', 'textRecognitionMs', 'textCrops', 'cachedTextCrops'), 0)
        for top in range(0, max(1, height - TILE_OVERLAP), TILE_SIZE - TILE_OVERLAP):
            for left in range(0, max(1, width - TILE_OVERLAP), TILE_SIZE - TILE_OVERLAP):
                tile = self.np.ascontiguousarray(rgb[top:min(top + TILE_SIZE, height), left:min(left + TILE_SIZE, width)])
                key = (tile.shape, hashlib.sha256(memoryview(tile)).digest())
                tiles += 1
                cached = next_cache.get(key, self.cache.get(key))
                if cached is None:
                    cached, tile_stats = self.tile(tile)
                    for name, value in tile_stats.items():
                        stats[name] += value
                else:
                    hits += 1
                if len(next_cache) < 80:
                    next_cache[key] = cached
                for item in cached:
                    x, y, w, h = item['rect']
                    items.append({**item, 'rect': [left + x, top + y, w, h]})
        self.cache = next_cache
        # Overlap margins yield duplicate boxes; never fuse nearby independent controls.
        unique = []
        for item in sorted(items, key=lambda entry: entry['confidence'], reverse=True):
            if not any(item['source'] == old['source'] and item['label'].casefold() == old['label'].casefold()
                       and overlap(item, old) > 0.8 for old in unique):
                unique.append(item)
        return {'items': unique, 'perception': {
                    **self.metadata, 'tiles': tiles, 'cachedTiles': hits,
                    'textCrops': stats['textCrops'], 'cachedTextCrops': stats['cachedTextCrops'],
                }, 'timings': {
                    **{name: round(stats[name], 2) for name in ('ocrMs', 'detectorMs', 'textDetectionMs', 'textRecognitionMs')},
                    'perceptionMs': round((time.perf_counter() - started) * 1000, 2),
                }}


def safe_error(error):
    if isinstance(error, PerceptionError):
        return {'error': {'code': error.code, 'message': str(error)}}
    return {'error': {'code': 'perception_failed', 'message': f'OmniParser/PaddleOCR failed ({type(error).__name__}); no fabricated or fallback detections were returned. Reinstall the isolated pinned runtime if necessary.'}}


def worker():
    incoming = sys.stdin.buffer
    # Native libraries can write directly to descriptor 1. Preserve a separate wire
    # descriptor, then discard all library diagnostics without recording screen data.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    with open(os.devnull, 'w') as quiet:
        os.dup2(quiet.fileno(), sys.stdout.fileno())
        os.dup2(quiet.fileno(), sys.stderr.fileno())
        def emit(value):
            output.write((json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8'))
        try:
            pipeline = PerceptionPipeline()
            emit({'ready': pipeline.metadata})
        except Exception as error:
            emit(safe_error(error))
            return
        while True:
            try:
                width, height, size = struct.unpack('<III', read_exact(incoming, 12))
                if not width or not height or width * height > MAX_PIXELS or size != width * height * 3:
                    raise PerceptionError('perception_frame_invalid', 'Invalid original-resolution RGB frame.')
                payload = read_exact(incoming, size)
                emit(pipeline.observe(width, height, payload))
                del payload
            except EOFError:
                pipeline.close()
                return
            except Exception as error:
                emit(safe_error(error))


def main():
    if sys.argv[1:] == ['--worker']:
        worker()
        return
    import argparse
    parser = argparse.ArgumentParser(description='Run real CPU perception on an explicitly supplied local image, never on the desktop.')
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    from PIL import Image
    client = PerceptionClient()
    try:
        with Image.open(args.image) as original, original.convert('RGB') as image:
            print(json.dumps(client.observe(image), ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
