import hashlib
import importlib.util
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from context_operator.foundations.local_models import OWNER, RUNTIME_ROOT, STANDALONE
from context_operator.foundations.speech.runtime import VoiceError, check_policy

MODEL_ROOT = RUNTIME_ROOT / 'models'
RUNTIME = RUNTIME_ROOT / 'runtimes/qwen3-reranker-0.6b/.venv'
MANIFEST = json.loads(Path(__file__).with_name('model.json').read_text(encoding='utf-8'))
MODEL_DIR = MODEL_ROOT / ('qwen3-reranker-0.6b-' + MANIFEST['revision'][:12])


class RankerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def assert_allowed():
    try:
        check_policy()
    except VoiceError as error:
        raise RankerError('local_model_start_blocked', str(error)) from error


def payload_store():
    if STANDALONE:
        return None
    policy = OWNER / 'local/model-storage.v1.json'
    try:
        policy.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RankerError('RANKER_PAYLOAD_OWNER', 'LocalModels storage policy is unreadable.') from error

    try:
        path = OWNER / 'scripts/model_payload_store.py'
        spec = importlib.util.spec_from_file_location('context_operator_payload_owner', path)
        sdk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk)
        store = sdk.PayloadStore(OWNER)
        for method in ('acquire_lease', 'ensure', 'mark_used', 'release_lease'):
            if not callable(getattr(store, method, None)):
                raise TypeError(f'PayloadStore.{method} is unavailable')
        return store
    except Exception as error:
        raise RankerError('RANKER_PAYLOAD_OWNER',
                          'LocalModels storage policy is installed but its payload SDK is unavailable or incompatible.') from error


def verify_artifact(path, artifact):
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise RankerError('RANKER_ARTIFACT', 'Missing pinned artifact: ' + artifact['path']) from error
    if size != artifact['size']:
        raise RankerError('RANKER_ARTIFACT', 'Pinned artifact size mismatch: ' + artifact['path'])
    digest = hashlib.sha256() if artifact['algorithm'] == 'sha256' else hashlib.sha1()
    if artifact['algorithm'] == 'git-blob-sha1':
        digest.update(('blob ' + str(artifact['size']) + '\0').encode('ascii'))
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != artifact['digest']:
        raise RankerError('RANKER_ARTIFACT', 'Pinned artifact digest mismatch: ' + artifact['path'])
