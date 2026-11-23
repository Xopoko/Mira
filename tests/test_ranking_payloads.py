from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from context_operator.foundations.ranking import owner, runtime


def send(stream, operation, request_id):
    stream.write(json.dumps({'op': operation, 'requestId': request_id}).encode('utf-8') + b'\n')
    stream.flush()


class RankingPayloadTests(unittest.TestCase):
    @contextmanager
    def runtime_pipe(self, store, ranker):
        reader, writer = os.pipe()
        events = queue.Queue()
        exits = queue.Queue()
        args = SimpleNamespace(model_dir=str(owner.MODEL_DIR))
        with os.fdopen(reader, 'rb') as stdin, os.fdopen(writer, 'wb') as output, \
             patch.object(runtime.sys, 'stdin', stdin), \
             patch.object(runtime.argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(runtime.owner, 'assert_allowed'), \
             patch.object(runtime.owner, 'payload_store', return_value=store), \
             patch.object(runtime, 'Reranker', return_value=ranker) as load, \
             patch.object(runtime, 'emit', side_effect=events.put):
            worker = threading.Thread(target=lambda: exits.put(runtime.main()), daemon=True)
            worker.start()
            try:
                yield output, events, exits, load
            finally:
                output.close()
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())

    def test_legacy_owner_without_policy_keeps_pinned_artifacts_and_configured_owner_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(owner, 'OWNER', Path(directory)), \
                patch.object(owner, 'STANDALONE', False):
            self.assertIsNone(owner.payload_store())
            with self.assertRaisesRegex(owner.RankerError, 'Missing pinned artifact') as missing:
                owner.verify_artifact(Path(directory) / 'weights', {'path': 'weights', 'size': 2})
            self.assertEqual(missing.exception.code, 'RANKER_ARTIFACT')

            policy = Path(directory) / 'local/model-storage.v1.json'
            policy.parent.mkdir()
            policy.write_text('{}', encoding='utf-8')
            sdk = Path(directory) / 'scripts/model_payload_store.py'
            for broken in (False, True):
                if broken:
                    sdk.parent.mkdir()
                    sdk.write_text('invalid python!', encoding='utf-8')
                with self.assertRaises(owner.RankerError) as failure:
                    owner.payload_store()
                self.assertEqual(failure.exception.code, 'RANKER_PAYLOAD_OWNER')

    def test_cold_close_or_eof_cancels_restore_and_releases_before_load(self):
        for how in ('close', 'eof'):
            with self.subTest(how=how):
                preparing = threading.Event()
                store = Mock()
                store.acquire_lease.return_value = 'fixture-lease'

                def ensure(_ids, *, cancel):
                    preparing.set()
                    if not cancel.wait(2):
                        raise TimeoutError('cold startup was not cancelled')
                    raise owner.RankerError('payload_cancelled', 'Fixture preparation cancelled.')

                store.ensure.side_effect = ensure
                with self.runtime_pipe(store, None) as (pipe, events, exits, load):
                    self.assertEqual(events.get(timeout=2)['type'], 'preparing')
                    self.assertTrue(preparing.wait(2))
                    if how == 'close':
                        send(pipe, 'close', 'stop')
                    else:
                        pipe.close()
                    self.assertEqual(exits.get(timeout=2), 0)
                    self.assertTrue(events.empty())
                    load.assert_not_called()
                store.release_lease.assert_called_once_with('fixture-lease', used=False)
                store.mark_used.assert_not_called()

    def test_lease_records_only_success_and_outlives_inflight_rank_on_close(self):
        store = Mock()
        store.acquire_lease.return_value = 'fixture-lease'
        ranker = Mock()
        ranking = threading.Event()
        cancelling = threading.Event()
        finish = threading.Event()
        marking = threading.Event()
        marking_cancel = queue.Queue()

        def rank(request, cancelled):
            if request['requestId'] == 'bad':
                raise owner.RankerError('RANKER_INPUT', 'Fixture rejection.')
            if request['requestId'] == 'during-mark':
                marking_cancel.put(cancelled)
            if request['requestId'] == 'pending':
                ranking.set()
                if cancelled.wait(2):
                    cancelling.set()
                finish.wait(2)
            return {'scores': [{'id': 'a', 'score': 1.}]}

        ranker.rank.side_effect = rank
        with self.runtime_pipe(store, ranker) as (pipe, events, exits, load):
            for state in ('preparing', 'prepared', 'ready'):
                self.assertEqual(events.get(timeout=2)['type'], state)
            load.assert_called_once()
            self.assertEqual([call[0] for call in store.mock_calls], ['acquire_lease', 'ensure'])

            send(pipe, 'rank', 'bad')
            self.assertEqual(events.get(timeout=2)['error']['code'], 'RANKER_INPUT')
            store.mark_used.assert_not_called()
            send(pipe, 'rank', 'good')
            self.assertEqual(events.get(timeout=2)['requestId'], 'good')
            store.mark_used.assert_called_once_with('fixture-lease')

            def mark_used(_token):
                marking.set()
                if not marking_cancel.get(timeout=2).wait(2):
                    raise TimeoutError('matching cancel was not received')

            store.mark_used.side_effect = mark_used
            send(pipe, 'rank', 'during-mark')
            self.assertTrue(marking.wait(2))
            send(pipe, 'cancel', 'during-mark')
            cancelled = events.get(timeout=2)
            self.assertEqual(cancelled['requestId'], 'during-mark')
            self.assertEqual(cancelled['error']['code'], 'RANKER_CANCELLED')
            self.assertNotIn('result', cancelled)

            send(pipe, 'rank', 'pending')
            self.assertTrue(ranking.wait(2))
            try:
                send(pipe, 'close', 'stop')
                self.assertTrue(cancelling.wait(2))
                store.release_lease.assert_not_called()
            finally:
                finish.set()
            self.assertEqual(exits.get(timeout=2), 0)
            self.assertTrue(events.empty())

        self.assertEqual(store.mark_used.call_count, 2)
        store.release_lease.assert_called_once_with('fixture-lease', used=False)


if __name__ == '__main__':
    unittest.main()
