from concurrent.futures import ThreadPoolExecutor
import json
import queue
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from context_operator.core.contracts import Element, Rect
from context_operator.providers import models


class RankerProviderTests(unittest.TestCase):
    def setUp(self):
        self.ranker = models.Ranker()
        self.addCleanup(lambda: self.ranker.close().wait(2))
        self.elements = [Element('button', 'Button', 'Button', Rect(0, 0, 10, 10), 1)]

    def process(self, *, ready=True):
        output, exited = queue.SimpleQueue(), threading.Event()
        child = MagicMock()
        child.stdout.__enter__.return_value = child.stdout
        child.stdout.readline.side_effect = lambda _limit: output.get()
        child.poll.side_effect = lambda: 0 if exited.is_set() else None

        def wait(*, timeout):
            if not exited.wait(timeout):
                raise subprocess.TimeoutExpired('reranker fixture', timeout)
            return 0

        def close_input():
            output.put('')
            exited.set()

        def emit(event):
            output.put(json.dumps(event) + '\n')

        child.wait.side_effect = wait
        child.stdin.close.side_effect = close_input
        model = models.MANIFEST['model']
        emit({'type': 'preparing', 'model': model})
        if ready:
            emit({'type': 'prepared', 'model': model})
            emit({'type': 'ready', 'model': model, 'revision': models.MANIFEST['revision'], 'device': 'cpu'})
        return child, emit, exited

    def test_commands_remain_available_while_the_owner_prepares_weights(self):
        child, emit, exited = self.process(ready=False)
        launched = threading.Event()

        def launch(*args, **kwargs):
            launched.set()
            return child

        with patch.object(models, 'check_policy'), patch.object(models.subprocess, 'Popen', side_effect=launch), ThreadPoolExecutor(1) as pool:
            warm = pool.submit(self.ranker.warm)
            try:
                self.assertTrue(launched.wait(2))
                with self.assertRaisesRegex(RuntimeError, 'Target matcher is busy'):
                    self.ranker.rank('button', self.elements, wait_for_ready=False)
            finally:
                emit({'error': {'message': 'Not enough disk space'}})
            with self.assertRaisesRegex(RuntimeError, 'Not enough disk space'):
                warm.result(timeout=2)
            self.assertTrue(exited.wait(2))

    def test_timeout_and_invalid_scores_retire_only_the_failed_process(self):
        with patch.object(models, 'check_policy'), patch.object(models.subprocess, 'Popen') as launch:
            for response_id in (None, 'foreign', 'button'):
                with self.subTest(response_id=response_id):
                    child, emit, exited = self.process()
                    launch.return_value = child
                    if response_id is not None:
                        def reply(line):
                            emit({'requestId': json.loads(line)['requestId'],
                                  'result': {'scores': [{'id': response_id, 'score': 2.0}]}})
                        child.stdin.write.side_effect = reply
                    if response_id == 'button':
                        self.assertEqual(self.ranker.rank('button', self.elements), {'button': 2.0})
                    else:
                        with patch.object(models, 'RANKER_REQUEST_SECONDS', .1), self.assertRaises(OSError) as failure:
                            self.ranker.rank('button', self.elements)
                        self.assertEqual(isinstance(failure.exception, TimeoutError), response_id is None)
                        self.assertFalse(self.ranker.is_ready)
                        self.assertTrue(exited.wait(2))
            self.assertEqual(launch.call_count, 3)

    def test_close_accounts_for_a_launch_in_progress(self):
        for stage in ('policy', 'popen'):
            with self.subTest(stage=stage):
                ranker = models.Ranker()
                child, _, exited = self.process()
                entered, proceed = threading.Event(), threading.Event()

                def block(*args, **kwargs):
                    entered.set()
                    proceed.wait(2)
                    return child

                with patch.object(models, 'check_policy', side_effect=block if stage == 'policy' else None), \
                     patch.object(models.subprocess, 'Popen', side_effect=block if stage == 'popen' else None) as launch, \
                     ThreadPoolExecutor(1) as pool:
                    warm = pool.submit(ranker.warm)
                    try:
                        self.assertTrue(entered.wait(2))
                        stopped = ranker.close()
                        self.assertIs(stopped, ranker.close())
                        if stage == 'popen':
                            self.assertFalse(stopped.is_set())
                    finally:
                        proceed.set()
                    with self.assertRaisesRegex(RuntimeError, 'Target matcher closed'):
                        warm.result(timeout=2)
                    self.assertTrue(stopped.wait(2))
                self.assertEqual(launch.called, stage == 'popen')
                self.assertEqual(exited.is_set(), stage == 'popen')
