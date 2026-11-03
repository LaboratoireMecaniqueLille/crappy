"""Fixture failures and timeout recovery must leave the next test usable."""

import json
import logging
from contextlib import nullcontext
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import psutil

from crappy import Block
from crappy.links import ImageLink
from tests.support.assertions import with_failure_manager
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.camera_process_test_case import CameraProcessTestBase
from tests.support.processes import (
    _signal_process_group, run_python, stop_process, wait_until)
from tests.support.vision import StubVisionBlock, VisionTestBase


class SetupFailure(RuntimeError):
  """A test-owned sentinel, distinct from cleanup failures."""


class TestTestSupport(unittest.TestCase):
  def run_failed_setup(self, case):
    result = unittest.TestResult()
    case.run(result)
    self.assertFalse(result.failures)
    self.assertEqual(len(result.errors), 1, result.errors)
    self.assertIn('SetupFailure', result.errors[0][1])

  def test_failed_block_setup_restores_logging_and_registration(self):
    logger = logging.getLogger('crappy')
    previous = (logger.handlers[:], logger.level, logger.disabled,
                logger.propagate, logging.root.manager.disable)
    registry = Block.classes.copy()

    class FailingSetup(BlockTestBase):
      def setUp(self):
        super().setUp()
        self._block = TestBlock()
        type('FailedSetupBlock', (Block,), {'__module__': __name__})
        raise SetupFailure('after Block acquisition')

      def runTest(self):
        raise AssertionError('body must not run after failed setup')

    self.run_failed_setup(FailingSetup())
    self.assertFalse(Block.instances)
    self.assertEqual(Block.classes, registry)
    self.assertEqual((logger.handlers, logger.level, logger.disabled,
                      logger.propagate, logging.root.manager.disable), previous)

  def test_failed_camera_setup_releases_workers_and_shared_objects(self):
    owned = {}

    class FailingSetup(CameraProcessTestBase):
      def setUp(self):
        super().setUp()
        self.make_shared()
        owned.update(worker=self._process, manager=self._manager._process,
                     queues=self._queues[:], pipes=self.make_pipe())
        self._process.start()
        if not self._process.initialized.wait(3.0):
          raise AssertionError('worker never initialized')
        raise SetupFailure('after worker startup')

      def runTest(self):
        raise AssertionError('body must not run after failed setup')

    self.run_failed_setup(FailingSetup())
    self.assertFalse(owned['worker'].is_alive())
    self.assertFalse(owned['manager'].is_alive())
    self.assertTrue(all(pipe.closed for pipe in owned['pipes']))
    for queue in owned['queues']:
      with self.assertRaises((ValueError, OSError)):
        queue.put('closed')

  def test_failed_vision_setup_releases_memory_and_manager(self):
    owned = {}

    class FailingSetup(VisionTestBase):
      def setUp(self):
        super().setUp()
        source = with_failure_manager(StubVisionBlock(img_shape=(2, 3), img_dtype='uint8'))
        target = with_failure_manager(StubVisionBlock())
        link = ImageLink(source, target)
        manager = self.make_manager(source)
        source.set_shared_objects()
        source.prepare()
        owned.update(name=link.get_buffers()[0], manager=manager._process)
        raise SetupFailure('after shared-memory allocation')

      def runTest(self):
        raise AssertionError('body must not run after failed setup')

    self.run_failed_setup(FailingSetup())
    self.assertFalse(owned['manager'].is_alive())
    with self.assertRaises(FileNotFoundError):
      attached = SharedMemory(name=owned['name'])
      self.addCleanup(attached.close)

  def test_emergency_worker_cleanup_fails_and_stops_the_worker(self):
    context = get_context('spawn')
    stop = context.Event()
    process = context.Process(target=stop.wait)
    process.start()
    self.addCleanup(stop_process, process, stop.set)
    with self.assertRaises(AssertionError):
      stop_process(process, lambda: None, timeout=0.05)
    self.assertFalse(process.is_alive())

  def test_subprocess_failures_reap_descendants_and_preserve_output(self):
    script = '''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
if sys.argv[2] == 'stubborn':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]))
print('parent output', flush=True)
print('parent diagnostic', file=sys.stderr, flush=True)
if sys.argv[2] != 'leak':
    time.sleep(30)
'''
    def gone(pid):
      try:
        process = psutil.Process(pid)
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
      except psutil.NoSuchProcess:
        return True

    cases = [('timeout', subprocess.TimeoutExpired, False),
             ('leak', AssertionError, False)]
    if os.name == 'posix':
      cases.append(('timeout', subprocess.TimeoutExpired, True))
      cases.append(('stubborn', subprocess.TimeoutExpired, False))

    for mode, error_type, denied_sigkill in cases:
      with self.subTest(mode=mode, denied_sigkill=denied_sigkill), \
           TemporaryDirectory() as folder:
        record = Path(folder) / 'pids.json'
        killpg = getattr(os, 'killpg', None)

        def reject_sigkill(group, sig):
          if sig == signal.SIGKILL:
            raise PermissionError('process group can no longer be signaled')
          return killpg(group, sig)

        signaling = (patch('tests.support.processes.os.killpg', reject_sigkill)
                     if denied_sigkill else nullcontext())
        with signaling, self.assertRaises(error_type) as caught:
          run_python(['-c', script, str(record), mode], timeout=3.0)
        if error_type is subprocess.TimeoutExpired:
          self.assertIn('parent output', caught.exception.stdout)
          self.assertIn('parent diagnostic', caught.exception.stderr)
        else:
          self.assertIn('parent output', str(caught.exception))
          self.assertIn('parent diagnostic', str(caught.exception))
        pids = json.loads(record.read_text())
        self.assertTrue(wait_until(lambda: all(gone(pid) for pid in pids)))

  def test_group_permission_errors_require_completed_shutdown(self):
    for parent_exited, live_child in ((True, False), (False, False),
                                       (True, True)):
      with self.subTest(parent_exited=parent_exited, live_child=live_child):
        process = Mock(pid=12345)
        process.poll.side_effect = [None, 0 if parent_exited else None]
        children = [Mock()] if live_child else []
        permission = PermissionError('process group cannot be signaled')
        with patch('tests.support.processes._owned_processes',
                   return_value=children), \
             patch('tests.support.processes.os.killpg',
                   side_effect=permission, create=True):
          if parent_exited and not live_child:
            _signal_process_group(process, {}, signal.SIGTERM)
          else:
            with self.assertRaises(PermissionError) as caught:
              _signal_process_group(process, {}, signal.SIGTERM)
            self.assertIs(caught.exception, permission)
