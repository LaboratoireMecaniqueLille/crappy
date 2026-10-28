"""Fixture failures and timeout recovery must leave the next test usable."""

import json
import logging
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import psutil

from crappy import Block
from crappy.links import ImageLink
from tests.support.assertions import with_failure_manager
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.camera_process_test_case import CameraProcessTestBase
from tests.support.processes import run_python, stop_process, wait_until
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
import subprocess
import sys
import time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]))
print('parent output', flush=True)
print('parent diagnostic', file=sys.stderr, flush=True)
if sys.argv[2] == 'timeout':
    time.sleep(30)
'''
    def gone(pid):
      try:
        process = psutil.Process(pid)
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
      except psutil.NoSuchProcess:
        return True

    for mode, error_type in (('timeout', subprocess.TimeoutExpired),
                              ('leak', AssertionError)):
      with self.subTest(mode=mode), TemporaryDirectory() as folder:
        record = Path(folder) / 'pids.json'
        with self.assertRaises(error_type) as caught:
          run_python(['-c', script, str(record), mode], timeout=3.0)
        if mode == 'timeout':
          self.assertIn('parent output', caught.exception.stdout)
          self.assertIn('parent diagnostic', caught.exception.stderr)
        else:
          self.assertIn('parent output', str(caught.exception))
          self.assertIn('parent diagnostic', str(caught.exception))
        pids = json.loads(record.read_text())
        self.assertTrue(wait_until(lambda: all(gone(pid) for pid in pids)))
