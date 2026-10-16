# coding: utf-8

"""The histogram worker is GUI-independent and belongs with the config tools."""

import logging
from inspect import signature, Parameter
from multiprocessing import Event, Queue
from queue import Empty
import unittest
from unittest.mock import Mock
import numpy as np

from crappy.tool.camera_config.config_tools import HistogramProcess
from crappy._failure import FailureManager


class TestHistogramProcess(unittest.TestCase):
  def setUp(self) -> None:
    self.stop_event = Event()
    self.processing_event = Event()
    self.img_in = Queue()
    self.img_out = Queue()
    self.log_queue = Queue()
    self.failure_queue = Queue()
    self.process = HistogramProcess(self.stop_event, self.processing_event,
                                    self.img_in, self.img_out,
                                    logging.CRITICAL, self.log_queue,
                                    self.failure_queue, 'test configuration',
                                    'test configuration.HistogramProcess')
    self.addCleanup(self.close_resources)

  def close_resources(self) -> None:
    """Release the child and all queues even if an assertion fails."""

    self.stop_event.set()
    if self.process.pid is not None:
      self.process.join(2.0)
      if self.process.is_alive():
        self.process.terminate()
        self.process.join(2.0)
      if self.process.is_alive():
        self.process.kill()
        self.process.join(2.0)
    for queue in (self.img_in, self.img_out, self.log_queue,
                  self.failure_queue):
      queue.cancel_join_thread()
      queue.close()

  def test_reporting_dependencies_are_required(self) -> None:
    for name in ('log_level', 'log_queue', 'failure_queue',
                 'block_name', 'owner_path'):
      with self.subTest(argument=name):
        self.assertIs(signature(HistogramProcess).parameters[name].default,
                      Parameter.empty)

  def test_queue_failure_does_not_skip_other_queue_or_close(self) -> None:
    """Every drain, feeder cancellation, and close is attempted before raising."""

    first, second = Mock(), Mock()
    first.get_nowait.side_effect = OSError('drain')
    second.get_nowait.side_effect = Empty
    errors = (first.get_nowait.side_effect, RuntimeError('cancel'))
    second.cancel_join_thread.side_effect = errors[1]
    self.process._img_in, self.process._img_out = first, second
    self.process._failure_manager = FailureManager(self.process.name, 'finish')
    self.process._failure_manager._logger = Mock()
    with self.assertRaises(ExceptionGroup) as caught:
      self.process._cleanup_queues()
    self.assertEqual(caught.exception.exceptions, errors)
    first.get_nowait.assert_called_once_with()
    first.cancel_join_thread.assert_called_once_with()
    second.cancel_join_thread.assert_called_once_with()
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()

  def test_histogram(self) -> None:
    """An actual process emits binary bars and optional auto-range markers."""

    self.assertFalse(self.process.is_alive())
    self.assertFalse(self.stop_event.is_set())
    self.assertFalse(self.processing_event.is_set())
    with self.assertRaises(Empty):
      self.img_in.get_nowait()
    with self.assertRaises(Empty):
      self.img_out.get_nowait()

    self.process.start()
    self.assertTrue(self.process.is_alive())

    for auto_range in (False, True):
      with self.subTest(auto_range=auto_range):
        self.img_in.put((np.full((32, 32), 128, dtype=np.uint8),
                         auto_range, 40, 200))
        try:
          histogram = self.img_out.get(timeout=5.0)
        except Empty:
          self.fail(f'No histogram received; worker exit code: {self.process.exitcode}')
        self.assertEqual(histogram.shape, (80, 512))
        self.assertEqual(histogram.dtype, np.dtype('uint8'))
        self.assertTrue(np.all(histogram[:, 256:258] == 0))
        if auto_range:
          self.assertTrue(np.all(histogram[:, 80] == 127))
          self.assertTrue(np.all(histogram[:, 400] == 127))
          self.assertEqual(set(np.unique(histogram)), {0, 127, 255})
        else:
          self.assertEqual(set(np.unique(histogram)), {0, 255})

    self.stop_event.set()
    self.process.join(2.0)
    self.assertFalse(self.process.is_alive())
    self.assertFalse(self.processing_event.is_set())
    self.assertEqual(self.process.exitcode, 0)
    with self.assertRaises(Empty):
      self.img_in.get_nowait()
    with self.assertRaises(Empty):
      self.img_out.get_nowait()


class FaultHistogram(HistogramProcess):
  """Injects hard exits and a large cleanup error into the real worker."""

  def __init__(self, *args, hard_exit=False, large_cleanup=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.hard_exit = hard_exit
    self.large_cleanup = large_cleanup

  def _hist_func(self, x, y, histo):
    if self.hard_exit:
      import os
      os._exit(13)
    return super()._hist_func(x, y, histo)

  def _cleanup_queues(self):
    super()._cleanup_queues()
    if self.large_cleanup:
      raise OSError('histogram cleanup ' + 'x' * 256_000)


class TestHistogramReporting(unittest.TestCase):
  """Checks configuration ownership without creating GUI widgets."""

  def setUp(self):
    from crappy._failure import FailureManager
    from crappy.tool.camera_config.base import ConfigurationLifecycle

    self.queues = [Queue() for _ in range(4)]
    self.img_in, self.img_out, self.log_queue, self.failure_queue = self.queues
    self.stop, self.processing = Event(), Event()
    self.manager = FailureManager('test configuration', 'configuration')
    self.manager._logger = Mock()
    self.worker = None
    self.lifecycle = None
    self.lifecycle_type = ConfigurationLifecycle
    self.addCleanup(self.release)

  def release(self):
    if self.lifecycle is not None:
      with self.manager.collect('test cleanup', raise_on_exit=False):
        with self.manager.step('close configuration resources'):
          self.lifecycle.close_resources()
    for queue in self.queues:
      queue.cancel_join_thread()
      queue.close()

  def start_worker(self, **kwargs):
    self.worker = FaultHistogram(
        self.stop, self.processing, self.img_in, self.img_out, None,
        self.log_queue, failure_queue=self.failure_queue,
        block_name=self.manager.block_name,
        owner_path='test configuration.HistogramProcess', **kwargs)
    self.lifecycle = self.lifecycle_type(
        self.stop, self.worker, (self.img_in, self.img_out), self.manager,
        self.failure_queue)
    self.worker.start()
    self.lifecycle.mark_histogram_started()

  def wait_for_failure(self):
    from time import monotonic, sleep

    deadline = monotonic() + 5
    while not self.lifecycle.failed and monotonic() < deadline:
      sleep(0.01)
    self.assertTrue(self.lifecycle.failed)

  def test_stop_during_worker_startup(self):
    """An early window close leaves time for a cold histogram worker to exit."""

    self.start_worker()
    self.lifecycle.close_resources()
    self.assertEqual(self.manager.snapshot(), ())
    self.assertTrue(self.lifecycle._process_closed)
    self.assertIsNone(self.lifecycle._receiver)

  def test_calculation_failure_reaches_configuration_without_duplicate_exit(self):
    from tests.block.block_test_base import exception_leaves

    self.start_worker()
    self.img_in.put(('invalid request',))
    self.wait_for_failure()
    self.lifecycle.close_resources()
    with self.assertRaises(ExceptionGroup) as caught:
      self.manager.raise_if_failed('configuration failures')
    leaves = exception_leaves(caught.exception)
    self.assertEqual(len(leaves), 1)
    self.assertIsInstance(leaves[0], ValueError)
    report = self.manager.snapshot()[0][0]
    self.assertEqual(report.block_name, 'test configuration')
    self.assertEqual(report.phase, 'configuration')
    self.assertTrue(report.owner_path.endswith('HistogramProcess'))
    self.manager._logger.error.assert_not_called()
    self.assertTrue(self.lifecycle._process_closed)
    self.assertIsNone(self.lifecycle._receiver)
    self.assertFalse(self.processing.is_set())

  def test_hard_exit_is_synthesized_after_draining(self):
    from crappy._exceptions import ProcessExitError
    from tests.block.block_test_base import exception_leaves

    self.start_worker(hard_exit=True)
    name, pid = self.worker.name, self.worker.pid
    self.img_in.put((np.ones((10, 10), dtype=np.uint8), False, 0, 255))
    self.wait_for_failure()
    with self.assertRaises(ExceptionGroup) as caught:
      self.lifecycle.close_resources()
    error, = exception_leaves(caught.exception)
    self.assertIsInstance(error, ProcessExitError)
    self.assertEqual((error.process_name, error.pid, error.exitcode),
                     (name, pid, 13))
    self.assertEqual(len(self.manager.snapshot()), 1)
    self.lifecycle.close_resources()
    self.assertEqual(len(self.manager.snapshot()), 1)

  def test_large_cleanup_report_is_flushed_before_receiver_shutdown(self):
    from tests.block.block_test_base import exception_leaves

    self.start_worker(large_cleanup=True)
    self.img_in.put(('invalid request',))
    self.wait_for_failure()
    self.lifecycle.close_resources()
    with self.assertRaises(ExceptionGroup) as caught:
      self.manager.raise_if_failed('configuration failures')
    leaves = exception_leaves(caught.exception)
    self.assertEqual(tuple(type(error) for error in leaves), (ValueError, OSError))
    self.assertGreater(len(str(leaves[1])), 256_000)
    self.assertEqual([report.phase for report, _ in self.manager.snapshot()],
                     ['configuration', 'configuration cleanup'])
    self.manager._logger.error.assert_not_called()

  def test_reporting_in_available_start_methods(self):
    import subprocess
    import sys
    from multiprocessing import get_all_start_methods

    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = ('from tests.camera_configuration_tools.test_histogram_process '
                  'import exercise_histogram_context; '
                  f'exercise_histogram_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script], text=True,
                                capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_histogram_context(method):
  """Exercises Queue flushing and hard exits in an isolated start method."""

  from multiprocessing import set_start_method
  set_start_method(method)
  names = ('test_stop_during_worker_startup',
           'test_calculation_failure_reaches_configuration_without_duplicate_exit',
           'test_hard_exit_is_synthesized_after_draining',
           'test_large_cleanup_report_is_flushed_before_receiver_shutdown')
  suite = unittest.TestSuite(TestHistogramReporting(name) for name in names)
  if not unittest.TextTestRunner().run(suite).wasSuccessful():
    raise SystemExit(1)
