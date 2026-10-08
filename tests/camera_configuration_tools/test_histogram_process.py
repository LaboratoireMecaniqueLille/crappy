# coding: utf-8

"""The histogram worker is GUI-independent and belongs with the config tools."""

import logging
from multiprocessing import Event, Queue
from queue import Empty
import unittest
from unittest.mock import Mock
import numpy as np

from crappy.tool.camera_config.config_tools import HistogramProcess


class TestHistogramProcess(unittest.TestCase):
  def setUp(self) -> None:
    self.stop_event = Event()
    self.processing_event = Event()
    self.img_in = Queue()
    self.img_out = Queue()
    self.log_queue = Queue()
    self.process = HistogramProcess(self.stop_event, self.processing_event,
                                    self.img_in, self.img_out,
                                    logging.CRITICAL, self.log_queue)
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
    for queue in (self.img_in, self.img_out, self.log_queue):
      queue.cancel_join_thread()
      queue.close()

  def test_queue_failure_does_not_skip_other_queue_or_close(self) -> None:
    """Every drain, feeder cancellation, and close is attempted before raising."""

    first, second = Mock(), Mock()
    first.get_nowait.side_effect = OSError('drain')
    second.get_nowait.side_effect = Empty
    errors = (first.get_nowait.side_effect, RuntimeError('cancel'))
    second.cancel_join_thread.side_effect = errors[1]
    self.process._img_in, self.process._img_out = first, second
    with self.assertRaises(ExceptionGroup) as caught:
      self.process._cleanup_queues()
    self.assertEqual(caught.exception.exceptions, errors)
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    second.cancel_join_thread.side_effect = None
    self.process._cleanup_queues()
    first.get_nowait.assert_called_once_with()
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
