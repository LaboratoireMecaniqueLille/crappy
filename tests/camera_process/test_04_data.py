# coding: utf-8

import numpy as np
from threading import BrokenBarrierError, Event as ThreadEvent, Thread
from time import monotonic
from unittest.mock import MagicMock, patch

from .camera_process_test_base import CameraProcessTestBase, TestCameraProcess


class TestData(CameraProcessTestBase):
  """Tests image and metadata transfer from shared objects."""

  def setUp(self) -> None:
    super().setUp()
    self._receivers = list()

  def tearDown(self) -> None:
    try:
      for process, thread in self._receivers:
        process._stop_event.set()
        with process._condition:
          process._condition.notify_all()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
    finally:
      super().tearDown()

  def start_receive(self, timeout=5.0):
    """Signals entry into a real timed Condition wait in a test thread."""

    entered = ThreadEvent()
    results, errors = list(), list()
    process = self._process
    real_wait = process._condition.wait

    def wait(timeout=None):
      self.assertIsNotNone(timeout)
      self.assertGreater(timeout, 0)
      entered.set()
      return real_wait(timeout)

    patcher = patch.object(process._condition, 'wait', side_effect=wait)
    patcher.start()
    self.addCleanup(patcher.stop)

    def receive():
      try:
        results.append(process._get_data(timeout=timeout))
      except BaseException as exc:
        errors.append(exc)

    thread = Thread(target=receive, daemon=True)
    self._receivers.append((process, thread))
    thread.start()
    self.assertTrue(entered.wait(timeout=1.0))
    return thread, entered, results, errors

  def test_get_data(self) -> None:
    """Tests _get_data on missing, new, repeated and updated frames."""

    self._process = TestCameraProcess()
    shared = self.make_shared(shape=(2, 3), dtype=np.uint16)

    self.assertFalse(self._process._get_data())

    img = np.arange(6, dtype=np.uint16).reshape(2, 3)
    metadata = {'ImageUniqueID': 1, 't(s)': 1.0, 'meta': 'first'}
    self.write_image(shared, img, metadata)

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata, metadata)
    np.testing.assert_array_equal(self._process.img, img)

    # The same frame should not be handled twice.
    self.assertFalse(self._process._get_data())

    img_2 = img + 10
    metadata_2 = {'ImageUniqueID': 2, 't(s)': 2.0, 'meta': 'second'}
    self.write_image(shared, img_2, metadata_2)

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata, metadata_2)
    np.testing.assert_array_equal(self._process.img, img_2)

  def test_get_data_rejects_missing_metadata_dictionary(self) -> None:
    """Tests the defensive check for an uninitialized metadata dictionary."""

    self._process = TestCameraProcess()
    self._process._lock = MagicMock()

    with self.assertRaisesRegex(RuntimeError, 'metadata dictionary'):
      self._process._get_data()

  def test_wait_wakes_after_publication(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    thread, _, results, errors = self.start_receive()
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    self.write_image(shared, image, {'ImageUniqueID': 2, 't(s)': 0.1})
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [True])
    np.testing.assert_array_equal(self._process.img, image)
    self.assertEqual(self._process.metadata['ImageUniqueID'], 2)

  def test_spurious_notification_keeps_waiting(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    thread, entered, results, errors = self.start_receive()
    entered.clear()

    with shared.condition:
      shared.condition.notify_all()
    self.assertTrue(entered.wait(timeout=1.0))
    self.assertTrue(thread.is_alive())
    self.assertEqual(results, [])

    self.write_image(shared, np.zeros((3, 4), dtype=np.uint8))
    thread.join(timeout=1.0)
    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [True])

  def test_wait_times_out_without_frame(self) -> None:
    self._process = TestCameraProcess()
    self.make_shared()
    started = monotonic()

    self.assertFalse(self._process._get_data(timeout=0.03))

    self.assertGreaterEqual(monotonic() - started, 0.02)
    self.assertLess(monotonic() - started, 1.0)

  def test_stop_notification_wakes_without_frame(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    thread, _, results, errors = self.start_receive()

    shared.stop_event.set()
    with shared.condition:
      shared.condition.notify_all()
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [False])

  def test_timeout_notices_stop_without_notification(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    thread, _, results, errors = self.start_receive(timeout=0.05)

    shared.stop_event.set()
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])
    self.assertEqual(results, [False])

  def test_broken_barrier_notification_aborts_wait(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    thread, _, results, errors = self.start_receive()

    shared.barrier.abort()
    with shared.condition:
      shared.condition.notify_all()
    thread.join(timeout=1.0)

    self.assertFalse(thread.is_alive())
    self.assertEqual(results, [])
    self.assertEqual(len(errors), 1)
    self.assertIsInstance(errors[0], BrokenBarrierError)

  def test_stop_takes_priority_over_pending_frame(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    self.write_image(shared, np.ones((3, 4), dtype=np.uint8))
    shared.stop_event.set()

    self.assertFalse(self._process._get_data(timeout=0.1))
    self.assertIsNone(self._process.metadata['ImageUniqueID'])

  def test_wait_validates_timeout_and_condition(self) -> None:
    self._process = TestCameraProcess()
    self.make_shared()
    for timeout in (-1, float('inf'), float('nan')):
      with self.subTest(timeout=timeout), self.assertRaises(ValueError):
        self._process._get_data(timeout=timeout)
    for timeout in (None, 'invalid'):
      with self.subTest(timeout=timeout), self.assertRaises(TypeError):
        self._process._get_data(timeout=timeout)

    self._process._condition = None
    with self.assertRaisesRegex(RuntimeError, 'Condition'):
      self._process._get_data(timeout=0.1)

  def test_condition_failure_uses_existing_run_error_handling(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()

    with (patch.object(shared.condition, 'wait_for',
                       side_effect=OSError('condition failure')),
          self.assertRaisesRegex(OSError, 'condition failure')):
      self._process.run()

    self.assertTrue(shared.stop_event.is_set())
    self.assertTrue(self._process.finished.is_set())
