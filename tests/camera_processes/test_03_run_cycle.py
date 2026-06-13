# coding: utf-8

import numpy as np

import crappy.blocks.camera_processes.camera_process as camera_process_module

from .camera_process_test_base import CameraProcessTestBase, TestCameraProcess


class TestRunCycle(CameraProcessTestBase):
  """Tests the CameraProcess execution cycle driven by run."""

  def test_normal_run(self) -> None:
    """Tests the nominal init/barrier/loop/finish sequence."""

    self._process = TestCameraProcess()
    shared = self.make_shared()
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    self.write_image(shared, img, {'ImageUniqueID': 3, 't(s)': 1.5})

    self._process.start()
    self._process.join(4.0)

    self.assertEqual(self._process.exitcode, 0)
    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
    self.assertEqual(self._process.loops.value, 1)
    self.assertEqual(self._process.last_image_id.value, 3)
    self.assertEqual(self._process.last_image_sum.value, float(np.sum(img)))

  def test_stop_event(self) -> None:
    """Tests that an already-set stop Event skips the loop body."""

    self._process = TestCameraProcess()
    shared = self.make_shared()
    shared.stop_event.set()

    self._process.start()
    self._process.join(4.0)

    self.assertEqual(self._process.exitcode, 0)
    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertFalse(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
    self.assertEqual(self._process.loops.value, 0)

  def test_handle_freq(self) -> None:
    """Tests the internal frequency bookkeeping during a run."""

    self._process = TestCameraProcess()
    shared = self.make_shared(display_freq=True)
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    self.write_image(shared, img)

    times = iter((0.0, 3.0))
    original_time = camera_process_module.time
    camera_process_module.time = lambda: next(times)

    try:
      self._process.run()

    finally:
      camera_process_module.time = original_time

    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())

    self.assertEqual(self._process.fps_count, 0)
    self.assertEqual(self._process._last_fps, 3.0)

  def test_raise_init(self) -> None:
    """Tests that an exception during init breaks the shared barrier."""

    self._process = TestCameraProcess(raise_in='init')
    shared = self.make_shared()

    self._process.start()
    self._process.join(4.0)

    self.assertNotEqual(self._process.exitcode, 0)
    self.assertTrue(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertFalse(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())

  def test_raise_loop(self) -> None:
    """Tests that an exception during loop sets the stop Event."""

    self._process = TestCameraProcess(raise_in='loop')
    shared = self.make_shared()
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    self.write_image(shared, img, {'ImageUniqueID': 7, 't(s)': 2.0})

    self._process.start()
    self._process.join(4.0)

    self.assertNotEqual(self._process.exitcode, 0)
    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
    self.assertEqual(self._process.loops.value, 1)
    self.assertEqual(self._process.last_image_id.value, 7)

  def test_raise_finish(self) -> None:
    """Tests that an exception during finish makes the process fail."""

    self._process = TestCameraProcess(raise_in='finish')
    shared = self.make_shared()
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    self.write_image(shared, img)

    self._process.start()
    self._process.join(4.0)

    self.assertEqual(self._process.exitcode, 0)
    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
    self.assertEqual(self._process.loops.value, 1)

  def test_broken_barrier(self) -> None:
    """Tests the behavior when the ready barrier is already broken."""

    self._process = TestCameraProcess()
    shared = self.make_shared()
    shared.barrier.abort()

    self._process.start()
    self._process.join(4.0)

    self.assertEqual(self._process.exitcode, 0)
    self.assertTrue(shared.barrier.broken)
    self.assertFalse(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertFalse(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())

  def test_keyboard_interrupt(self) -> None:
    """Tests that KeyboardInterrupt exits cleanly through finish."""

    self._process = TestCameraProcess(raise_in='keyboard')
    shared = self.make_shared()
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    self.write_image(shared, img)

    self._process.start()
    self._process.join(4.0)

    self.assertEqual(self._process.exitcode, 0)
    self.assertFalse(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())

    self.assertTrue(self._process.initialized.is_set())
    self.assertTrue(self._process.looped.is_set())
    self.assertTrue(self._process.finished.is_set())
