# coding: utf-8

"""Tk integration checks for the neutral configuration lifecycle."""

from multiprocessing import Queue
from unittest.mock import patch

from .camera_configuration_test_base import ConfigurationWindowTestBase, tk
from crappy.tool.camera_config import CameraConfig
import crappy.tool.camera_config.camera_config as camera_config_module


class TestConfigurationLifecycle(ConfigurationWindowTestBase):
  def test_run_closes_normally(self) -> None:
    """run() starts acquisition, waits, and returns after a valid close."""

    self._config.after(0, self._config.finish)
    self._config.run()

    self.assertTrue(self._config._stop_event.is_set())
    self.assertFalse(self._config._histogram_process.is_alive())
    self._config.stop()

  def test_callback_error_reaches_run(self) -> None:
    """Tk's callback handler retains its error for the Block caller."""

    def fail() -> None:
      raise ValueError('scheduled callback failed')

    self._config.after(0, fail)
    with patch.object(camera_config_module, 'showerror',
                      side_effect=RuntimeError('dialog failed')):
      with self.assertRaisesRegex(ValueError, 'scheduled callback failed'):
        self._config.run()

    self.assertTrue(self._config._stop_event.is_set())
    self.assertFalse(self._config._histogram_process.is_alive())

  def test_callback_error_before_run_is_preserved(self) -> None:
    """An error swallowed by an early Tk update is raised by run()."""

    try:
      raise ValueError('early callback failed')
    except ValueError as error:
      self._config.report_callback_exception(type(error), error,
                                             error.__traceback__)

    with self.assertRaisesRegex(ValueError, 'early callback failed'):
      self._config.run()

  def test_start_failure_cleans_unstarted_process(self) -> None:
    """A failed histogram start still closes its queues and Tk window."""

    with patch.object(self._config._histogram_process, 'start',
                      side_effect=RuntimeError('cannot start')):
      with self.assertRaisesRegex(RuntimeError, 'cannot start'):
        self._config.run()

    self.assertTrue(self._config._stop_event.is_set())
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)
    with self.assertRaises(tk.TclError):
      self._config.wm_state()

  def test_constructor_failure_cleans_resources(self) -> None:
    """Failure during layout creation releases process resources."""

    class BrokenConfig(CameraConfig):
      instance = None

      def _set_layout(self) -> None:
        type(self).instance = self
        self.update_idletasks()
        raise ValueError('layout failed')

    with self.assertRaisesRegex(ValueError, 'layout failed'):
      BrokenConfig(self._camera, self._log_queue,
                   self._log_level, self._freq, None)

    broken = BrokenConfig.instance
    self.assertTrue(broken._stop_event.is_set())
    self.assertTrue(broken._img_in._closed)
    self.assertTrue(broken._img_out._closed)
    self.assertFalse(broken._histogram_process.is_alive())
    with self.assertRaises(tk.TclError):
      broken.wm_state()

  def test_histogram_constructor_failure_closes_created_queues(self) -> None:
    """Early construction failure also closes queues before Tk is abandoned."""

    queues = []

    def make_queue(*args, **kwargs):
      queue = Queue(*args, **kwargs)
      queues.append(queue)
      return queue

    with (patch.object(camera_config_module, 'Queue', side_effect=make_queue),
          patch.object(camera_config_module, 'HistogramProcess',
                       side_effect=RuntimeError('histogram failed'))):
      with self.assertRaisesRegex(RuntimeError, 'histogram failed'):
        CameraConfig(self._camera, self._log_queue,
                     self._log_level, self._freq, None)

    self.assertEqual(len(queues), 2)
    self.assertTrue(all(queue._closed for queue in queues))
