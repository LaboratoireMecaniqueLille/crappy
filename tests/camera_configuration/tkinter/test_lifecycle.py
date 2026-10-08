# coding: utf-8

"""Tk window lifecycle, cleanup, cancellation, and callback failures."""

from multiprocessing import Event, Queue
from unittest.mock import patch

from ._fixtures import TkinterConfigTestCase, tk
from crappy.tool.camera_config.tkinter import TkinterCameraConfig
import crappy.tool.camera_config.tkinter.camera_config as camera_config_module


class TestFinish(TkinterConfigTestCase):
  """Class for testing the exit behavior of the configuration window.

  .. versionadded:: 2.0.8
  """

  start_histogram_process = True

  def test_exit(self) -> None:
    """Tests whether the configuration window exits as expected when closed."""

    # The stop event should not be set
    self.assertFalse(self._config._stop_event.is_set())

    # The histogram process should still be alive
    self.assertTrue(self._config._histogram_process.is_alive())

    # Destroying the main window
    self._config.finish()

    # The stop event should be set
    self.assertTrue(self._config._stop_event.is_set())

    # This call should raise an error as the window shouldn't exist anymore
    with self.assertRaises(tk.TclError):
      self._config.wm_state()

    # The histogram process should have been killed
    self.assertTrue(self._config._lifecycle._process_closed)


class TestConfigurationLifecycle(TkinterConfigTestCase):

  def test_run_closes_normally(self) -> None:
    """run() starts acquisition, waits, and returns after a valid close."""

    self._config.after(0, self._config.finish)
    self._config.run()

    self.assertTrue(self._config._stop_event.is_set())
    self.assertTrue(self._config._lifecycle._process_closed)
    self._config.stop()

  def test_shutdown_closes_without_validating_selection(self) -> None:
    """A Block stop destroys the window and releases its resources."""

    stop_event = Event()
    self._config.watch_shutdown(stop_event.is_set)
    self._config.after(0, stop_event.set)

    with (patch.object(self._config, '_validate_close') as validate,
          patch.object(self._config, '_on_valid_close') as finalize):
      self._config.run()

    validate.assert_not_called()
    finalize.assert_not_called()
    self.assertTrue(self._config._stop_event.is_set())
    self.assertTrue(self._config._lifecycle._process_closed)
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)
    with self.assertRaises(tk.TclError):
      self._config.wm_state()

  def test_shutdown_before_run_does_not_start_histogram(self) -> None:
    """An already stopped Block never starts configuration resources."""

    stop_event = Event()
    stop_event.set()
    self._config.watch_shutdown(stop_event.is_set)

    with patch.object(self._config._histogram_process, 'start') as start:
      self._config.run()

    start.assert_not_called()
    self.assertTrue(self._config._stop_event.is_set())
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)

  def test_user_close_after_shutdown_skips_selection_validation(self) -> None:
    """A pending shutdown takes precedence over the normal close checks."""

    stop_event = Event()
    stop_event.set()
    self._config.watch_shutdown(stop_event.is_set)

    with patch.object(self._config, '_validate_close') as validate:
      self._config.finish()

    validate.assert_not_called()
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)

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
    self.assertTrue(self._config._lifecycle._process_closed)

  def test_callback_error_before_run_is_preserved(self) -> None:
    """An early Tk error is logged with its traceback and raised by run()."""

    try:
      raise ValueError('early callback failed')
    except ValueError as error:
      logger = self._config._logger
      assert logger is not None
      with patch.object(logger, 'exception') as exception_log:
        self._config.report_callback_exception(type(error), error,
                                               error.__traceback__)

      exception_log.assert_called_once()
      self.assertEqual(exception_log.call_args.args[0],
                       'Configuration callback failed')
      self.assertIs(exception_log.call_args.kwargs['exc_info'][1], error)

    with self.assertRaisesRegex(ValueError, 'early callback failed'):
      self._config.run()

  def test_callback_keyboard_interrupt_closes_silently(self) -> None:
    """Tk callback cancellation closes resources without a dialog or error."""

    interrupt = KeyboardInterrupt()

    def cancel() -> None:
      raise interrupt

    self._config.after(0, cancel)
    with (patch.object(camera_config_module, 'showerror') as dialog,
          patch.object(self._config._lifecycle, '_log') as log):
      with self.assertRaises(KeyboardInterrupt) as raised:
        self._config.run()

    self.assertIs(raised.exception, interrupt)
    dialog.assert_not_called()
    log.assert_not_called()
    self.assertTrue(self._config._window_closed)
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)
    self.assertTrue(self._config._lifecycle._process_closed)

  def test_wait_keyboard_interrupt_closes_silently(self) -> None:
    """An interrupt outside a Tk callback also cleans up before propagating."""

    with (patch.object(self._config, 'wait_window',
                       side_effect=KeyboardInterrupt),
          patch.object(camera_config_module, 'showerror') as dialog,
          patch.object(self._config._lifecycle, '_log') as log):
      with self.assertRaises(KeyboardInterrupt):
        self._config.run()

    dialog.assert_not_called()
    log.assert_not_called()
    self.assertTrue(self._config._window_closed)
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)
    self.assertTrue(self._config._lifecycle._process_closed)

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

  def test_destroy_failure_uses_exception_logging(self) -> None:
    """A Tk destruction error is recorded without skipping resource cleanup."""

    logger = self._config._logger
    assert logger is not None
    destroy = self._config.destroy

    def destroy_then_fail() -> None:
      """Destroy the real window, then simulate a reported Tk failure."""

      destroy()
      raise tk.TclError('destroy failed')

    with (patch.object(self._config, 'destroy',
                       side_effect=destroy_then_fail),
          patch.object(logger, 'exception') as exception_log):
      self._config.stop()

    exception_log.assert_called_once()
    self.assertEqual(exception_log.call_args.args[0],
                     'Cannot destroy the configuration window')
    self.assertIsInstance(exception_log.call_args.kwargs['exc_info'][1],
                          tk.TclError)
    self.assertTrue(self._config._stop_event.is_set())
    self.assertTrue(self._config._img_in._closed)
    self.assertTrue(self._config._img_out._closed)

  def test_constructor_failure_cleans_resources(self) -> None:
    """Failure during layout creation releases process resources."""

    class BrokenConfig(TkinterCameraConfig):
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
    self.assertTrue(broken._lifecycle._process_closed)
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
        TkinterCameraConfig(self._camera, self._log_queue,
                     self._log_level, self._freq, None)

    self.assertEqual(len(queues), 2)
    self.assertTrue(all(queue._closed for queue in queues))
