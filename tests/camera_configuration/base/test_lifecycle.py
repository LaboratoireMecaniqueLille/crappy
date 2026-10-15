# coding: utf-8

"""Headless checks for the configurator's process and error lifecycle."""

import unittest
import logging
from importlib import import_module
from multiprocessing import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crappy.tool.camera_config.base._configuration_lifecycle import (
  ConfigurationLifecycle, ExceptionInfo)
from crappy.tool.camera_config.pyqt import PyQtCameraConfig
from crappy.tool.camera_config.tkinter import TkinterCameraConfig


from crappy._failure import FailureManager
from crappy._exceptions import ShutdownTimeoutError
from tests.block.block_test_base import exception_leaves


class _Queue:
  def __init__(self) -> None:
    self.cancel_calls = 0
    self.close_calls = 0

  def cancel_join_thread(self) -> None:
    self.cancel_calls += 1

  def close(self) -> None:
    self.close_calls += 1


class _Process:
  def __init__(self, alive: bool = False) -> None:
    self.alive = alive
    self.name = 'histogram'
    self.pid = None
    self.exitcode = None
    self.join_calls = 0
    self.terminate_calls = 0
    self.kill_calls = 0
    self.close_calls = 0

  def is_alive(self) -> bool:
    return self.alive

  def join(self, timeout: float | None = None) -> None:
    self.join_calls += 1

  def terminate(self) -> None:
    self.terminate_calls += 1

  def kill(self) -> None:
    self.kill_calls += 1
    self.alive = False

  def close(self) -> None:
    if self.alive:
      raise ValueError("Cannot close a live process")
    self.close_calls += 1


class TestConfigurationLifecycle(unittest.TestCase):
  def test_requires_owner_manager(self) -> None:
    """The resource owner must supply the same manager as its configurator."""

    with self.assertRaisesRegex(TypeError, 'manager'):
      ConfigurationLifecycle(Event(), _Process(), ())

  def test_constructor_rollback_attempts_all_acquired_resources(self) -> None:
    """Early constructor failure and a cleanup interrupt retain all errors."""

    for backend, config_type in (('tkinter', TkinterCameraConfig),
                                 ('pyqt', PyQtCameraConfig)):
      with self.subTest(backend=backend):
        module = import_module(
            f'crappy.tool.camera_config.{backend}.camera_config')
        window = SimpleNamespace(_get_application=Mock(), destroy=Mock(),
                                  close=Mock(return_value=True),
                                  _log_level=None, _log_queue=Mock(), log=Mock())
        queues, process = (Mock(), Mock(), Mock()), Mock()
        primary, interrupt = ValueError('initialization'), KeyboardInterrupt()
        errors = (OSError('cancel feeder'), RuntimeError('close queue'))
        process.close.side_effect = interrupt
        queues[0].cancel_join_thread.side_effect = errors[0]
        queues[1].close.side_effect = errors[1]
        manager = FailureManager('test configurator', 'configuration')
        manager._logger = Mock()
        with (patch.object(module, 'super', create=True,
                           return_value=SimpleNamespace(__init__=Mock())),
              patch.object(module, 'Queue', side_effect=queues),
              patch.object(module, 'HistogramProcess', return_value=process),
              patch.object(module, 'ConfigurationLifecycle',
                           side_effect=primary)):
          with self.assertRaises(BaseExceptionGroup) as caught:
            config_type.__init__(window, Mock(), Mock(), None, None, None,
                                 failure_manager=manager)

        self.assertEqual(exception_leaves(caught.exception),
                         (primary, interrupt, *errors))
        self.assertIsNone(interrupt.__cause__)
        self.assertEqual(manager._logger.error.call_count, 4)
        process.close.assert_called_once_with()
        for queue in queues:
          queue.cancel_join_thread.assert_called_once_with()
          queue.close.assert_called_once_with()
        if backend == 'tkinter':
          window.destroy.assert_called_once_with()
        else:
          window.close.assert_called_once_with()

  def make_lifecycle(self, process: _Process):
    """Create resources and collect lifecycle log messages."""

    stop_event = Event()
    queues = (_Queue(), _Queue())
    messages = []
    manager = FailureManager('test configuration', 'configuration')
    manager._logger = Mock()
    manager._logger.error.side_effect = messages.append
    lifecycle = ConfigurationLifecycle(stop_event, process, queues, manager)
    return lifecycle, stop_event, queues, messages

  def test_unstarted_process_cleanup_is_idempotent(self) -> None:
    """An unopened window closes queues without joining an unstarted process."""

    process = _Process()
    lifecycle, stop_event, queues, _ = self.make_lifecycle(process)
    lifecycle.close_resources()
    lifecycle.close_resources()

    self.assertTrue(stop_event.is_set())
    self.assertTrue(lifecycle.closed)
    self.assertEqual(process.join_calls, 0)
    self.assertEqual(process.close_calls, 1)
    for queue in queues:
      self.assertEqual((queue.cancel_calls, queue.close_calls), (1, 1))

  def test_hung_process_is_terminated_then_killed(self) -> None:
    """A process that ignores graceful shutdown reaches the kill stage."""

    process = _Process(alive=True)
    lifecycle, _, _, _ = self.make_lifecycle(process)
    lifecycle.mark_histogram_started()
    with self.assertRaises(ExceptionGroup) as caught:
      lifecycle.close_resources()
    self.assertIsInstance(exception_leaves(caught.exception)[0],
                          ShutdownTimeoutError)

    self.assertEqual(process.join_calls, 3)
    self.assertEqual(process.terminate_calls, 1)
    self.assertEqual(process.kill_calls, 1)

  def test_process_failures_still_reach_kill_close_and_all_queues(self) -> None:
    """A failed join or terminate cannot skip escalation or Queue cleanup."""

    process = Mock()
    process.is_alive.return_value = True
    errors = (OSError('join'), RuntimeError('terminate'))
    process.join.side_effect = [errors[0], None, None]
    process.terminate.side_effect = errors[1]
    process.kill.side_effect = (
        lambda: setattr(process.is_alive, 'return_value', False))
    lifecycle, _, queues, _ = self.make_lifecycle(process)
    lifecycle.mark_histogram_started()
    with self.assertRaises(ExceptionGroup) as caught:
      lifecycle.close_resources()
    leaves = exception_leaves(caught.exception)
    self.assertEqual((leaves[0], leaves[2]), errors)
    self.assertIsInstance(leaves[1], ShutdownTimeoutError)
    process.kill.assert_called_once_with()
    process.close.assert_called_once_with()
    self.assertEqual([call.args for call in process.join.call_args_list],
                     [(2.0,), (1.0,), (1.0,)])
    for queue in queues:
      self.assertEqual((queue.cancel_calls, queue.close_calls), (1, 1))
    lifecycle.close_resources()
    process.close.assert_called_once_with()

  def test_interrupt_during_stop_preserves_remaining_failures(self) -> None:
    """Even failed signaling must release all resources before interruption."""

    process = _Process()
    lifecycle, _, queues, _ = self.make_lifecycle(process)
    interrupt, error = KeyboardInterrupt(), OSError('queue close')
    lifecycle._stop_event = Mock()
    lifecycle._stop_event.set.side_effect = interrupt
    with patch.object(queues[0], 'close', side_effect=error):
      with self.assertRaises(BaseExceptionGroup) as caught:
        lifecycle.close_resources()
    self.assertEqual(exception_leaves(caught.exception), (interrupt, error))
    self.assertIsNone(interrupt.__cause__)
    self.assertEqual(process.close_calls, 1)
    self.assertEqual(queues[1].close_calls, 1)

  def test_tk_stop_attempts_every_schedule_window_and_histogram(self) -> None:
    """Tk cleanup can be checked without opening a real display."""

    errors = (OSError('cancel'), RuntimeError('destroy'), ValueError('worker'))
    window = SimpleNamespace(
        _window_closed=False, _window_destroyed=False,
        _img_acq_sched_obj='acquisition', _upd_var_sched_obj='indicator',
        _shutdown_sched_obj='shutdown', after_cancel=Mock(),
        destroy=Mock(), log=Mock(), _lifecycle=Mock())
    window._failure_manager = FailureManager('test Tk window', 'finish')
    window._failure_manager._logger = Mock()
    window._running = False
    window.after_cancel.side_effect = [errors[0], None, None]
    window.destroy.side_effect = errors[1]
    window._lifecycle.close_resources.side_effect = errors[2]
    with self.assertRaises(ExceptionGroup) as caught:
      TkinterCameraConfig.stop(window)
    self.assertEqual(caught.exception.exceptions, errors)
    self.assertEqual(window.after_cancel.call_count, 3)
    window._lifecycle.close_resources.assert_called_once_with()
    window.after_cancel.side_effect = None
    window.destroy.side_effect = None
    window._lifecycle.close_resources.side_effect = None
    TkinterCameraConfig.stop(window)
    self.assertEqual(window.after_cancel.call_count, 4)
    self.assertTrue(window._window_destroyed)

  def test_qt_stop_attempts_all_timers_window_event_loop_and_histogram(
      self) -> None:
    """Qt timer and event-loop failures cannot strand histogram resources."""

    timers = [Mock(), Mock(), Mock()]
    errors = (OSError('first timer'), ValueError('last timer'),
              RuntimeError('window'), OSError('event loop'),
              ValueError('worker'))
    window = SimpleNamespace(
        _window_closed=False, _window_destroyed=False, _stopped_timers=[],
        _acquisition_timer=timers[0], _indicator_timer=timers[1],
        _shutdown_timer=timers[2], close=Mock(),
        _event_loop=Mock(), _lifecycle=Mock())
    window._failure_manager = FailureManager('test Qt window', 'finish')
    window._failure_manager._logger = Mock()
    window._running = False
    timers[0].stop.side_effect = errors[0]
    timers[2].stop.side_effect = errors[1]
    window.close.side_effect = errors[2]
    event_loop = window._event_loop
    event_loop.quit.side_effect = errors[3]
    window._lifecycle.close_resources.side_effect = errors[4]
    with self.assertRaises(ExceptionGroup) as caught:
      PyQtCameraConfig.stop(window)
    self.assertEqual(caught.exception.exceptions, errors)
    window._lifecycle.close_resources.assert_called_once_with()
    timers[0].stop.side_effect = timers[2].stop.side_effect = None
    window.close.side_effect = event_loop.quit.side_effect = None
    window._lifecycle.close_resources.side_effect = None
    PyQtCameraConfig.stop(window)
    timers[1].stop.assert_called_once_with()
    self.assertTrue(window._window_destroyed)
    self.assertIsNone(window._event_loop)

  def test_callback_error_is_retained_and_close_is_requested(self) -> None:
    """An event-loop callback error is exposed with its original traceback."""

    lifecycle, _, _, messages = self.make_lifecycle(_Process())
    try:
      raise ValueError('bad callback')
    except ValueError as error:
      original = error
      original_traceback = error.__traceback__
      lifecycle.record_callback_failure(error, original_traceback)

    close_calls = []
    lifecycle.request_close(lambda: close_calls.append(True))
    with self.assertRaises(ExceptionGroup) as raised:
      lifecycle.raise_if_failed()

    self.assertEqual(exception_leaves(raised.exception), (original,))
    self.assertEqual(close_calls, [True])
    self.assertEqual(len(messages), 1)
    self.assertIn('bad callback', messages[0])
    self.assertIs(lifecycle.manager.snapshot()[0][1], original)
    self.assertIs(original.__traceback__, original_traceback)

  def test_close_failure_keeps_exception_details(self) -> None:
    """A backend close failure is logged with its original traceback."""

    lifecycle, _, _, messages = self.make_lifecycle(_Process())

    def fail_close() -> None:
      """Simulate a GUI backend rejecting closure."""

      raise RuntimeError('close failed')

    lifecycle.request_close(fail_close)

    self.assertEqual(len(messages), 1)
    error = lifecycle.manager.snapshot()[0][1]
    self.assertIsInstance(error, RuntimeError)
    self.assertIsNotNone(error.__traceback__)

  def test_keyboard_interrupt_is_retained_as_a_group_member(self) -> None:
    """An interrupt is retained and reported without changing its cause."""

    lifecycle, _, _, messages = self.make_lifecycle(_Process())
    interrupt = KeyboardInterrupt()
    lifecycle.record_callback_failure(interrupt, None)

    with self.assertRaises(BaseExceptionGroup) as raised:
      lifecycle.raise_if_failed()

    self.assertEqual(exception_leaves(raised.exception), (interrupt,))
    self.assertEqual(len(messages), 1)
    self.assertIsNone(interrupt.__cause__)

  def test_queue_cleanup_failures_keep_exception_details(self) -> None:
    """Resource cleanup reports caught failures as exception records."""

    lifecycle, _, queues, _ = self.make_lifecycle(_Process())
    errors = (RuntimeError('join failed'), OSError('close failed'))
    with (patch.object(queues[0], 'cancel_join_thread',
                       side_effect=errors[0]),
          patch.object(queues[0], 'close',
                       side_effect=errors[1])):
      with self.assertRaises(ExceptionGroup) as caught:
        lifecycle.close_resources()

    self.assertEqual(caught.exception.exceptions, errors)
    self.assertEqual(queues[1].close_calls, 1)
    for error in errors:
      self.assertTrue(error.__notes__[0].startswith('Configuration cleanup'))
      self.assertIsNotNone(error.__traceback__)
    lifecycle.close_resources()
    lifecycle.close_resources()
    self.assertEqual(queues[1].close_calls, 1)


  def test_multiple_callbacks_and_close_failures_are_retained_once(self):
    lifecycle, _, _, messages = self.make_lifecycle(_Process())
    cause = KeyError('original callback cause')
    first, second, cleanup = (ValueError('first callback'),
                               KeyboardInterrupt('second callback'),
                               OSError('window close'))
    first.__cause__ = cause
    grouped = BaseExceptionGroup('callbacks', [first, second])
    lifecycle.record_callback_failure(grouped, None)
    lifecycle.request_close(Mock(side_effect=cleanup))
    lifecycle.record_callback_failure(grouped, None)
    with self.assertRaises(BaseExceptionGroup) as caught:
      lifecycle.raise_if_failed()
    self.assertEqual(exception_leaves(caught.exception), (first, second, cleanup))
    self.assertIs(first.__cause__, cause)
    self.assertIsNone(second.__cause__)
    self.assertEqual(len(messages), 2)

  def test_cancel_request_is_not_a_callback_incident(self):
    from crappy._exceptions import StopRequested
    lifecycle, _, _, messages = self.make_lifecycle(_Process())
    lifecycle.record_callback_failure(StopRequested(), None)
    lifecycle.raise_if_failed()
    self.assertEqual(lifecycle.manager.snapshot(), ())
    self.assertEqual(messages, [])


if __name__ == '__main__':
  unittest.main()
