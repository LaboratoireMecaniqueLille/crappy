# coding: utf-8

"""Headless checks for the configurator's process and error lifecycle."""

import unittest
import logging
from multiprocessing import Event
from unittest.mock import patch

from crappy.tool.camera_config.base._configuration_lifecycle import (
  ConfigurationLifecycle, ExceptionInfo)


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
    self.join_calls = 0
    self.terminate_calls = 0
    self.kill_calls = 0

  def is_alive(self) -> bool:
    return self.alive

  def join(self, timeout: float | None = None) -> None:
    self.join_calls += 1

  def terminate(self) -> None:
    self.terminate_calls += 1

  def kill(self) -> None:
    self.kill_calls += 1
    self.alive = False


class TestConfigurationLifecycle(unittest.TestCase):
  def make_lifecycle(self, process: _Process):
    """Create resources and collect lifecycle log messages."""

    stop_event = Event()
    queues = (_Queue(), _Queue())
    messages: list[tuple[int, str, ExceptionInfo | None]] = []

    def record(level: int,
               msg: str,
               *,
               exc_info: ExceptionInfo | None = None) -> None:
      """Keep ordinary and exception records for assertions."""

      messages.append((level, msg, exc_info))

    lifecycle = ConfigurationLifecycle(stop_event, process, queues, record)
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
    for queue in queues:
      self.assertEqual((queue.cancel_calls, queue.close_calls), (1, 1))

  def test_hung_process_is_terminated_then_killed(self) -> None:
    """A process that ignores graceful shutdown reaches the kill stage."""

    process = _Process(alive=True)
    lifecycle, _, _, _ = self.make_lifecycle(process)
    lifecycle.mark_histogram_started()
    lifecycle.close_resources()

    self.assertEqual(process.join_calls, 3)
    self.assertEqual(process.terminate_calls, 1)
    self.assertEqual(process.kill_calls, 1)

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
    with self.assertRaises(ValueError) as raised:
      lifecycle.raise_if_failed()

    self.assertIs(raised.exception, original)
    self.assertEqual(close_calls, [True])
    self.assertEqual(messages[-1][0:2],
                     (logging.ERROR, 'Configuration callback failed'))
    info = messages[-1][2]
    assert info is not None
    self.assertIs(info[1], original)
    self.assertIs(info[2], original_traceback)

  def test_close_failure_keeps_exception_details(self) -> None:
    """A backend close failure is logged with its original traceback."""

    lifecycle, _, _, messages = self.make_lifecycle(_Process())

    def fail_close() -> None:
      """Simulate a GUI backend rejecting closure."""

      raise RuntimeError('close failed')

    lifecycle.request_close(fail_close)

    self.assertEqual(messages[-1][0:2],
                     (logging.ERROR, 'Could not close configuration window'))
    info = messages[-1][2]
    assert info is not None
    self.assertIsInstance(info[1], RuntimeError)
    self.assertIsNotNone(info[2])

  def test_keyboard_interrupt_is_retained_without_error_logging(self) -> None:
    """Cancellation reaches the caller without being reported as a failure."""

    lifecycle, _, _, messages = self.make_lifecycle(_Process())
    interrupt = KeyboardInterrupt()
    lifecycle.record_callback_failure(interrupt, None)

    with self.assertRaises(KeyboardInterrupt) as raised:
      lifecycle.raise_if_failed()

    self.assertIs(raised.exception, interrupt)
    self.assertEqual(messages, [])

  def test_queue_cleanup_failures_keep_exception_details(self) -> None:
    """Resource cleanup reports caught failures as exception records."""

    lifecycle, _, queues, messages = self.make_lifecycle(_Process())
    with (patch.object(queues[0], 'cancel_join_thread',
                       side_effect=RuntimeError('join failed')),
          patch.object(queues[0], 'close',
                       side_effect=OSError('close failed'))):
      lifecycle.close_resources()

    self.assertEqual([message for _, message, _ in messages],
                     ['Could not cancel histogram queue thread join',
                      'Could not close histogram queue'])
    for level, _, info in messages:
      self.assertEqual(level, logging.ERROR)
      assert info is not None
      self.assertIsNotNone(info[2])


if __name__ == '__main__':
  unittest.main()
