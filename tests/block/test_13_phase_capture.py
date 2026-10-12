# coding: utf-8

"""Automatic Block capture, phase tracking, and independent finalization."""

import logging
from multiprocessing import Barrier, Event, get_all_start_methods
from queue import Queue
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep
from unittest.mock import Mock, patch

from crappy import Block
from crappy._exceptions import (CameraConfigError, CameraPrepareError,
                                CameraRuntimeError, GeneratorStop, LinkDataError,
                                PrepareError, ReaderStop, SchedulerStop,
                                StartTimeout, StopRequested, T0NotSetError)
from crappy._failure import FailureManager

from .block_test_base import BlockTestBase, TestBlock


class TestBlockHookFailure(TestBlock):
  """Raises from a selected hook, with an optional independent finish error."""

  def __init__(self, phase: str = 'main', error: BaseException | None = None,
               finish_error: BaseException | None = None,
               later_loop: bool = False, block_finish: bool = False) -> None:
    super().__init__(stop=False)
    self.failure_phase = phase
    self.error = error if error is not None else ValueError('hook failure')
    self.finish_error = finish_error
    self.later_loop = later_loop
    self.block_finish = block_finish
    self.finish_entered = Event()
    self.release_finish = Event()

  def prepare(self) -> None:
    super().prepare()
    if self.failure_phase == 'prepare':
      raise self.error

  def begin(self) -> None:
    super().begin()
    if self.failure_phase == 'begin':
      raise self.error

  def main(self) -> None:
    if self.failure_phase == 'custom main':
      raise self.error
    super().main()

  def loop(self) -> None:
    super().loop()
    if self.failure_phase == 'main':
      if not self.later_loop or self.loops.value == 2:
        raise self.error
    else:
      self.stop()

  def finish(self) -> None:
    super().finish()
    self.finish_entered.set()
    if self.block_finish and not self.release_finish.wait(10):
      raise RuntimeError('test did not release finish')
    if self.failure_phase == 'finish':
      raise self.error
    if self.finish_error is not None:
      raise self.finish_error


def _exercise_phase_context(method: str) -> None:
  """Checks source capture and main-entry bytes in real child Processes."""

  from multiprocessing import set_start_method

  set_start_method(method)
  for phase in ('prepare', 'begin', 'main', 'custom main', 'finish'):
    block = TestBlockHookFailure(phase, finish_error=OSError('finish failure'))
    Block.prepare_all(log_level=logging.CRITICAL)
    manager, entered = Block.run_failure_manager, block._main_entered
    assert manager is not None and entered is not None
    assert not hasattr(entered, 'get_lock')
    Block.launch_all(no_raise=True)
    assert block.exitcode == 0, block.exitcode
    assert block.finished.is_set()
    assert entered.value == (phase not in ('prepare', 'begin'))
    expected_phase = 'main' if phase == 'custom main' else phase
    expected = [expected_phase] if phase == 'finish' else [expected_phase, 'finish']
    assert [report.phase for report, _ in manager.snapshot()] == expected
    assert block._main_entered is None
    assert all(error is None for _, error in manager.snapshot())
    try:
      manager.raise_if_failed('remote hook failures')
    except ExceptionGroup as errors:
      assert len(errors.exceptions) == len(expected)
      assert isinstance(errors.exceptions[0], ValueError)
      if phase != 'finish':
        assert isinstance(errors.exceptions[1], OSError)
    else:
      raise AssertionError('hook failures were not retained')


class TestPhaseCapture(BlockTestBase):
  def setup_block(self, block: TestBlock | None = None) -> TestBlock:
    """Supplies the resources normally assigned by prepare_all()."""

    self._block = block if block is not None else TestBlock()
    block = self._block
    block._ready_barrier = Barrier(1)
    block._start_event, block._stop_event = Event(), Event()
    block._pause_event, block._raise_event, block._kbi_event = (
        Event(), Event(), Event())
    block._start_event.set()
    block._logger = Mock(spec=logging.Logger)
    self.incoming = Queue()
    self.queue = Mock(spec_set=('put', 'close', 'join_thread'))
    self.queue.put.side_effect = self.incoming.put
    block._failure_queue = self.queue
    return block

  def run_block(self, block: TestBlock) -> FailureManager:
    """Uses an inspectable source logger without changing process globals."""

    with patch.object(block, '_set_block_logger'):
      block.run()
    self.assertTrue(block.finished.is_set())
    self.queue.close.assert_called_once_with()
    self.queue.join_thread.assert_called_once_with()
    assert block._failure_manager is not None
    return block._failure_manager

  def test_each_hook_has_its_origin_phase(self):
    for phase in ('prepare', 'begin', 'main', 'custom main', 'finish'):
      with self.subTest(phase=phase):
        error = ValueError(phase)
        block = self.setup_block(TestBlockHookFailure(phase, error))
        manager = self.run_block(block)
        report, actual = manager.snapshot()[0]
        self.assertIs(actual, error)
        self.assertEqual(report.phase, 'main' if phase == 'custom main' else phase)
        self.assertIn('raise self.error', report.formatted_traceback)
        self.assertEqual(len(manager.snapshot()), 1)
        self.assertEqual(block._main_entered.value,
                         phase not in ('prepare', 'begin'))
        block._logger.error.assert_called_once_with(str(report))

  def test_later_loop_failure_is_still_main(self):
    block = self.setup_block(TestBlockHookFailure(later_loop=True))
    manager = self.run_block(block)
    self.assertEqual(block.loops.value, 2)
    self.assertEqual(manager.snapshot()[0][0].phase, 'main')
    self.assertEqual(block._main_entered.value, 1)

  def test_bootstrap_logger_failure_still_sends_and_finishes(self):
    block = self.setup_block()
    error = OSError('logger setup')
    with patch.object(block, '_set_block_logger', side_effect=error):
      block.run()
    self.assertEqual(block._failure_manager.snapshot()[0][0].phase, 'bootstrap')
    self.assertIs(block._failure_manager.snapshot()[0][1], error)
    self.assertTrue(block.finished.is_set())
    self.assertFalse(block.prepared.is_set())
    self.assertTrue(block._ready_barrier.broken)
    self.assertEqual(self.queue.put.call_count, 1)
    self.queue.close.assert_called_once_with()

  def test_required_resources_are_real_phase_failures(self):
    cases = (('_ready_barrier', 'synchronize'), ('_stop_event', 'synchronize'),
             ('_start_event', 'start'), ('_pause_event', 'main'),
             ('_main_entered', 'main'))
    for name, phase in cases:
      with self.subTest(resource=name):
        block = self.setup_block()
        setattr(block, name, None)
        manager = self.run_block(block)
        self.assertEqual(len(manager.snapshot()), 1)
        report, error = manager.snapshot()[0]
        self.assertEqual(report.phase, phase)
        self.assertIsInstance(error, RuntimeError)

  def test_runtime_and_cleanup_failures_are_independent(self):
    origin, stop, finish, close = (ValueError('prepare'), OSError('stop'),
                                  KeyboardInterrupt('finish'), OSError('pipe'))
    block = self.setup_block(TestBlockHookFailure('prepare', origin, finish))
    block._stop_event = Mock()
    block._stop_event.set.side_effect = stop
    connection = Mock()
    connection.close.side_effect = close

    def add_connection():
      block._config_connections_to_close.append(connection)
      raise origin

    with patch.object(block, 'prepare', side_effect=add_connection):
      manager = self.run_block(block)
    self.assertEqual([error for _, error in manager.snapshot()],
                     [origin, stop, finish, close])
    self.assertEqual([report.phase for report, _ in manager.snapshot()],
                     ['prepare', 'framework cleanup', 'finish',
                      'framework cleanup'])
    self.assertTrue(block._raise_event.is_set())
    self.assertTrue(block._kbi_event.is_set())
    self.assertEqual(block._config_connections_to_close, [connection])
    connection.close.assert_called_once_with()

  def test_inherited_pipe_failures_are_retained_without_wrapper(self):
    block = self.setup_block()
    first, second = Mock(), Mock()
    error = OSError('inherited pipe')
    first.close.side_effect = [error, None]
    block._config_connections_to_close = [first, second]
    manager = self.run_block(block)
    self.assertEqual(manager.snapshot()[0][0].phase, 'bootstrap')
    self.assertIs(manager.snapshot()[0][1], error)
    self.assertEqual(len(manager.snapshot()), 1)
    self.assertFalse(block.prepared.is_set())
    self.assertEqual(first.close.call_count, 2)
    second.close.assert_called_once_with()
    self.assertEqual(block._config_connections_to_close, [])

  def test_cancellation_does_not_hide_group_members_or_causes(self):
    real, interrupt = ValueError('real'), KeyboardInterrupt('interrupt')
    error = BaseExceptionGroup('outer', [StopRequested('cancel'),
              BaseExceptionGroup('nested', [real, interrupt]), GeneratorStop()])
    error.add_note('original note')
    error.__cause__ = OSError('cause')
    block = self.setup_block(TestBlockHookFailure(error=error))
    manager = self.run_block(block)
    report, retained = manager.snapshot()[0]
    self.assertEqual(len(manager.snapshot()), 1)
    self.assertEqual(retained.message, 'outer')
    self.assertEqual(retained.exceptions[0].exceptions, (real, interrupt))
    self.assertIs(retained.__cause__, error.__cause__)
    self.assertIn('original note', retained.__notes__)
    self.assertNotIn('StopRequested', report.formatted_traceback)
    self.assertNotIn('GeneratorStop', report.formatted_traceback)
    self.assertTrue(block._raise_event.is_set())
    self.assertTrue(block._kbi_event.is_set())

  def test_interrupt_cause_is_not_an_independent_interrupt(self):
    error = ValueError('failure')
    error.__cause__ = KeyboardInterrupt('cause only')
    block = self.setup_block(TestBlockHookFailure(error=error))
    self.run_block(block)
    self.assertTrue(block._raise_event.is_set())
    self.assertFalse(block._kbi_event.is_set())

  def test_normal_completion_is_only_consumed_in_main(self):
    for completion in (GeneratorStop, SchedulerStop, ReaderStop):
      for phase in ('main', 'prepare', 'finish'):
        with self.subTest(completion=completion.__name__, phase=phase):
          error = completion()
          block = self.setup_block(TestBlockHookFailure(phase, error))
          manager = self.run_block(block)
          self.assertEqual(len(manager.snapshot()), phase != 'main')
          self.assertEqual(block._raise_event.is_set(), phase != 'main')
          if phase == 'main':
            block._logger.warning.assert_called_once_with(
                'main requested completion')

  def test_main_completion_diagnostic_failure_still_finishes(self):
    failure = OSError('completion diagnostic')
    block = self.setup_block(TestBlockHookFailure(error=GeneratorStop()))
    block._logger.warning.side_effect = failure
    manager = self.run_block(block)
    self.assertEqual(len(manager.snapshot()), 1)
    self.assertEqual(manager.snapshot()[0][0].phase, 'main')
    self.assertIs(manager.snapshot()[0][1], failure)

  def test_prepare_failure_aborts_execution_before_begin_and_main(self):
    block = self.setup_block(TestBlockHookFailure('prepare'))
    with (patch.object(block, 'begin') as begin,
          patch.object(block, 'main') as main):
      manager = self.run_block(block)
    begin.assert_not_called()
    main.assert_not_called()
    self.assertEqual(manager.snapshot()[0][0].phase, 'prepare')

  def test_barrier_cancellation_and_finish_failure_are_distinct(self):
    error = OSError('finish')
    block = self.setup_block(TestBlockHookFailure('unused', finish_error=error))
    block._ready_barrier.abort()
    manager = self.run_block(block)
    self.assertEqual(len(manager.snapshot()), 1)
    self.assertIs(manager.snapshot()[0][1], error)
    self.assertEqual(manager.snapshot()[0][0].phase, 'finish')
    self.assertFalse(block.begun.is_set())
    self.assertEqual(block._main_entered.value, 0)

  def test_timeout_after_cancellation_is_not_a_failure(self):
    block = self.setup_block()

    def cancel_start(timeout):
      block._stop_event.set()
      return False

    assert block._start_event is not None
    with patch.object(block._start_event, 'wait', side_effect=cancel_start):
      manager = self.run_block(block)
    self.assertEqual(manager.snapshot(), ())
    self.assertFalse(block.begun.is_set())
    block = self.setup_block()
    assert block._start_event is not None
    with patch.object(block._start_event, 'wait', return_value=False):
      manager = self.run_block(block)
    self.assertIsInstance(manager.snapshot()[0][1], StartTimeout)
    self.assertEqual(manager.snapshot()[0][0].phase, 'start')

  def test_a_stop_request_does_not_hide_a_prepare_failure(self):
    error = ValueError('prepare despite stop')
    block = self.setup_block(TestBlockHookFailure('prepare', error))
    block._stop_event.set()
    manager = self.run_block(block)
    self.assertIs(manager.snapshot()[0][1], error)

  def test_stop_before_startup_does_not_enter_main(self):
    block = self.setup_block()
    block._stop_event.set()
    manager = self.run_block(block)
    self.assertEqual(manager.snapshot(), ())
    self.assertFalse(block.begun.is_set())
    self.assertEqual(block._main_entered.value, 0)

  def test_former_special_cases_keep_their_actual_errors(self):
    errors = (LinkDataError, CameraConfigError, CameraPrepareError,
              CameraRuntimeError, PrepareError, T0NotSetError,
              KeyboardInterrupt, SystemExit)
    for error_type in errors:
      with self.subTest(error=error_type.__name__):
        error = error_type('actual diagnostic')
        block = self.setup_block(TestBlockHookFailure(error=error))
        manager = self.run_block(block)
        self.assertIs(manager.snapshot()[0][1], error)
        self.assertEqual(manager.snapshot()[0][0].phase, 'main')

  def test_report_failure_cannot_skip_finish_or_transmission(self):
    origin, logging_error = ValueError('origin'), OSError('logger')
    block = self.setup_block(TestBlockHookFailure(error=origin))
    block._logger.error.side_effect = logging_error
    with patch('crappy._failure.sys.stderr', Mock()):
      manager = self.run_block(block)
    self.assertEqual([error for _, error in manager.snapshot()],
                     [origin, logging_error])
    self.assertEqual(self.queue.put.call_count, 2)
    block._logger.error.assert_called_once()

  def test_explicit_capture_then_raise_is_not_logged_twice(self):
    error = ValueError('already captured')
    block = self.setup_block()

    def capture_and_raise():
      block._failure_manager.capture_exception(error)
      raise error

    with patch.object(block, 'main', side_effect=capture_and_raise):
      manager = self.run_block(block)
    self.assertEqual(len(manager.snapshot()), 1)
    block._logger.error.assert_called_once()
    root = FailureManager('main', 'main')
    root._logger = Mock()
    while root.receive((self.incoming,)):
      pass
    with self.assertRaises(ExceptionGroup):
      root.raise_if_failed('received failure')
    root._logger.error.assert_not_called()

  def test_failure_is_received_before_blocked_finish_returns(self):
    for phase in ('prepare', 'main'):
      with self.subTest(phase=phase):
        block = TestBlockHookFailure(phase, block_finish=True)
        self._block = block
        Block.prepare_all(log_level=logging.CRITICAL)
        manager = Block.run_failure_manager
        observed = []

        def release_after_receipt():
          try:
            deadline = monotonic() + 5
            while not manager.snapshot() and monotonic() < deadline:
              sleep(0.01)
            observed.append(bool(manager.snapshot()) and block.is_alive()
                            and block.finish_entered.wait(3))
          finally:
            block.release_finish.set()

        releaser = Thread(target=release_after_receipt)
        releaser.start()
        try:
          Block.launch_all(no_raise=True)
        finally:
          block.release_finish.set()
          releaser.join(3)
        self.assertEqual(observed, [True])
        self.assertFalse(releaser.is_alive())
        self.assertEqual(block.exitcode, 0)
        self.assertEqual([report.phase for report, _ in manager.snapshot()],
                         [phase])

  def test_real_phase_capture_in_all_start_methods(self):
    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = ('from tests.block.test_13_phase_capture import '
                  '_exercise_phase_context; '
                  f'_exercise_phase_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
