# coding: utf-8

"""FailureManager's inline collection contexts, retries, and isolation."""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from threading import Barrier
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch

from crappy._exceptions import StopRequested
from crappy._failure import FailureManager


class TestFailureManagerCollections(TestCase):
  def setUp(self) -> None:
    """Use one manager across cleanup attempts, with observable logging."""

    self.manager = FailureManager('owner', 'finish')
    self.emitted = []
    logger = patch.object(self.manager, '_logger')
    self.logger = logger.start()
    self.logger.error.side_effect = self.emitted.append
    self.addCleanup(logger.stop)

  def test_all_independent_steps_run_before_grouped_raise(self) -> None:
    first, last = OSError('first'), ValueError('last')
    interrupt = KeyboardInterrupt()
    cause = RuntimeError('interrupt cause')
    interrupt.__cause__ = cause
    operations = [Mock(side_effect=first), Mock(side_effect=interrupt),
                  Mock(side_effect=last), Mock()]
    with self.assertRaises(BaseExceptionGroup) as caught:
      with self.manager.collect('owner.finish'):
        for index, operation in enumerate(operations):
          with self.manager.step(f'step {index}'):
            operation()
          self.assertEqual(len(self.emitted), min(index + 1, 3))
    self.assertEqual(caught.exception.exceptions, (first, interrupt, last))
    self.assertIs(interrupt.__cause__, cause)
    for index, error in enumerate((first, interrupt, last)):
      self.assertIn(f'owner.finish step: step {index}', error.__notes__)
    for operation in operations:
      operation.assert_called_once_with()

  def test_singleton_and_success_contract(self) -> None:
    with self.manager.collect('success'):
      with self.manager.step('close'):
        Mock()()
    self.manager.raise_if_failed('run')
    error = OSError('one')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('custom label'):
        with self.manager.step('failure'):
          Mock(side_effect=error)()
    self.assertEqual(caught.exception.message, 'custom label failures')
    self.assertEqual(caught.exception.exceptions, (error,))

  def test_cancellation_is_not_success_or_a_retained_failure(self) -> None:
    succeeded = False
    with self.manager.collect('cancelled close'):
      with self.manager.step('cancel'):
        Mock(side_effect=StopRequested())()
        succeeded = True
    self.assertFalse(succeeded)
    self.assertEqual(self.manager.snapshot(), ())
    self.assertEqual(self.emitted, [])
    error = OSError('real')
    with self.assertRaises(BaseExceptionGroup) as caught:
      with self.manager.collect('mixed close'):
        self.manager._record_context_exception(
            BaseExceptionGroup('mixed', [StopRequested(),
                                         KeyboardInterrupt(), error]),
            'mixed operation')
    group = caught.exception.exceptions[0]
    assert isinstance(group, BaseExceptionGroup)
    self.assertIsInstance(group.exceptions[0], KeyboardInterrupt)
    self.assertIs(group.exceptions[1], error)

  def test_reported_error_can_be_raised_by_local_collection(self) -> None:
    error = ValueError('origin')
    self.manager.capture_exception(error)
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('rollback'):
        self.manager._record_context_exception(error, 'initialization')
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(len(self.emitted), 1)
    self.assertEqual(len(self.manager.snapshot()), 1)

  def test_prompt_reporting_without_summary_reemission(self) -> None:
    error = OSError('close')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('close resource'):
          Mock(side_effect=error)()
        self.assertEqual(len(self.emitted), 1)
        report = self.manager.snapshot()[0][0]
        self.assertEqual(report.phase, 'finish')
        self.assertIn('close resource', report.formatted_traceback)
        self.assertEqual(self.emitted, [str(report)])
        with self.manager.step('later resource'):
          self.assertEqual(len(self.emitted), 1)
    self.assertEqual(caught.exception.exceptions, (error,))
    with self.assertRaises(ExceptionGroup):
      self.manager.raise_if_failed('run')
    self.assertEqual(len(self.emitted), 1)

  def test_dispatch_failure_does_not_shadow_or_skip_cleanup(self) -> None:
    origin, sink_error = ValueError('origin'), KeyboardInterrupt('sink')
    self.logger.error.side_effect = sink_error
    later = Mock()
    with patch('crappy._failure.sys.stderr', StringIO()):
      with self.assertRaises(BaseExceptionGroup) as caught:
        with self.manager.collect('finish'):
          with self.manager.step('close'):
            Mock(side_effect=origin)()
          with self.manager.step('later'):
            later()
    self.assertEqual(caught.exception.exceptions, (origin, sink_error))
    self.assertTrue(any('report failure from close' in note
                        for note in sink_error.__notes__))
    self.assertEqual(tuple(error for _, error in self.manager.snapshot()),
                     (origin, sink_error))
    self.logger.error.assert_called_once()
    later.assert_called_once_with()

  def test_collection_prunes_only_its_known_members(self) -> None:
    old, new = ValueError('old'), ValueError('new')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        self.manager._record_context_exception(old, 'first')
        self.manager._record_context_exception(
            ExceptionGroup('later', [old, new]), 'later')
        self.manager._record_context_exception(old, 'again')
    self.assertIs(caught.exception.exceptions[0], old)
    later = caught.exception.exceptions[1]
    assert isinstance(later, BaseExceptionGroup)
    self.assertEqual(later.exceptions, (new,))
    self.assertEqual(len(caught.exception.exceptions), 2)
    self.assertEqual(len(self.emitted), 2)

  def test_release_retry_succeeds_without_erasing_run_history(self) -> None:
    resources = {'first': Mock(), 'second': Mock()}
    handles = dict(resources)
    error = OSError('first close')
    resources['first'].close.side_effect = [error, None]

    def close_resources():
      """Retry only handles that were not released successfully."""

      with self.manager.collect('close resources'):
        for name, handle in tuple(handles.items()):
          with self.manager.step(f'close {name}'):
            handle.close()
            del handles[name]

    with self.assertRaises(ExceptionGroup) as caught:
      close_resources()
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(handles, {'first': resources['first']})
    close_resources()
    close_resources()
    self.assertEqual(handles, {})
    self.assertEqual(resources['first'].close.call_count, 2)
    resources['second'].close.assert_called_once_with()
    with self.assertRaises(ExceptionGroup) as caught:
      self.manager.raise_if_failed('run')
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(len(self.emitted), 1)

  def test_constructor_and_rollback_failure_members(self) -> None:
    origin, rollback = ValueError('initialize'), OSError('rollback')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('constructor'):
        self.manager._record_context_exception(origin, 'initialization')
        with self.manager.step('rollback'):
          Mock(side_effect=rollback)()
    self.assertEqual(caught.exception.exceptions, (origin, rollback))

  def test_system_exit_does_not_skip_cleanup(self) -> None:
    exit_error = SystemExit(0)
    later = Mock()
    with self.assertRaises(BaseExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('exit'):
          Mock(side_effect=exit_error)()
        with self.manager.step('later'):
          later()
    self.assertEqual(caught.exception.exceptions, (exit_error,))
    later.assert_called_once_with()

  def test_specific_recovery_inside_generic_context(self) -> None:
    handle = Mock()
    handle.unlink.side_effect = FileNotFoundError('already unlinked')
    recovered = False
    with self.manager.collect('unlink resource'):
      with self.manager.step('unlink'):
        try:
          handle.unlink()
        except FileNotFoundError:
          recovered = True
    self.assertTrue(recovered)
    self.assertEqual(self.manager.snapshot(), ())
    self.assertEqual(self.emitted, [])

    error = PermissionError('unexpected permission')
    handle.unlink.side_effect = error
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('unlink another resource'):
        with self.manager.step('unlink'):
          try:
            handle.unlink()
          except FileNotFoundError:
            self.fail('wrong exception handler')
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(len(self.emitted), 1)

  def test_explicit_handler_uses_the_same_recording_path(self) -> None:
    error = OSError('close failed')
    operation = Mock(side_effect=error)
    closed = False
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        try:
          operation()
        except OSError as error_caught:
          self.manager._record_context_exception(error_caught,
                                                 'close resource')
        else:
          closed = True
    self.assertFalse(closed)
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(len(self.emitted), 1)

  def test_record_then_reraise_inside_context_is_not_duplicated(self) -> None:
    error = OSError('close failed')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('close resource'):
          try:
            Mock(side_effect=error)()
          except OSError as error_caught:
            self.manager._record_context_exception(error_caught,
                                                   'specific close handling')
            raise
    self.assertEqual(caught.exception.exceptions, (error,))
    with self.assertRaises(ExceptionGroup):
      self.manager.raise_if_failed('run')
    self.assertEqual(len(self.emitted), 1)

  def test_nested_groups_interrupts_and_causes_are_preserved(self) -> None:
    error, interrupt = OSError('close'), KeyboardInterrupt('interrupt')
    cause = ValueError('cause')
    nested = BaseExceptionGroup('nested', [interrupt, error])
    nested.__cause__ = cause
    with self.assertRaises(BaseExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('grouped close'):
          Mock(side_effect=nested)()
    self.assertIs(caught.exception.exceptions[0], nested)
    self.assertIs(nested.__cause__, cause)
    self.assertEqual(len(self.emitted), 1)

  def test_failure_skips_success_update_but_not_next_operation(self) -> None:
    closed = False
    handle = Mock()
    handle.close.side_effect = OSError('close')
    later = Mock()
    with self.assertRaises(ExceptionGroup):
      with self.manager.collect('finish'):
        with self.manager.step('close resource'):
          handle.close()
          closed = True
        with self.manager.step('later resource'):
          later()
    self.assertFalse(closed)
    later.assert_called_once_with()

  def test_all_reporting_sinks_failing_does_not_skip_cleanup(self) -> None:
    origin = ValueError('origin')
    sink_error, emergency_error = OSError('logger'), RuntimeError('stderr')
    self.logger.error.side_effect = sink_error
    emergency = Mock()
    emergency.write.side_effect = emergency_error
    later = Mock()
    with patch('crappy._failure.sys.stderr', emergency):
      with self.assertRaises(ExceptionGroup) as caught:
        with self.manager.collect('finish'):
          with self.manager.step('close'):
            Mock(side_effect=origin)()
          with self.manager.step('later'):
            later()
    self.assertIs(caught.exception.exceptions[0], origin)
    reporting_failures = caught.exception.exceptions[1]
    assert isinstance(reporting_failures, BaseExceptionGroup)
    self.assertEqual(reporting_failures.exceptions,
                     (sink_error, emergency_error))
    later.assert_called_once_with()
    self.logger.error.assert_called_once()

  def test_stack_is_available_before_and_after_nested_collect(self) -> None:
    stack = self.manager._collections.stack
    self.assertEqual(stack, [])
    with self.manager.collect('outer'):
      self.assertIs(self.manager._collections.stack, stack)
      self.assertEqual(len(stack), 1)
      with self.manager.collect('inner'):
        self.assertIs(self.manager._collections.stack, stack)
        self.assertEqual(len(stack), 2)
      self.assertEqual(len(stack), 1)
    self.assertIs(self.manager._collections.stack, stack)
    self.assertEqual(stack, [])
    self.assertEqual(self.emitted, [])

  def test_new_thread_starts_outside_the_main_collection(self) -> None:
    main_stack = self.manager._collections.stack
    operation = Mock()

    def check_thread():
      """Access fresh thread-local state before entering any collection."""

      stack = self.manager._collections.stack
      self.assertEqual(stack, [])
      self.assertIsNot(stack, main_stack)
      with self.assertRaisesRegex(RuntimeError, 'inside collect'):
        with self.manager.step('outside collection'):
          operation()
      operation.assert_not_called()
      with self.assertRaisesRegex(RuntimeError, 'inside collect'):
        self.manager._record_context_exception(OSError('outside collection'),
                                               'close')
      with self.manager.collect('worker'):
        self.assertIs(self.manager._collections.stack, stack)
        self.assertEqual(len(stack), 1)
        with self.manager.step('close'):
          operation()
      self.assertIs(self.manager._collections.stack, stack)
      self.assertEqual(stack, [])

    with self.manager.collect('main'):
      with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(check_thread).result(timeout=5)
      self.assertEqual(len(main_stack), 1)
    self.assertEqual(main_stack, [])
    operation.assert_called_once_with()
    self.assertEqual(self.manager.snapshot(), ())
    self.assertEqual(self.emitted, [])

  def test_nested_collections_restore_the_outer_collection(self) -> None:
    first, inner = ValueError('first'), OSError('inner')
    last = ValueError('last')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('outer'):
        with self.manager.step('first'):
          Mock(side_effect=first)()
        with self.manager.step('inner cleanup'):
          with self.manager.collect('inner'):
            with self.manager.step('close'):
              Mock(side_effect=inner)()
        with self.manager.step('last'):
          Mock(side_effect=last)()
    group = caught.exception
    self.assertIs(group.exceptions[0], first)
    inner_group = group.exceptions[1]
    assert isinstance(inner_group, ExceptionGroup)
    self.assertEqual(inner_group.message, 'inner failures')
    self.assertEqual(inner_group.exceptions, (inner,))
    self.assertIs(group.exceptions[2], last)
    self.assertEqual(len(self.emitted), 3)

  def test_simultaneous_collections_do_not_mix_their_errors(self) -> None:
    errors = [OSError(f'worker-{index}') for index in range(8)]
    barrier = Barrier(8)

    def close(index):
      """Run several collections concurrently on the same manager."""

      try:
        with self.manager.collect(f'worker-{index}'):
          barrier.wait(timeout=5)
          with self.manager.step('close'):
            Mock(side_effect=errors[index])()
          barrier.wait(timeout=5)
      except ExceptionGroup as caught:
        return caught
      self.fail('the worker error was lost')

    with ThreadPoolExecutor(max_workers=8) as executor:
      groups = list(executor.map(close, range(8)))
    for index, group in enumerate(groups):
      assert group is not None
      self.assertEqual(group.exceptions, (errors[index],))
    self.assertEqual(len(self.manager.snapshot()), 8)
    self.assertEqual(len(self.emitted), 8)

  def test_background_capture_does_not_pollute_collection(self) -> None:
    close_error, background_error = OSError('close'), ValueError('background')
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('close'):
          Mock(side_effect=close_error)()
        with ThreadPoolExecutor(max_workers=1) as executor:
          executor.submit(self.manager.capture_exception,
                          background_error).result(timeout=5)
    self.assertEqual(caught.exception.exceptions, (close_error,))
    with self.assertRaises(ExceptionGroup) as summary:
      self.manager.raise_if_failed('run')
    self.assertEqual(summary.exception.exceptions,
                     (close_error, background_error))
    self.assertEqual(len(self.emitted), 2)

  def test_body_error_does_not_shadow_step_errors(self) -> None:
    first, body_error = OSError('close'), ValueError('body')
    later = Mock()
    with self.assertRaises(ExceptionGroup) as caught:
      with self.manager.collect('finish'):
        with self.manager.step('close'):
          Mock(side_effect=first)()
        Mock(side_effect=body_error)()
        later()
    self.assertEqual(caught.exception.exceptions, (first, body_error))
    later.assert_not_called()
    self.assertEqual(len(self.emitted), 2)
    with self.manager.collect('retry'):
      Mock()()

  def test_step_and_record_require_an_active_collection(self) -> None:
    operation = Mock()
    with self.assertRaisesRegex(RuntimeError, 'inside collect'):
      with self.manager.step('close'):
        operation()
    operation.assert_not_called()
    with self.assertRaisesRegex(RuntimeError, 'inside collect'):
      self.manager._record_context_exception(OSError('close'), 'close')
    self.assertEqual(self.manager.snapshot(), ())

  def test_collection_requires_a_nonempty_label(self) -> None:
    operation = Mock()
    invalid_labels: tuple[Any, ...] = ('', None)
    for label in invalid_labels:
      with self.subTest(label=label):
        with self.assertRaises(ValueError):
          with self.manager.collect(label):
            operation()
    operation.assert_not_called()
