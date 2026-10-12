# coding: utf-8

"""Exception containment, cancellation, and metadata-preserving subgroups."""

from unittest import TestCase
import sys

from crappy._exceptions import StopRequested
from crappy._failure import FailureManager


class _TaggedGroup(ExceptionGroup):
  tag: str

  def __new__(cls, message, exceptions, tag):
    group = super().__new__(cls, message, exceptions)
    setattr(group, 'tag', tag)
    return group

  def __init__(self, message, exceptions, tag):
    super().__init__(message, exceptions)

  def derive(self, exceptions):
    return type(self)(self.message, exceptions, self.tag)


class _BrokenGroup(ExceptionGroup):
  def derive(self, exceptions):
    raise TypeError('incompatible subgroup constructor')


class _WrongChildrenGroup(ExceptionGroup):
  def derive(self, exceptions):
    return self


class TestExceptionGroups(TestCase):
  def setUp(self) -> None:
    self.manager = FailureManager('owner', 'main')

  def test_singleton_is_always_a_group(self) -> None:
    error = ValueError('invalid')
    group = self.manager._make_exception_group('failures', [error])
    assert isinstance(group, BaseExceptionGroup)
    self.assertIs(type(group), ExceptionGroup)
    self.assertEqual(group.exceptions, (error,))
    self.assertIsNone(self.manager._make_exception_group('failures', []))

  def test_interrupts_are_siblings_with_unchanged_causes(self) -> None:
    first, second = KeyboardInterrupt('first'), KeyboardInterrupt('second')
    cause, error = RuntimeError('cause'), OSError('close')
    first.__cause__ = cause
    group = self.manager._make_exception_group('failures',
                                               [first, error, second])
    assert isinstance(group, BaseExceptionGroup)
    self.assertIs(type(group), BaseExceptionGroup)
    self.assertEqual(group.exceptions, (first, error, second))
    self.assertIs(first.__cause__, cause)

  def test_nested_members_remain_catchable_with_except_star(self) -> None:
    value, io = ValueError('value'), OSError('io')
    interrupt = KeyboardInterrupt()
    nested = BaseExceptionGroup('nested', [interrupt, io])
    values = ExceptionGroup('values', [value])
    group = self.manager._make_exception_group('run', [values, nested])
    assert isinstance(group, BaseExceptionGroup)
    caught = {}
    try:
      raise group
    except* ValueError as errors:
      caught['value'] = errors
    except* OSError as errors:
      caught['io'] = errors
    except* KeyboardInterrupt as errors:
      caught['interrupt'] = errors
    for key, error in (('value', value), ('io', io), ('interrupt', interrupt)):
      self.assertIn(error,
                    tuple(self.manager._iter_exception_members(caught[key])))

  def test_cancellation_only_group_disappears(self) -> None:
    nested = ExceptionGroup('nested', [StopRequested()])
    group = ExceptionGroup('cancel', [StopRequested('wait'), nested])
    self.assertIsNone(self.manager._strip_cancellations(group))
    self.assertIsNone(self.manager._make_exception_group('run', [group]))

  def test_filtered_group_preserves_metadata(self) -> None:
    error = ValueError('keep')
    inner = ExceptionGroup('inner', [StopRequested(), error])
    original = BaseExceptionGroup('outer', [KeyboardInterrupt(), inner])
    cause, context = OSError('cause'), RuntimeError('context')
    original.__cause__, original.__context__ = cause, context
    original.__suppress_context__ = False
    original.add_note('original note')
    try:
      raise original
    except BaseExceptionGroup as caught:
      traceback = caught.__traceback__
      result = self.manager._strip_cancellations(caught)
    assert isinstance(result, BaseExceptionGroup)
    self.assertEqual(result.message, 'outer')
    self.assertIs(result.exceptions[0], original.exceptions[0])
    self.assertEqual(result.exceptions[1].message, 'inner')
    self.assertEqual(result.exceptions[1].exceptions, (error,))
    self.assertIs(result.__traceback__, traceback)
    self.assertIs(result.__cause__, cause)
    self.assertIs(result.__context__, context)
    self.assertFalse(result.__suppress_context__)
    self.assertEqual(result.__notes__, ['original note'])
    self.assertIsNot(result.__notes__, original.__notes__)
    self.assertEqual(len(inner.exceptions), 2)

  def test_unchanged_group_is_returned_by_identity(self) -> None:
    original = ExceptionGroup('outer', [ValueError(),
                                        ExceptionGroup('inner', [OSError()])])
    self.assertIs(self.manager._strip_cancellations(original), original)

  def test_cancellation_in_a_cause_is_not_promoted_or_removed(self) -> None:
    original = RuntimeError('active operation failed')
    cause = StopRequested('explanation')
    original.__cause__ = cause
    self.assertIs(self.manager._strip_cancellations(original), original)
    self.assertIs(original.__cause__, cause)

  def test_endpoint_errors_require_shutdown_classification(self) -> None:
    # Shared helpers cannot infer from an exception alone whether shutdown was
    # expected. The owned call site must translate it to StopRequested.
    errors = (EOFError('active peer'), OSError('close'), ValueError('state'))
    for original in errors:
      with self.subTest(error=type(original).__name__):
        self.assertIs(self.manager._strip_cancellations(original), original)

  def test_custom_subgroup_and_broken_derive_fallback(self) -> None:
    error = ValueError('keep')
    original = _TaggedGroup('tagged', [StopRequested(), error], 'metadata')
    result = self.manager._strip_cancellations(original)
    assert isinstance(result, _TaggedGroup)
    self.assertIs(type(result), _TaggedGroup)
    self.assertEqual(result.tag, 'metadata')
    self.assertEqual(result.exceptions, (error,))
    for group_type in (_BrokenGroup, _WrongChildrenGroup):
      with self.subTest(group_type=group_type):
        original = group_type('broken', [StopRequested(), error])
        result = self.manager._strip_cancellations(original)
        assert isinstance(result, BaseExceptionGroup)
        self.assertIs(type(result), ExceptionGroup)
        self.assertEqual(result.message, 'broken')
        self.assertEqual(result.exceptions, (error,))
        self.assertIn(group_type.__name__, result.__notes__[-1])

  def test_member_walk_ignores_causes_and_shared_references(self) -> None:
    error = ValueError('shared')
    error.__cause__ = OSError('cause')
    group = ExceptionGroup('outer', [error, ExceptionGroup('inner', [error])])
    members = tuple(self.manager._iter_exception_members(group))
    self.assertEqual(sum(member is error for member in members), 1)
    self.assertNotIn(error.__cause__, members)
    self.assertIs(self.manager._filter_exception(group, lambda member: True),
                  group)

  def test_filter_beyond_recursion_limit(self) -> None:
    error = ValueError('leaf')
    group = ExceptionGroup('inner', [StopRequested(), error])
    for depth in range(sys.getrecursionlimit() + 10):
      group = ExceptionGroup(f'depth-{depth}', [group])
    result = self.manager._strip_cancellations(group)
    assert isinstance(result, BaseExceptionGroup)
    while isinstance(result.exceptions[0], BaseExceptionGroup):
      result = result.exceptions[0]
    self.assertEqual(result.exceptions, (error,))

  def test_shared_subgroups_remain_shared_after_filtering(self) -> None:
    error = ValueError('leaf')
    shared = ExceptionGroup('shared', [StopRequested(), error])
    group = ExceptionGroup('outer', [shared, shared])
    result = self.manager._strip_cancellations(group)
    assert isinstance(result, BaseExceptionGroup)
    self.assertIs(result.exceptions[0], result.exceptions[1])
    self.assertEqual(result.exceptions[0].exceptions, (error,))
