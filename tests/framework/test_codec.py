# coding: utf-8

"""Exception graph transport, per-node fallback and remote diagnostics."""

from dataclasses import replace
from io import StringIO
import pickle
import subprocess
import sys
from threading import Lock
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from crappy import _exceptions, _global
from crappy._collection import CollectionEntry, CollectionUnavailableError
from crappy._exceptions import (DefinitionError, FailureTransportError,
                                RemoteBaseFailure, RemoteFailure,
                                RemoteKeyboardInterrupt, RemoteSystemExit)
from crappy._failure import FailureManager, FailureReport


class _BadMessage(Exception):
  def __str__(self):
    raise ValueError('broken formatter')


def _fail_to_load():
  raise RuntimeError('reconstruction failed')


class _FailingLoads(Exception):
  def __reduce__(self):
    return _fail_to_load, ()


def _interrupt_loading():
  raise KeyboardInterrupt('interrupted reconstruction')


class _InterruptingLoads(Exception):
  def __reduce__(self):
    return _interrupt_loading, ()


class _WrongType(Exception):
  def __reduce__(self):
    return ValueError, ('different type',)


def _different_ancestry():
  error_type = type('_WrongAncestry', (OSError,), {'__module__': __name__})
  return error_type('different ancestry')


class _WrongAncestry(Exception):
  def __reduce__(self):
    return _different_ancestry, ()


class _TaggedGroup(ExceptionGroup):
  tag: str

  def __new__(cls, message, exceptions, tag):
    group = super().__new__(cls, message, exceptions)
    setattr(group, 'tag', tag)
    return group

  def __init__(self, message, exceptions, tag):
    super().__init__(message, exceptions)

  def derive(self, exceptions):
    """Preserves the tag when deriving a subgroup."""

    return type(self)(self.message, exceptions, self.tag)

  def __reduce__(self):
    return type(self), (self.message, self.exceptions, self.tag)


class _BrokenGroup(_TaggedGroup):
  def derive(self, exceptions):
    """Simulates an incompatible custom group contract."""

    raise TypeError('broken derivation')


class _BrokenGroupLoads(_TaggedGroup):
  def __reduce__(self):
    return _fail_to_load, ()


class _BrokenRestoredGroup(_TaggedGroup):
  def derive(self, exceptions):
    """Accepts source derivation but fails after prototype reconstruction."""

    if self.tag == 'restored':
      raise TypeError('restored prototype cannot derive')
    return super().derive(exceptions)

  def __reduce__(self):
    return type(self), (self.message, self.exceptions, 'restored')


class _EncodingInterrupt(Exception):
  def __reduce__(self):
    raise KeyboardInterrupt('interrupted serialization')


class _SingletonError(Exception):
  def __reduce__(self):
    return _get_singleton, ()


_singleton_error = _SingletonError('shared reconstruction')


def _get_singleton():
  return _singleton_error


class _MutatingSerializer(Exception):
  target: ValueError

  def __reduce__(self):
    self.target.args = ('mutated message',)
    self.target.add_note('mutated note')
    return type(self), self.args


class TestExceptionCodec(TestCase):
  def setUp(self) -> None:
    """Observe source-only logging and final-owner reconstruction."""

    self.source = FailureManager('Camera', 'main', owner_path='Camera.worker')
    self.parent = FailureManager('coordinator', 'finish')
    self.source_logger = patch.object(self.source, '_logger').start()
    self.parent_logger = patch.object(self.parent, '_logger').start()
    self.addCleanup(patch.stopall)

  def round_trip(self, error: BaseException) -> tuple[FailureReport, Any]:
    """Transfer built-in wire values and materialize at the final owner."""

    report = self.source.capture_exception(error)
    assert report is not None
    wire = pickle.loads(pickle.dumps(report.to_wire()))
    received = FailureReport.from_wire(wire)
    self.parent.forward(received)
    self.assertEqual(self.parent.snapshot(), ((received, None),))
    with self.assertRaises(BaseExceptionGroup) as caught:
      self.parent.raise_if_failed('summary')
    restored = caught.exception.exceptions[0]
    self.assertIs(self.parent.snapshot()[0][1], restored)
    self.assertIsNone(self.parent.capture_exception(restored))
    self.parent_logger.error.assert_not_called()
    return received, restored

  def test_value_error_type_message_and_except_star(self) -> None:
    report, restored = self.round_trip(ValueError('invalid value'))
    self.assertIs(type(restored), ValueError)
    self.assertEqual(restored.args, ('invalid value',))
    self.assertEqual(report.nodes[0].mro,
                     ('builtins.ValueError', 'builtins.Exception',
                      'builtins.BaseException', 'builtins.object'))
    caught = []
    try:
      raise ExceptionGroup('run', [restored])
    except* ValueError as errors:
      caught.extend(errors.exceptions)
    self.assertEqual(caught, [restored])
    self.source_logger.error.assert_called_once_with(str(report))

  def test_os_error_retains_errno_and_both_filenames(self) -> None:
    error = OSError(18, 'cross-device rename', 'source', None, 'destination')
    _, restored = self.round_trip(error)
    self.assertIs(type(restored), type(error))
    self.assertEqual(restored.errno, error.errno)
    self.assertEqual(restored.filename, 'source')
    self.assertEqual(restored.filename2, 'destination')

  def test_termination_types_and_codes(self) -> None:
    for error in (KeyboardInterrupt('interrupt'), SystemExit(42),
                  SystemExit('exit message'), SystemExit(None),
                  SystemExit(False),
                  BaseException('custom termination')):
      with self.subTest(error=repr(error)):
        self.source.clear()
        self.parent.clear()
        report, restored = self.round_trip(error)
        self.assertIs(type(restored), type(error))
        self.assertEqual(restored.args, error.args)
        if isinstance(error, SystemExit):
          self.assertEqual(restored.code, error.code)
          self.assertEqual(report.nodes[0].system_exit_code, error.code)

  def test_local_class_fallback_preserves_interruption_ancestry(self) -> None:
    for base, expected in ((Exception, RemoteFailure),
                           (KeyboardInterrupt, RemoteKeyboardInterrupt),
                           (SystemExit, RemoteSystemExit),
                           (BaseException, RemoteBaseFailure)):
      with self.subTest(base=base):
        self.source.clear()
        self.parent.clear()

        class LocalError(base):
          """An Exception type unavailable to pickle outside this method."""

          pass
        report, restored = self.round_trip(LocalError(17))
        self.assertIs(type(restored), expected)
        self.assertIs(restored.report, report)
        self.assertIs(restored.node, report.nodes[0])
        self.assertIsNotNone(restored.node.serialization_error)
        if base is SystemExit:
          self.assertEqual(restored.code, 17)

  def test_unpicklable_leaf_does_not_downgrade_its_siblings(self) -> None:
    broken = ValueError('cannot pickle this leaf')
    setattr(broken, 'lock', Lock())
    good = OSError(2, 'missing file', 'input.txt')
    report, restored = self.round_trip(BaseExceptionGroup(
        'outer', [ExceptionGroup('inner', [broken, good]),
                  KeyboardInterrupt()]))
    self.assertIs(type(restored), BaseExceptionGroup)
    inner, interrupt = restored.exceptions
    self.assertIs(type(inner), ExceptionGroup)
    self.assertIs(type(inner.exceptions[0]), RemoteFailure)
    self.assertIs(type(inner.exceptions[1]), FileNotFoundError)
    self.assertEqual(inner.exceptions[1].filename, 'input.txt')
    self.assertIs(type(interrupt), KeyboardInterrupt)
    self.assertEqual(len(self.source.snapshot()), 1)
    self.assertEqual(self.source_logger.error.call_count, 1)
    self.assertEqual(report.nodes[0].group_message, 'outer')

  def test_invalid_reconstruction_uses_leaf_fallbacks(self) -> None:
    for error in (_FailingLoads('loads'), _WrongType('type'),
                  _WrongAncestry('ancestry')):
      with self.subTest(error=type(error)):
        self.source.clear()
        self.parent.clear()
        report, restored = self.round_trip(error)
        self.assertIsNotNone(report.nodes[0].original_bytes)
        self.assertIsNone(report.nodes[0].serialization_error)
        self.assertIs(type(restored), RemoteFailure)
        self.assertIsNotNone(restored.reconstruction_error)
        self.assertEqual(restored.node.message, error.args[0])

  def test_reconstruction_interrupt_is_retained_with_the_remote_origin(self):
    report = self.source.capture_exception(_InterruptingLoads('origin'))
    assert report is not None
    self.parent.forward(FailureReport.from_wire(report.to_wire()))
    with self.assertRaises(BaseExceptionGroup) as caught:
      self.parent.raise_if_failed('summary')
    origin, interruption = caught.exception.exceptions
    self.assertIs(type(origin), RemoteFailure)
    self.assertIsInstance(interruption, KeyboardInterrupt)
    self.assertEqual(self.parent.snapshot()[1][0].phase, 'decoding')
    self.assertEqual(len(self.parent.snapshot()), 2)
    self.parent_logger.error.assert_called_once()
    with self.assertRaises(BaseExceptionGroup) as repeated:
      self.parent.raise_if_failed('summary')
    self.assertEqual(repeated.exception.exceptions, (origin, interruption))
    self.parent_logger.error.assert_called_once()

  def test_decoder_and_logger_failures_keep_the_remote_origin(self) -> None:
    report = self.source.capture_exception(_InterruptingLoads('origin'))
    assert report is not None
    self.parent.forward(FailureReport.from_wire(report.to_wire()))
    logger_error = OSError('coordinator logger failed')
    self.parent_logger.error.side_effect = logger_error
    with patch('crappy._failure.sys.stderr', StringIO()):
      with self.assertRaises(BaseExceptionGroup) as caught:
        self.parent.raise_if_failed('summary')
    origin, interruption, reporting_error = caught.exception.exceptions
    self.assertIs(type(origin), RemoteFailure)
    self.assertIsInstance(interruption, KeyboardInterrupt)
    self.assertIs(reporting_error, logger_error)
    self.assertEqual(len(self.parent.snapshot()), 3)
    self.parent_logger.error.assert_called_once()
    with self.assertRaises(BaseExceptionGroup) as repeated:
      self.parent.raise_if_failed('summary')
    self.assertEqual(repeated.exception.exceptions,
                     (origin, interruption, reporting_error))
    self.parent_logger.error.assert_called_once()

  def test_custom_group_prototype_survives_an_unpicklable_child(self) -> None:
    error = ValueError('unpicklable child')
    setattr(error, 'lock', Lock())
    report, restored = self.round_trip(
        _TaggedGroup('tagged', [error, OSError('other')], 'custom metadata'))
    self.assertIs(type(restored), _TaggedGroup)
    self.assertEqual(restored.tag, 'custom metadata')
    self.assertEqual(restored.message, 'tagged')
    self.assertIs(type(restored.exceptions[0]), RemoteFailure)
    self.assertIs(type(restored.exceptions[1]), OSError)
    payload = report.nodes[0].original_bytes
    assert payload is not None
    prototype = pickle.loads(payload)
    self.assertEqual(tuple(type(child) for child in prototype.exceptions),
                     (Exception, Exception))

  def test_custom_group_failure_degrades_only_that_group(self) -> None:
    for group_type in (_BrokenGroup, _BrokenGroupLoads, _BrokenRestoredGroup):
      with self.subTest(group=group_type):
        self.source.clear()
        self.parent.clear()
        _, restored = self.round_trip(group_type(
            'custom', [ValueError('first'), OSError('second')], 'tag'))
        self.assertIs(type(restored), ExceptionGroup)
        self.assertEqual(restored.message, 'custom')
        self.assertEqual(tuple(type(child) for child in restored.exceptions),
                         (ValueError, OSError))
        self.assertTrue(any(group_type.__name__ in note
                            for note in restored.__notes__))

  def test_mixed_ancestry_fallback_keeps_except_star_interrupt_matching(self):
    class MixedInterrupt(Exception, KeyboardInterrupt):
      """A local class with both ordinary and interruption ancestry."""

    _, restored = self.round_trip(ExceptionGroup(
        'mixed ancestry', [MixedInterrupt('interrupt'), ValueError('good')]))
    self.assertIs(type(restored), BaseExceptionGroup)
    caught = []
    try:
      raise restored
    except* KeyboardInterrupt as errors:
      caught.extend(errors.exceptions)
    except* ValueError:
      pass
    self.assertIs(type(caught[0]), RemoteKeyboardInterrupt)

  def test_shared_children_and_cyclic_causes_keep_identity(self) -> None:
    first, second = ValueError('first'), OSError('second')
    shared = RuntimeError('shared cause')
    first.__cause__ = second.__cause__ = shared
    first.__context__ = second
    second.__context__ = first
    shared.__cause__ = shared
    first.__suppress_context__ = False
    inner = ExceptionGroup('shared subgroup', [first, second, first])
    report, restored = self.round_trip(ExceptionGroup('outer', [inner, inner]))
    self.assertEqual(len(report.nodes), 5)
    self.assertIs(restored.exceptions[0], restored.exceptions[1])
    first, second, repeated = restored.exceptions[0].exceptions
    self.assertIs(first, repeated)
    self.assertIs(first.__cause__, second.__cause__)
    assert first.__cause__ is not None
    self.assertIs(first.__cause__.__cause__, first.__cause__)
    self.assertIs(first.__context__, second)
    self.assertIs(second.__context__, first)
    self.assertFalse(first.__suppress_context__)

  def test_distinct_nodes_cannot_restore_to_one_object(self) -> None:
    _, restored = self.round_trip(ExceptionGroup(
        'independent', [_SingletonError('first'), _SingletonError('second')]))
    self.assertIsNot(restored.exceptions[0], restored.exceptions[1])
    self.assertIs(type(restored.exceptions[1]), RemoteFailure)

  def test_independent_reports_cannot_restore_to_one_object(self) -> None:
    for error in (_SingletonError('first'), _SingletonError('second')):
      report = self.source.capture_exception(error)
      assert report is not None
      self.parent.forward(FailureReport.from_wire(report.to_wire()))
    with self.assertRaises(ExceptionGroup) as caught:
      self.parent.raise_if_failed('summary')
    first, second = caught.exception.exceptions
    self.assertIsNot(first, second)
    self.assertIs(type(first), _SingletonError)
    self.assertIs(type(second), RemoteFailure)

  def test_causal_group_can_refer_back_to_the_root(self) -> None:
    error = ValueError('root')
    error.__cause__ = ExceptionGroup('cause', [error, OSError('related')])
    _, restored = self.round_trip(error)
    self.assertIs(restored.__cause__.exceptions[0], restored)

  def test_original_notes_and_stack_are_text_without_live_frames(self) -> None:
    error = ValueError('remote error')
    error.add_note('original note')
    try:
      raise error
    except ValueError:
      report, restored = self.round_trip(error)
    self.assertIsNotNone(error.__traceback__)
    self.assertIsNone(restored.__traceback__)
    self.assertIs(type(report.nodes[0].formatted_stack), str)
    self.assertIn('test_original_notes_and_stack',
                  report.nodes[0].formatted_stack)
    self.assertEqual(restored.__notes__[0], 'original note')
    self.assertTrue(any('owner Camera.worker' in note
                        for note in restored.__notes__))
    self.assertTrue(any(report.nodes[0].formatted_stack in note
                        for note in restored.__notes__))
    self.assertTrue(all(report.formatted_traceback not in note
                        for note in restored.__notes__))

  def test_broken_message_keeps_type_and_safe_text(self) -> None:
    report, restored = self.round_trip(_BadMessage())
    self.assertIs(type(restored), _BadMessage)
    self.assertIn('message unavailable', report.nodes[0].message)
    self.assertIn('_BadMessage', report.formatted_traceback)

  def test_diagnostics_are_snapshotted_before_user_serialization(self) -> None:
    target = ValueError('original message')
    target.add_note('original note')
    mutator = _MutatingSerializer('mutator')
    mutator.target = target
    report, restored = self.round_trip(
        ExceptionGroup('group', [mutator, target]))
    target_node = report.nodes[report.nodes[0].children[1]]
    self.assertEqual(target_node.message, 'original message')
    self.assertEqual(target_node.notes, ('original note',))
    self.assertNotIn('mutated message', report.formatted_traceback)
    self.assertEqual(restored.exceptions[1].__notes__[0], 'original note')
    self.assertNotIn('mutated note', restored.exceptions[1].__notes__)

  def test_encoding_interrupt_is_retained_beside_the_origin(self) -> None:
    error = _EncodingInterrupt('origin')
    with self.assertRaises(KeyboardInterrupt) as caught:
      self.source.capture_exception(error)
    history = self.source.snapshot()
    self.assertEqual(len(history), 2)
    self.assertIs(history[0][1], error)
    self.assertIs(history[1][1], caught.exception)
    self.assertEqual(history[1][0].phase, 'encoding')
    self.assertTrue(all(node.original_bytes is None
                        for node in history[0][0].nodes))
    self.assertEqual(self.source_logger.error.call_count, 2)
    with self.assertRaises(BaseExceptionGroup) as summary:
      self.source.raise_if_failed('summary')
    self.assertEqual(summary.exception.exceptions, (error, caught.exception))

  def test_logger_encoding_interrupt_does_not_shadow_the_origin(self) -> None:
    origin = ValueError('origin')
    logger_error = _EncodingInterrupt('logger failed')
    self.source_logger.error.side_effect = logger_error
    with patch('crappy._failure.sys.stderr', StringIO()) as emergency:
      with self.assertRaises(BaseExceptionGroup) as caught:
        self.source.capture_exception(origin)
    self.assertIs(caught.exception.exceptions[0], logger_error)
    self.assertIsInstance(caught.exception.exceptions[1], KeyboardInterrupt)
    self.assertEqual(tuple(error for _, error in self.source.snapshot()),
                     (origin, *caught.exception.exceptions))
    for text in ('ValueError: origin', '_EncodingInterrupt: logger failed',
                 'KeyboardInterrupt: interrupted serialization'):
      self.assertIn(text, emergency.getvalue())
    self.source_logger.error.assert_called_once()

  def test_encoding_and_logger_failure_keep_all_incidents_once(self) -> None:
    origin = _EncodingInterrupt('origin')
    logger_error = OSError('logger failed')
    self.source_logger.error.side_effect = logger_error
    with patch('crappy._failure.sys.stderr', StringIO()):
      with self.assertRaises(BaseExceptionGroup) as caught:
        self.source.capture_exception(origin)
    self.assertEqual(len(caught.exception.exceptions), 2)
    self.assertIsInstance(caught.exception.exceptions[0], KeyboardInterrupt)
    self.assertIs(caught.exception.exceptions[1], logger_error)
    self.assertEqual(len(self.source.snapshot()), 3)

  def test_simple_exit_code_survives_metadata_only_fallback(self) -> None:
    for code in (9, 'message', False, None, Lock()):
      with self.subTest(code=code):
        self.source.clear()
        self.parent.clear()
        error = SystemExit(code)
        setattr(error, 'lock', Lock())
        report, restored = self.round_trip(error)
        self.assertIs(type(restored), RemoteSystemExit)
        if type(code) in (int, str, bool, type(None)):
          self.assertEqual(restored.code, code)
        else:
          self.assertIsNone(restored.code)
          self.assertTrue(any('SystemExit code' in note
                              for note in report.nodes[0].notes))

  def test_large_and_deep_groups_are_not_flattened(self) -> None:
    error = ExceptionGroup('wide', [ValueError(str(index))
                                    for index in range(30)])
    depth = sys.getrecursionlimit() + 10
    for index in range(depth):
      error = ExceptionGroup(f'depth {index}', [error])
    report, restored = self.round_trip(error)
    self.assertEqual(len(report.nodes), depth + 31)
    for index in reversed(range(depth)):
      self.assertEqual(restored.message, f'depth {index}')
      restored = restored.exceptions[0]
    self.assertEqual(len(restored.exceptions), 30)
    self.assertEqual(tuple(child.args[0] for child in restored.exceptions),
                     tuple(str(index) for index in range(30)))

  def test_deep_diagnostics_keep_notes_and_causal_links(self) -> None:
    error = ValueError('original')
    error.add_note('leaf note')
    cause = OSError('causal explanation')
    cause.add_note('cause note')
    error.__cause__ = cause
    group = ExceptionGroup('inner', [error])
    for _ in range(sys.getrecursionlimit() + 10):
      group = ExceptionGroup('outer', [group])
    report, _ = self.round_trip(group)
    for text in ('ValueError: original', 'OSError: causal explanation',
                 'leaf note', 'cause note', 'Group children:', 'Cause:'):
      self.assertIn(text, report.formatted_traceback)

  def test_codec_receives_a_report_from_a_fresh_python_process(self) -> None:
    probe = """
import pickle
import sys
from unittest.mock import patch
from crappy._failure import FailureManager

class ChildOnlyError(ValueError):
    pass

error = ChildOnlyError('defined only in child')
error.add_note('child note')
error.__cause__ = OSError(2, 'missing input', 'input.txt')
manager = FailureManager('Camera', 'main', owner_path='Camera.worker')
with patch.object(manager, '_logger'):
    report = manager.capture_exception(
        BaseExceptionGroup('worker',
                           [ValueError('good'), error, SystemExit(7)]))
sys.stdout.buffer.write(pickle.dumps(report.to_wire()))
"""
    child = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                           check=True, timeout=20)
    report = FailureReport.from_wire(pickle.loads(child.stdout))
    self.assertNotEqual(report.pid, self.parent.pid)
    self.parent.forward(report)
    with self.assertRaises(BaseExceptionGroup) as caught:
      self.parent.raise_if_failed('summary')
    worker_errors = caught.exception.exceptions[0]
    assert isinstance(worker_errors, BaseExceptionGroup)
    good, fallback, exit_error = worker_errors.exceptions
    self.assertIs(type(good), ValueError)
    assert isinstance(fallback, RemoteFailure)
    self.assertIs(type(fallback), RemoteFailure)
    self.assertEqual(fallback.node.type_name, 'ChildOnlyError')
    self.assertEqual(fallback.__notes__[0], 'child note')
    self.assertIs(type(fallback.__cause__), FileNotFoundError)
    assert isinstance(fallback.__cause__, FileNotFoundError)
    self.assertEqual(fallback.__cause__.filename, 'input.txt')
    assert isinstance(exit_error, SystemExit)
    self.assertEqual(exit_error.code, 7)
    self.parent_logger.error.assert_not_called()

  def test_wire_is_builtins_without_exception_reconstruction(self) -> None:
    report = self.source.capture_exception(_FailingLoads('unloadable'))
    assert report is not None
    wire = report.to_wire()
    pending: list[Any] = [wire]
    while pending:
      value = pending.pop()
      self.assertIn(type(value),
                    (dict, list, str, int, bool, bytes, type(None)))
      if type(value) is dict:
        pending.extend(value.keys())
        pending.extend(value.values())
      elif type(value) is list:
        pending.extend(value)
    with patch('crappy._failure.pickle.loads', side_effect=AssertionError):
      received = FailureReport.from_wire(wire)
      self.parent.forward(received)
      self.assertEqual(self.parent.snapshot(), ((received, None),))
      self.assertEqual(received.to_wire(), wire)
    self.assertEqual(received, report)
    self.assertEqual(wire['version'], 1)
    self.assertFalse(hasattr(report, 'version'))

  def test_invalid_wire_fields_and_references(self) -> None:
    report = self.source.capture_exception(ValueError('error'))
    assert report is not None
    for changes in ({'version': 2}, {'version': True}, {'pid': False},
                    {'root': 99}, {'nodes': ()}, {'unexpected': 'field'}):
      with self.subTest(changes=changes):
        wire = report.to_wire()
        wire.update(changes)
        with self.assertRaises(FailureTransportError):
          FailureReport.from_wire(wire)
    for name, value in (('mro', ['wrong type']), ('notes', [ValueError()]),
                        ('children', [None]), ('cause', 99),
                        ('original_bytes', Lock())):
      with self.subTest(name=name):
        wire = report.to_wire()
        wire['nodes'][0][name] = value
        with self.assertRaises(FailureTransportError):
          FailureReport.from_wire(wire)
    wire = report.to_wire()
    del wire['phase']
    with self.assertRaises(FailureTransportError):
      FailureReport.from_wire(wire)

  def test_cyclic_group_containment_fails_before_unpickling(self) -> None:
    report = self.source.capture_exception(
        ExceptionGroup('group', [ValueError()]))
    assert report is not None
    node = replace(report.nodes[0], children=(0,))
    received = replace(report, nodes=(node, *report.nodes[1:]))
    self.parent.forward(received)
    with patch('crappy._failure.pickle.loads', side_effect=AssertionError):
      with self.assertRaises(FailureTransportError):
        self.parent.raise_if_failed('summary')

  def test_shared_exceptions_moved_without_global_aliases(self) -> None:
    names = ('LinkDataError', 'StartTimeout', 'PrepareError',
             'CameraConfigError', 'CameraPrepareError', 'CameraRuntimeError',
             'T0NotSetError',
             'DefinitionError', 'GeneratorStop', 'SchedulerStop', 'ReaderStop',
             'CrappyFail')
    for name in names:
      with self.subTest(name=name):
        error_type = getattr(_exceptions, name)
        self.assertEqual(error_type.__module__, 'crappy._exceptions')
        self.assertFalse(hasattr(_global, name))
        error = error_type() if name == 'CrappyFail' else error_type('message')
        error.add_note('retained note')
        restored = pickle.loads(pickle.dumps(error))
        self.assertIs(type(restored), error_type)
        self.assertEqual(str(restored), str(error))
        self.assertEqual(restored.__notes__, ['retained note'])
    self.assertEqual(DefinitionError('message').args, ('message',))

  def test_collection_entry_and_import_cause_round_trip(self) -> None:
    entry = CollectionEntry('Device', 'InOut', 'collection.device')
    cause = ModuleNotFoundError("No module named 'dependency'",
                                name='dependency')
    error = CollectionUnavailableError(entry, cause)
    error.__cause__ = cause
    direct = pickle.loads(pickle.dumps(error))
    self.assertEqual(direct.entry, entry)
    self.assertIs(type(direct.cause), ModuleNotFoundError)
    self.assertIs(direct.cause, direct.__cause__)
    self.assertEqual(direct.cause.name, 'dependency')
    _, restored = self.round_trip(error)
    self.assertIs(type(restored), CollectionUnavailableError)
    self.assertEqual(restored.entry, entry)
    self.assertEqual(restored.cause.name, 'dependency')
    self.assertIs(restored.cause, restored.__cause__)
    self.assertIs(type(restored.__cause__), ModuleNotFoundError)
    self.assertEqual(restored.__cause__.name, 'dependency')
