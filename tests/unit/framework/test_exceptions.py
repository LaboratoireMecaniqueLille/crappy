# coding: utf-8

"""New framework exception constructors have predictable pickle contracts."""

from dataclasses import replace
import pickle
from unittest import TestCase

from crappy._exceptions import (FailureTransportError, ProcessExitError,
                                RemoteBaseFailure,
                                RemoteFailure, RemoteFailureMetadata,
                                RemoteKeyboardInterrupt,
                                RemoteSystemExit, ShutdownTimeoutError)
from crappy._failure import ExceptionNode, FailureReport


class TestFrameworkExceptions(TestCase):
  def setUp(self) -> None:
    """A fallback reuses an existing report and one of its node records."""

    self.node = ExceptionNode('package', 'OriginalError', 'original message',
                              ('package.OriginalError', 'builtins.Exception',
                               'builtins.BaseException', 'builtins.object'),
                              formatted_stack='remote traceback')
    self.report = FailureReport(
        failure_id='a' * 32, block_name='Block-1', owner_path='Block-1.worker',
        process_name='worker', pid=123, phase='main',
        formatted_traceback='complete diagnostic', nodes=(self.node,))

  def test_fallback_metadata_round_trips_with_ancestry_and_notes(self) -> None:
    reason = 'original object could not be reconstructed'
    fallbacks = ((RemoteFailure, Exception),
                 (RemoteKeyboardInterrupt, KeyboardInterrupt),
                 (RemoteSystemExit, SystemExit),
                 (RemoteBaseFailure, BaseException))
    for error_type, base_type in fallbacks:
      with self.subTest(error_type=error_type):
        original_type = type('OriginalError', (base_type,),
                             {'__module__': 'package'})
        mro = tuple(f'{base.__module__}.{base.__qualname__}'
                    for base in original_type.__mro__)
        node = replace(self.node, mro=mro)
        report = replace(self.report, nodes=(node,))
        original = RemoteFailureMetadata.from_report(report, 0, reason)
        self.assertIs(type(original), error_type)
        self.assertIs(original.report, report)
        self.assertIs(original.node, node)
        original.add_note('retained note')
        restored = pickle.loads(pickle.dumps(original))
        self.assertIs(type(restored), error_type)
        self.assertIsInstance(restored, base_type)
        self.assertEqual(restored.node.mro, mro)
        self.assertEqual(restored.report, report)
        self.assertEqual(restored.node_index, 0)
        self.assertIs(restored.node, restored.report.nodes[0])
        self.assertEqual(restored.report.failure_id, self.report.failure_id)
        self.assertEqual(restored.report.phase, 'main')
        self.assertEqual(restored.node.formatted_stack, 'remote traceback')
        self.assertEqual(restored.reconstruction_error, reason)
        self.assertEqual(restored.__notes__, ['retained note'])
        self.assertIn('package.OriginalError: original message', str(restored))
    for code in (42, 'exit message', None, False):
      with self.subTest(code=code):
        node = replace(self.node,
                       mro=('package.OriginalError', 'builtins.SystemExit',
                            'builtins.BaseException', 'builtins.object'),
                       system_exit_code=code)
        report = replace(self.report, nodes=(node,))
        original = RemoteFailureMetadata.from_report(
            report, 0, 'original unavailable')
        self.assertIsInstance(original, SystemExit)
        restored = pickle.loads(pickle.dumps(original))
        assert isinstance(restored, SystemExit)
        self.assertEqual(restored.code, code)

  def test_interrupt_fallback_is_catchable_with_except_star(self) -> None:
    class UserCancelled(KeyboardInterrupt):
      pass

    node = ExceptionNode(
        UserCancelled.__module__, UserCancelled.__qualname__, 'interrupted',
        tuple(f'{base.__module__}.{base.__qualname__}'
              for base in UserCancelled.__mro__))
    # The actual local class cannot be pickled, but its ancestry remains data.
    with self.assertRaises((AttributeError, pickle.PicklingError)):
      pickle.dumps(UserCancelled())
    report = replace(self.report, nodes=(node,))
    error = RemoteFailureMetadata.from_report(
            report, 0, 'original unavailable')
    caught = []
    try:
      raise BaseExceptionGroup('run', [error])
    except* KeyboardInterrupt as errors:
      caught.extend(errors.exceptions)
    self.assertEqual(caught, [error])

  def test_synthetic_exceptions_round_trip_diagnostic_attributes(self) -> None:
    errors = (
        ProcessExitError('worker exited without completion',
                         process_name='worker', pid=123, phase='main',
                         exitcode=-9),
        ShutdownTimeoutError('worker finish timed out', process_name='worker',
                             pid=123, phase='finish', timeout=3.0),
        FailureTransportError('incomplete report', failure_id='c' * 32,
                              process_name='worker', pid=123, phase='finish'),
    )
    for original in errors:
      with self.subTest(error_type=type(original)):
        original.add_note('operation note')
        restored = pickle.loads(pickle.dumps(original))
        self.assertIs(type(restored), type(original))
        self.assertEqual(restored.args, original.args)
        self.assertEqual(restored.__dict__, original.__dict__)

  def test_fallback_metadata_rejects_live_objects(self) -> None:
    for field in ('message', 'formatted_stack', 'serialization_error',
                  'system_exit_code'):
      with self.subTest(field=field):
        with self.assertRaises(TypeError):
          replace(self.node, **{field: ValueError('live exception')})
    invalid_reason = ValueError('live exception')
    with self.assertRaises(TypeError):
      RemoteFailure(self.report, 0, invalid_reason)  # type: ignore[arg-type]

  def test_fallback_requires_a_text_reconstruction_reason(self) -> None:
    for factory in (RemoteFailure, RemoteFailureMetadata.from_report):
      with self.subTest(factory=factory):
        with self.assertRaises(TypeError):
          factory(self.report, 0)
        with self.assertRaises(TypeError):
          factory(self.report, 0, None)

  def test_process_exit_requires_all_origin_fields(self) -> None:
    fields = dict(message='worker exit', process_name='worker', pid=123,
                  phase='main', exitcode=9)
    for name in fields:
      with self.subTest(missing=name):
        arguments = dict(fields)
        arguments.pop(name)
        with self.assertRaises(TypeError):
          ProcessExitError(**arguments)

  def test_fallback_selects_the_failed_leaf_not_the_report_root(self) -> None:
    root = ExceptionNode('builtins', 'ExceptionGroup', 'group',
                         tuple(f'{base.__module__}.{base.__qualname__}'
                               for base in ExceptionGroup.__mro__),
                         children=(1, 2), group_message='group')
    sibling = ExceptionNode('builtins', 'ValueError', 'sibling',
                            tuple(f'{base.__module__}.{base.__qualname__}'
                                  for base in ValueError.__mro__))
    report = replace(self.report, nodes=(root, sibling, self.node))
    original = RemoteFailureMetadata.from_report(
        report, 2, 'local exception type unavailable')
    self.assertIs(original.node, self.node)
    self.assertEqual(str(original), 'package.OriginalError: original message')
    restored = pickle.loads(pickle.dumps(original))
    self.assertEqual(restored.node_index, 2)
    self.assertIs(restored.node, restored.report.nodes[2])
    self.assertEqual(restored.node.formatted_stack, 'remote traceback')
    self.assertEqual(restored.report.formatted_traceback,
                     'complete diagnostic')
    for group_node in (root, replace(root, children=(), group_message=None)):
      with self.subTest(group_message=group_node.group_message):
        group_report = replace(report, nodes=(group_node, sibling, self.node))
        with self.assertRaises(ValueError):
          RemoteFailureMetadata.from_report(group_report, 0, 'group')

  def test_fallback_requires_a_valid_report_node_index(self) -> None:
    for index in (-1, 1, True):
      with self.subTest(index=index):
        for factory in (RemoteFailure, RemoteFailureMetadata.from_report):
          with self.assertRaises(ValueError):
            factory(self.report, index, 'original unavailable')
    with self.assertRaises(TypeError):
      replace(self.report, nodes=())

  def test_report_dataclasses_round_trip_without_live_objects(self) -> None:
    node = replace(self.node, cause=0, context=0,
                   notes=('original note',), original_bytes=b'opaque bytes')
    report = replace(self.report, nodes=(node,))
    restored = pickle.loads(pickle.dumps(report))
    self.assertEqual(restored, report)
    self.assertEqual(restored.nodes[0].cause, 0)
    self.assertEqual(restored.nodes[0].notes, ('original note',))
    self.assertEqual(restored.nodes[0].mro, node.mro)

  def test_fallback_does_not_guess_from_the_original_name(self) -> None:
    original_type = type('KeyboardInterrupt', (Exception,),
                         {'__module__': 'package'})
    node = ExceptionNode(
        original_type.__module__, original_type.__qualname__, 'error',
        tuple(f'{base.__module__}.{base.__qualname__}'
              for base in original_type.__mro__))
    report = replace(self.report, nodes=(node,))
    restored = RemoteFailureMetadata.from_report(
        report, 0, 'original unavailable')
    self.assertIs(type(restored), RemoteFailure)

  def test_multiple_inheritance_keeps_interruption_semantics(self) -> None:
    fallbacks = ((KeyboardInterrupt, RemoteKeyboardInterrupt),
                 (SystemExit, RemoteSystemExit))
    for special_base, expected in fallbacks:
      with self.subTest(base=special_base):
        original_type = type('OriginalError', (Exception, special_base),
                             {'__module__': 'package'})
        mro = tuple(f'{base.__module__}.{base.__qualname__}'
                    for base in original_type.__mro__)
        self.assertLess(mro.index('builtins.Exception'),
                        mro.index(f'builtins.{special_base.__qualname__}'))
        report = replace(self.report, nodes=(replace(self.node, mro=mro),))
        restored = RemoteFailureMetadata.from_report(
            report, 0, 'original unavailable')
        self.assertIs(type(restored), expected)

  def test_ancestry_can_include_non_exception_mixins(self) -> None:
    class Mixin:
      pass

    original_type = type('OriginalError', (Exception, Mixin),
                         {'__module__': 'package'})
    mro = tuple(f'{base.__module__}.{base.__qualname__}'
                for base in original_type.__mro__)
    self.assertGreater(mro.index(f'{Mixin.__module__}.{Mixin.__qualname__}'),
                       mro.index('builtins.BaseException'))
    report = replace(self.report, nodes=(replace(self.node, mro=mro),))
    self.assertIs(type(RemoteFailureMetadata.from_report(
            report, 0, 'original unavailable')),
                  RemoteFailure)
