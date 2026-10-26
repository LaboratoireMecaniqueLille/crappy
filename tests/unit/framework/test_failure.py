# coding: utf-8

"""Local report identity, first-receipt retention, and emission boundaries."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import gc
from inspect import signature
from io import StringIO
from threading import Barrier
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch
from uuid import UUID
from weakref import ref

from crappy._exceptions import FailureTransportError, StopRequested
from crappy._failure import ExceptionNode, FailureManager, FailureReport


class _BadMessage(Exception):
  def __str__(self):
    raise ValueError('broken string formatter')


class TestFailureManager(TestCase):
  def setUp(self) -> None:
    self.emitted = []
    self.manager = FailureManager('Block-1', 'main',
                                  owner_path='Block-1.worker')
    logger = patch.object(self.manager, '_logger')
    self.logger = logger.start()
    self.logger.error.side_effect = self.emitted.append
    self.addCleanup(logger.stop)

  def test_report_is_immutable_and_local_exception_is_retained(self) -> None:
    error = ValueError('original')
    error.add_note('original note')
    try:
      raise error
    except ValueError as caught:
      report = self.manager.capture_exception(caught)
    assert report is not None
    self.assertEqual(UUID(hex=report.failure_id).hex, report.failure_id)
    for field in ('block_name', 'owner_path', 'process_name', 'pid', 'phase'):
      self.assertEqual(getattr(report, field), getattr(self.manager, field))
    self.assertEqual(report.nodes[0].module, 'builtins')
    self.assertEqual(report.nodes[0].type_name, 'ValueError')
    self.assertEqual(report.nodes[0].message, 'original')
    self.assertIn('Traceback', report.formatted_traceback)
    self.assertIn('original note', report.formatted_traceback)
    self.assertEqual(len(report.nodes), 1)
    self.assertFalse(hasattr(report, 'root'))
    self.assertEqual(report.nodes[0].notes, ('original note',))
    self.assertIs(self.manager.snapshot()[0][1], error)
    self.assertIsNotNone(error.__traceback__)
    with self.assertRaises(FrozenInstanceError):
      setattr(report, 'failure_id', 'overwrite')

  def test_report_string_preserves_origin_and_diagnostics(self) -> None:
    diagnostic = ('Traceback (most recent call last):\n'
                  'ValueError: first line\nsecond line\n'
                  'additional diagnostic note\n')
    report = FailureReport(
        failure_id='0123456789abcdef0123456789abcdef',
        block_name='Block-1', owner_path='Block-1.worker',
        process_name='worker', pid=123, phase='finish',
        formatted_traceback=diagnostic,
        nodes=(ExceptionNode('builtins', 'ValueError', 'first line\nsecond line',
                             tuple(f'{base.__module__}.{base.__qualname__}'
                                   for base in ValueError.__mro__)),))
    self.assertEqual(
        str(report),
        'Failure 0123456789abcdef0123456789abcdef: Block Block-1, '
        'owner Block-1.worker, process worker (PID 123), phase finish\n'
        + diagnostic)
    self.assertTrue(repr(report).startswith('FailureReport('))

  def test_recapture_and_forward_do_not_emit_again(self) -> None:
    error = OSError('same incident')
    report = self.manager.capture_exception(error)
    assert report is not None
    self.assertIsNone(self.manager.capture_exception(error))
    self.assertIsNone(self.manager.forward(report))
    self.assertEqual(self.emitted, [str(report)])
    with self.assertRaises(ExceptionGroup) as caught:
      self.manager.raise_if_failed('summary')
    self.assertEqual(caught.exception.message, 'summary')
    self.assertEqual(caught.exception.exceptions, (error,))
    self.assertEqual(self.emitted, [str(report)])

  def test_independent_identical_looking_errors(self) -> None:
    first, second = ValueError('same'), ValueError('same')
    self.manager.capture_exception(first)
    self.manager.capture_exception(second)
    self.assertEqual(len(self.emitted), 2)
    reports = self.manager.snapshot()
    self.assertNotEqual(reports[0][0].failure_id, reports[1][0].failure_id)
    with self.assertRaises(ExceptionGroup) as caught:
      self.manager.raise_if_failed('summary')
    self.assertEqual(caught.exception.exceptions, (first, second))

  def test_runtime_interrupt_and_error_are_raised_together(self) -> None:
    interrupt, error = KeyboardInterrupt('interrupted'), OSError('independent')
    self.manager.capture_exception(interrupt)
    self.manager.capture_exception(error)
    with self.assertRaises(BaseExceptionGroup) as caught:
      self.manager.raise_if_failed('run')
    self.assertNotIsInstance(caught.exception, ExceptionGroup)
    self.assertEqual(caught.exception.message, 'run')
    self.assertEqual(caught.exception.exceptions, (interrupt, error))
    self.assertEqual(len(self.emitted), 2)

  def test_later_group_reports_only_new_subtrees(self) -> None:
    old, new = ValueError('old'), OSError('new')
    old_report = self.manager.capture_exception(old)
    assert old_report is not None
    inner = ExceptionGroup('inner', [old, new])
    inner.add_note('inner note')
    original = ExceptionGroup('later', [old, inner])
    new_report = self.manager.capture_exception(original)
    assert new_report is not None
    retained = self.manager.snapshot()
    self.assertEqual(len(retained), 2)
    self.assertIs(retained[0][1], old)
    later = retained[1][1]
    assert isinstance(later, ExceptionGroup)
    self.assertEqual(later.message, 'later')
    nested = later.exceptions[0]
    assert isinstance(nested, ExceptionGroup)
    self.assertEqual(nested.message, 'inner')
    self.assertEqual(nested.exceptions, (new,))
    self.assertEqual(nested.__notes__, ['inner note'])
    self.assertNotIn('ValueError: old', new_report.formatted_traceback)
    self.assertNotEqual(old_report.failure_id, new_report.failure_id)
    self.assertIsNone(self.manager.capture_exception(original))
    self.assertIsNone(self.manager.capture_exception(later))
    self.assertEqual(len(self.emitted), 2)
    self.assertEqual(original.exceptions, (old, inner))

  def test_cancellation_never_enters_the_history(self) -> None:
    self.assertIsNone(self.manager.capture_exception(StopRequested()))
    self.assertIsNone(self.manager.capture_exception(
        ExceptionGroup('cancel', [StopRequested()])))
    error = OSError('real')
    self.manager.capture_exception(
        ExceptionGroup('mixed', [StopRequested(), error]))
    retained = self.manager.snapshot()[0][1]
    assert isinstance(retained, ExceptionGroup)
    self.assertEqual(retained.exceptions, (error,))
    self.assertEqual(len(self.emitted), 1)

  def test_phase_override_does_not_mutate_manager(self) -> None:
    report = self.manager.capture_exception(OSError('finish'), phase='finish')
    assert report is not None
    self.assertEqual(report.phase, 'finish')
    self.assertEqual(self.manager.phase, 'main')

  def test_forward_retains_original_origin_id_and_identity(self) -> None:
    report = self.manager.capture_exception(ValueError('worker'))
    assert report is not None
    parent_records = []
    parent = FailureManager('parent', 'finish')
    with self.assertRaises(TypeError):
      parent.forward(report, ValueError('late attachment'))
    with patch.object(parent, '_logger') as logger:
      logger.error.side_effect = parent_records.append
      self.assertIsNone(parent.forward(report))
      self.assertEqual(parent.snapshot(), ((report, None),))
      with self.assertRaises(ExceptionGroup) as caught:
        parent.raise_if_failed('summary')
      restored, = caught.exception.exceptions
      self.assertIs(type(restored), ValueError)
      self.assertEqual(str(restored), 'worker')
      self.assertIsNone(parent.capture_exception(restored))
      self.assertIsNone(parent.forward(report))
    self.assertEqual(parent_records, [])
    self.assertIs(parent.snapshot()[0][0], report)
    self.assertIs(parent.snapshot()[0][1], restored)

  def test_inconsistent_same_id_is_rejected_without_overwrite(self) -> None:
    report = self.manager.capture_exception(ValueError('original'))
    assert report is not None
    inconsistent = replace(report, formatted_traceback='different contents')
    with self.assertRaises(FailureTransportError) as caught:
      self.manager.forward(inconsistent)
    self.assertEqual(caught.exception.failure_id, report.failure_id)
    self.assertIs(self.manager.snapshot()[0][0], report)
    self.assertEqual(self.emitted, [str(report)])

  def test_materialize_forwarded_report_without_reemission(self) -> None:
    report = self.manager.capture_exception(ValueError('original'))
    assert report is not None
    parent = FailureManager('parent', 'finish')
    with patch.object(parent, '_logger') as logger:
      self.assertIsNone(parent.forward(report))
      with self.assertRaises(ExceptionGroup) as caught:
        parent.raise_if_failed('summary')
      restored, = caught.exception.exceptions
      self.assertIs(type(restored), ValueError)
      self.assertEqual(str(restored), 'original')
      with self.assertRaises(ExceptionGroup) as repeated:
        parent.raise_if_failed('summary')
      self.assertEqual(repeated.exception.exceptions, (restored,))
      self.assertIsNone(parent.capture_exception(restored))
      logger.error.assert_not_called()

  def test_complete_wide_deep_group_diagnostics_and_causes(self) -> None:
    group = ExceptionGroup('wide', [ValueError(f'leaf-{index}')
                                    for index in range(25)])
    for depth in range(15):
      group = ExceptionGroup(f'depth-{depth}', [group])
    cause = RuntimeError('causal explanation')
    group.__cause__ = cause
    report = self.manager.capture_exception(group)
    assert report is not None
    for index in range(25):
      self.assertIn(f'ValueError: leaf-{index}\n', report.formatted_traceback)
    self.assertIn('RuntimeError: causal explanation',
                  report.formatted_traceback)
    self.assertNotIn('... (max_group_depth', report.formatted_traceback)

  def test_broken_string_formatter_does_not_lose_type(self) -> None:
    report = self.manager.capture_exception(_BadMessage())
    assert report is not None
    self.assertEqual(report.nodes[0].type_name, '_BadMessage')
    self.assertIn('message unavailable', report.nodes[0].message)
    self.assertIn('_BadMessage', report.formatted_traceback)

  def test_sink_failure_does_not_erase_the_origin(self) -> None:
    sink_error = OSError('log sink')
    origin = ValueError('origin')
    self.logger.error.side_effect = sink_error
    emergency = StringIO()
    with patch('crappy._failure.sys.stderr', emergency):
      with self.assertRaises(OSError) as caught:
        self.manager.capture_exception(origin)
    self.assertIs(caught.exception, sink_error)
    retained = self.manager.snapshot()
    self.assertIs(retained[0][1], origin)
    self.assertIs(retained[1][1], sink_error)
    self.assertEqual(retained[1][0].phase, 'reporting')
    self.assertIn('ValueError: origin', emergency.getvalue())
    self.assertIn('OSError: log sink', emergency.getvalue())
    self.assertEqual(emergency.getvalue(),
                     ''.join(str(report) for report, _ in retained))
    self.assertIsNone(self.manager.capture_exception(origin))
    self.assertIsNone(self.manager.capture_exception(sink_error))
    self.logger.error.assert_called_once_with(str(retained[0][0]))

  def test_default_dispatch_reports_full_diagnostics_promptly(self) -> None:
    manager = FailureManager('Block-1', 'main', owner_path='Block-1.worker')
    error = ValueError('original')
    try:
      raise error
    except ValueError as caught:
      with self.assertLogs('crappy', level='ERROR') as logged:
        report = manager.capture_exception(caught)
    assert report is not None
    self.assertEqual(len(logged.records), 1)
    message = logged.records[0].getMessage()
    self.assertEqual(message, str(report))
    for text in (report.failure_id, report.block_name,
                 report.owner_path, report.process_name,
                 str(report.pid), report.phase,
                 report.formatted_traceback):
      self.assertIn(text, message)
    with self.assertNoLogs('crappy', level='ERROR'):
      manager.capture_exception(error)
      with self.assertRaises(ExceptionGroup):
        manager.raise_if_failed('summary')

  def test_reporting_options_are_not_accepted(self) -> None:
    constructor = signature(FailureManager)
    for options in ({'on_report': None}, {'ledger': None}):
      with self.subTest(options=options):
        with self.assertRaises(TypeError):
          constructor.bind('Block-1', 'main', **options)

  def test_emergency_sink_failure_retains_independent_errors(self) -> None:
    origin = ValueError('origin')
    sink_error, emergency_error = OSError('logger'), RuntimeError('stderr')
    self.logger.error.side_effect = sink_error
    emergency = Mock()
    emergency.write.side_effect = emergency_error
    with patch('crappy._failure.sys.stderr', emergency):
      with self.assertRaises(ExceptionGroup) as caught:
        self.manager.capture_exception(origin)
    self.assertEqual(caught.exception.exceptions,
                     (sink_error, emergency_error))
    self.assertEqual(tuple(error for _, error in
                           self.manager.snapshot()),
                     (origin, sink_error, emergency_error))
    self.logger.error.assert_called_once()

  def test_forward_does_not_use_a_failing_logger(self) -> None:
    report = self.manager.capture_exception(ValueError('worker'))
    assert report is not None
    parent = FailureManager('parent', 'finish')
    with patch.object(parent, '_logger') as logger:
      logger.error.side_effect = OSError('parent sink')
      with patch('crappy._failure.sys.stderr') as emergency:
        emergency.write.side_effect = RuntimeError('stderr')
        self.assertIsNone(parent.forward(report))
        self.assertIsNone(parent.forward(report))
        with self.assertRaises(ExceptionGroup) as caught:
          parent.raise_if_failed('summary')
        restored, = caught.exception.exceptions
        self.assertIs(type(restored), ValueError)
        self.assertEqual(str(restored), 'worker')
        self.assertIsNone(parent.capture_exception(restored))
        emergency.write.assert_not_called()
        emergency.flush.assert_not_called()
      logger.error.assert_not_called()
    self.assertEqual(parent.snapshot(), ((report, restored),))
    self.assertEqual(self.emitted, [str(report)])

  def test_forward_through_multiple_managers_logs_only_at_origin(self) -> None:
    error = BaseExceptionGroup('worker failures',
                               [KeyboardInterrupt('worker'), OSError('close')])
    report = self.manager.capture_exception(error)
    assert report is not None
    parent = FailureManager('parent', 'finish')
    root = FailureManager('root', 'finish')
    parent_error = ValueError('independent parent failure')
    with patch.object(parent, '_logger') as parent_logger:
      with patch.object(root, '_logger') as root_logger:
        self.assertIsNone(parent.forward(report))
        self.assertIsNone(root.forward(parent.snapshot()[0][0]))
        self.assertIsNone(parent.forward(report))
        self.assertIsNone(root.forward(report))
        parent_report = parent.capture_exception(parent_error)
        assert parent_report is not None
        self.assertIsNone(root.forward(parent_report))
        with self.assertRaises(BaseExceptionGroup) as caught:
          root.raise_if_failed('summary')
        restored, restored_parent = caught.exception.exceptions
        self.assertIs(type(restored), BaseExceptionGroup)
        self.assertEqual(tuple(type(leaf) for leaf in restored.exceptions),
                         (KeyboardInterrupt, OSError))
        self.assertEqual(str(restored_parent), str(parent_error))
        self.assertIsNone(root.capture_exception(restored))
        self.assertIsNone(root.capture_exception(restored_parent))
        root_logger.error.assert_not_called()
      parent_logger.error.assert_called_once_with(str(parent_report))
    self.assertEqual(root.snapshot(), ((report, restored),
                                       (parent_report, restored_parent)))
    self.assertEqual(self.emitted, [str(report)])

  def test_failed_capture_rolls_back_claim(self) -> None:
    error = ValueError('original')
    with patch.object(self.manager, '_format_exception',
                      side_effect=KeyboardInterrupt):
      with self.assertRaises(KeyboardInterrupt):
        self.manager.capture_exception(error)
    self.assertEqual(self.manager.snapshot(), ())
    self.assertIsNotNone(self.manager.capture_exception(error))

  def test_simultaneous_captures_claim_the_same_error_once(self) -> None:
    error = ValueError('same object')
    barrier = Barrier(8)

    def capture(_):
      barrier.wait(timeout=5)
      return self.manager.capture_exception(error)

    with ThreadPoolExecutor(max_workers=8) as executor:
      reports = list(executor.map(capture, range(8)))
    self.assertEqual(sum(report is not None for report in reports), 1)
    self.assertEqual(len(self.emitted), 1)
    self.assertEqual(len(self.manager.snapshot()), 1)

  def test_sink_runs_outside_history_lock(self) -> None:
    origin = ValueError('original')
    manager: FailureManager

    def inspect_from_another_thread(message):
      with ThreadPoolExecutor(max_workers=1) as executor:
        def inspect():
          self.assertIsNone(manager.capture_exception(origin))
          return manager.snapshot()[0][0]
        report = executor.submit(inspect).result(timeout=5)
        self.assertEqual(message, str(report))

    manager = FailureManager('Block-1', 'main', owner_path='Block-1.worker')
    with patch.object(manager, '_logger') as logger:
      logger.error.side_effect = inspect_from_another_thread
      manager.capture_exception(origin)

  def test_snapshot_remains_stable_after_later_capture(self) -> None:
    error = ValueError('original')
    report = self.manager.capture_exception(error)
    assert report is not None
    snapshot = self.manager.snapshot()
    self.manager.capture_exception(OSError('later'))
    self.assertEqual(snapshot, ((report, error),))
    self.assertEqual(len(self.manager.snapshot()), 2)
    self.assertEqual(len(self.emitted), 2)

  def test_new_manager_has_independent_claims_and_history(self) -> None:
    error = ValueError('session')
    self.manager.capture_exception(error)
    manager = FailureManager('Block-1', 'main')
    with patch.object(manager, '_logger') as logger:
      self.assertEqual(manager.snapshot(), ())
      self.assertIsNotNone(manager.capture_exception(error))
      logger.error.assert_called_once()
    self.assertEqual(len(self.emitted), 1)
    self.assertEqual(len(self.manager.snapshot()), 1)
    self.assertFalse(hasattr(manager, 'clear'))

  def test_graph_schema_keeps_local_indexes_not_incident_ids(self) -> None:
    error = ExceptionGroup('group', [ValueError('leaf')])
    report = self.manager.capture_exception(error)
    assert report is not None
    report = replace(report, nodes=(
        ExceptionNode('builtins', 'ExceptionGroup', 'group',
                      tuple(f'{base.__module__}.{base.__qualname__}'
                            for base in ExceptionGroup.__mro__),
                      children=(1,), group_message='group'),
        ExceptionNode('builtins', 'ValueError', 'leaf',
                      tuple(f'{base.__module__}.{base.__qualname__}'
                            for base in ValueError.__mro__), cause=1),
    ))
    self.assertEqual(report.nodes[0].children, (1,))
    self.assertEqual(report.nodes[1].cause, 1)
    with self.assertRaises(FrozenInstanceError):
      setattr(report, 'nodes', ())

  def test_schema_rejects_mutable_or_non_diagnostic_values(self) -> None:
    node = ExceptionNode('builtins', 'ValueError', 'leaf',
                         tuple(f'{base.__module__}.{base.__qualname__}'
                               for base in ValueError.__mro__))
    for field, value in (('notes', ['mutable']), ('children', [0]),
                         ('original_bytes', object()), ('message', object())):
      with self.subTest(field=field):
        with self.assertRaises(TypeError):
          replace(node, **{field: value})
    report = self.manager.capture_exception(ValueError())
    assert report is not None
    with self.assertRaises(ValueError):
      replace(report, nodes=(replace(node, cause=1),))
    for failure_id in ('not-a-uuid', report.failure_id.upper()):
      with self.subTest(failure_id=failure_id):
        with self.assertRaises(ValueError):
          replace(report, failure_id=failure_id)
    with self.assertRaises(TypeError):
      invalid_fields = {'nodes': ValueError('not diagnostic values')}
      replace(report, **invalid_fields)
    with self.assertRaises(ValueError):
      replace(report, pid=True)

  def test_manager_origin_defaults_and_capture_snapshot(self) -> None:
    with patch('crappy._failure.current_process') as process:
      process.return_value.name = 'worker'
      with patch('crappy._failure.getpid', return_value=123):
        manager = FailureManager('camera', 'main')
    with patch.object(manager, '_logger'):
      report = manager.capture_exception(ValueError('origin'))
      assert report is not None
      self.assertEqual(report.owner_path, 'camera')
      self.assertEqual(report.process_name, 'worker')
      self.assertEqual(report.pid, 123)
      manager.phase = 'finish'
      manager.owner_path = 'camera.other'
      later = manager.capture_exception(OSError('later'))
    assert later is not None
    self.assertEqual(report.phase, 'main')
    self.assertEqual(report.owner_path, 'camera')
    self.assertEqual(later.phase, 'finish')
    self.assertEqual(later.owner_path, 'camera.other')

  def test_origin_fields_require_nonempty_labels(self) -> None:
    report = self.manager.capture_exception(ValueError('origin'))
    assert report is not None
    for field in ('block_name', 'owner_path', 'process_name', 'phase'):
      with self.subTest(field=field):
        with self.assertRaises(ValueError):
          replace(report, **{field: ''})
    for arguments in ({'block_name': '', 'phase': 'main'},
                      {'block_name': 'owner', 'phase': ''},
                      {'block_name': 'owner', 'phase': 'main',
                       'owner_path': ''}):
      with self.subTest(arguments=arguments):
        with self.assertRaises(ValueError):
          FailureManager(**arguments)

  def test_report_rejects_incomplete_or_mutable_graphs(self) -> None:
    report = self.manager.capture_exception(ValueError('origin'))
    assert report is not None
    node = report.nodes[0]
    invalid_nodes: tuple[Any, ...] = ((), [node], (ValueError('not a node'),))
    for nodes in invalid_nodes:
      with self.subTest(nodes=nodes):
        with self.assertRaises(TypeError):
          replace(report, nodes=nodes)
    self.assertIs(signature(FailureReport).parameters['nodes'].default,
                  signature(FailureReport).empty)
    for name in ('root', 'exception_module', 'exception_type',
                 'exception_message'):
      self.assertFalse(hasattr(report, name))
      self.assertNotIn(name, report.to_wire())
      with self.assertRaises(TypeError):
        replace(report, **{name: None})

  def test_mro_is_required_immutable_diagnostic_data(self) -> None:
    self.assertIs(signature(ExceptionNode).parameters['mro'].default,
                  signature(ExceptionNode).empty)
    with self.assertRaises(TypeError):
      ExceptionNode('builtins', 'ValueError', 'origin')
    node = ExceptionNode('builtins', 'ValueError', 'origin',
                         tuple(f'{base.__module__}.{base.__qualname__}'
                               for base in ValueError.__mro__))
    invalid_mros: tuple[Any, ...] = (
        list(node.mro), (ValueError,), ('',), ('builtins.ValueError', None))
    for mro in invalid_mros:
      with self.subTest(mro=mro):
        with self.assertRaises(TypeError):
          replace(node, mro=mro)
    with self.assertRaises(FrozenInstanceError):
      setattr(node, 'mro', ())

  def test_mro_requires_the_original_exception_and_python_roots(self) -> None:
    node = ExceptionNode('builtins', 'ValueError', 'origin',
                         tuple(f'{base.__module__}.{base.__qualname__}'
                               for base in ValueError.__mro__))
    for mro in ((), node.mro[1:], node.mro[:-1],
                ('builtins.ValueError', 'builtins.object')):
      with self.subTest(mro=mro):
        with self.assertRaises(ValueError):
          replace(node, mro=mro)

  def test_simultaneous_overlapping_groups_retain_all_leaves(self) -> None:
    shared = ValueError('shared')
    independent = [OSError(f'worker-{index}') for index in range(8)]
    barrier = Barrier(8)

    def capture(index):
      barrier.wait(timeout=5)
      group = ExceptionGroup(f'worker-{index}', [shared, independent[index]])
      return self.manager.capture_exception(group)

    with ThreadPoolExecutor(max_workers=8) as executor:
      list(executor.map(capture, range(8)))
    leaves = []
    for _, error in self.manager.snapshot():
      assert isinstance(error, ExceptionGroup)
      leaves.extend(error.exceptions)
    self.assertEqual(sum(leaf is shared for leaf in leaves), 1)
    for error in independent:
      self.assertEqual(sum(leaf is error for leaf in leaves), 1)
    self.assertEqual(len(self.emitted), 8)

  def test_manager_lifetime_releases_strong_exception_references(self) -> None:
    manager = FailureManager('Block-1', 'main')
    manager._logger = Mock()
    error = _BadMessage()
    reference = ref(error)
    error.__cause__ = error
    manager.capture_exception(error)
    del error
    gc.collect()
    self.assertIsNotNone(reference())
    del manager
    gc.collect()
    self.assertIsNone(reference())
