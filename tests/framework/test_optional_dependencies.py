# coding: utf-8

"""Truthful optional-dependency diagnostics and their reporting transport."""

from multiprocessing import get_all_start_methods, get_context
import pickle
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from crappy._collection import CollectionEntry, load_collection_class
from crappy._exceptions import OptionalDependencyError
from crappy._failure import FailureManager, FailureReport
from crappy._global import OptionalModule
from tests.block.block_test_base import exception_leaves


def failing_import(_):
  """Produces a real traceback and a nested import cause without dependencies."""

  try:
    raise OSError('native loader detail')
  except OSError as error:
    raise ModuleNotFoundError('missing transitive dependency',
                              name='transitive_dependency') from error


def report_import_failures(queue):
  """Sends optional and collection failures from an independent process."""

  manager = FailureManager('dependency owner', 'prepare')
  manager._logger = Mock()
  manager.set_queue(queue)
  with manager.collect('imports', raise_on_exit=False):
    with manager.step('optional dependency'):
      with patch('crappy._global.import_module', side_effect=failing_import):
        OptionalModule('optional_backend', lazy_import=True,
                       install_name='backend-package').feature
    with manager.step('collection dependency'):
      entry = CollectionEntry('Device', 'InOut', 'collection.device')
      with patch('crappy._collection.import_module', side_effect=failing_import):
        load_collection_class(entry, {})
  queue.close()
  queue.join_thread()


class TestOptionalModule(TestCase):
  def test_eager_cause_is_retained_without_importing_again(self):
    try:
      failing_import('backend')
    except ImportError as error:
      cause = error
      module = OptionalModule('backend', 'Needed for preview', cause=error,
                              install_name='backend-package')

    with patch('crappy._global.import_module') as importer:
      with self.assertRaises(OptionalDependencyError) as caught:
        module.feature

    importer.assert_not_called()
    error = caught.exception
    self.assertIsInstance(error, ImportError)
    self.assertIs(error.__cause__, cause)
    self.assertIsNotNone(cause.__traceback__)
    self.assertEqual(error.name, 'backend')
    self.assertEqual(error.import_name, 'backend')
    self.assertEqual(error.install_name, 'backend-package')
    self.assertIn('Needed for preview', str(error))
    self.assertNotIn('not installed', str(error))

  def test_lazy_failure_keeps_missing_transitive_and_abi_causes(self):
    failures = (
      ModuleNotFoundError("No module named 'backend'", name='backend'),
      ModuleNotFoundError("No module named 'child'", name='child'),
      ImportError('undefined native symbol', name='backend', path='native.so'),
    )
    for cause in failures:
      with self.subTest(cause=cause):
        module = OptionalModule('backend', lazy_import=True,
                                install_name='backend-package')
        with patch('crappy._global.import_module', side_effect=cause):
          with self.assertRaises(OptionalDependencyError) as caught:
            module.feature
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIn('could not be loaded', str(caught.exception))
        self.assertIn('backend-package', str(caught.exception))
        self.assertNotIn('Missing module', str(caught.exception))

  def test_successful_lazy_import_is_cached(self):
    module = OptionalModule('backend.submodule', lazy_import=True)
    actual = SimpleNamespace(feature=42)
    with patch('crappy._global.import_module', return_value=actual) as importer:
      self.assertEqual(module.feature, 42)
      self.assertEqual(module.feature, 42)
    importer.assert_called_once_with('backend.submodule')

  def test_failed_lazy_import_can_succeed_on_a_later_attempt(self):
    module = OptionalModule('backend', lazy_import=True)
    actual = SimpleNamespace(feature=42)
    with patch('crappy._global.import_module',
               side_effect=[ImportError('temporary failure'), actual]) as importer:
      with self.assertRaises(OptionalDependencyError):
        module.feature
      self.assertEqual(module.feature, 42)
      self.assertEqual(module.feature, 42)
    self.assertEqual(importer.call_count, 2)
    self.assertIsNone(module._cause)

  def test_legacy_eager_placeholder_does_not_invent_a_cause(self):
    module = OptionalModule('dependency', 'Feature guidance')
    with patch('crappy._global.import_module') as importer:
      with self.assertRaises(OptionalDependencyError) as caught:
        module()
    importer.assert_not_called()
    self.assertIsNone(caught.exception.__cause__)
    self.assertIn('Feature guidance', str(caught.exception))

  def test_lazy_call_uses_the_resolved_callable(self):
    actual = Mock(return_value='result')
    module = OptionalModule('callable_dependency', lazy_import=True)
    with patch('crappy._global.import_module', return_value=actual) as importer:
      self.assertEqual(module(3, enabled=True), 'result')
      self.assertEqual(module(4), 'result')
    importer.assert_called_once_with('callable_dependency')
    self.assertEqual(actual.call_args_list[0].args, (3,))
    self.assertEqual(actual.call_args_list[0].kwargs, {'enabled': True})

  def test_resolved_module_has_normal_non_callable_behavior(self):
    module = OptionalModule('backend', lazy_import=True)
    with patch('crappy._global.import_module', return_value=ModuleType('backend')):
      with self.assertRaises(TypeError) as caught:
        module()
    self.assertNotIsInstance(caught.exception, OptionalDependencyError)

  def test_attribute_error_is_not_relabelled_as_dependency_failure(self):
    module = OptionalModule('backend', lazy_import=True)
    with patch('crappy._global.import_module', return_value=ModuleType('backend')):
      with self.assertRaises(AttributeError):
        module.absent

  def test_callable_failures_propagate_unchanged(self):
    for error in (ValueError('calculation'), RuntimeError('runtime failure'),
                  ImportError('runtime import')):
      with self.subTest(error=error):
        actual = Mock(side_effect=error)
        module = OptionalModule('backend', lazy_import=True)
        with patch('crappy._global.import_module', return_value=actual):
          with self.assertRaises(type(error)) as caught:
            module()
        self.assertIs(caught.exception, error)

  def test_unrelated_initialization_errors_and_interruptions_propagate(self):
    failures = (ValueError('module bug'), KeyboardInterrupt(),
                BaseExceptionGroup('import interrupted',
                                   [ValueError('module bug'), SystemExit(2)]))
    for error in failures:
      with self.subTest(error=error):
        module = OptionalModule('backend', lazy_import=True)
        with patch('crappy._global.import_module', side_effect=error):
          with self.assertRaises(type(error)) as caught:
            module.feature
        self.assertIs(caught.exception, error)

  def test_dependency_error_constructor_metadata_round_trips(self):
    error = OptionalDependencyError('cv2', 'opencv-python', 'Preview requires it')
    error.add_note('original note')
    restored = pickle.loads(pickle.dumps(error))
    self.assertIs(type(restored), OptionalDependencyError)
    self.assertEqual(restored.import_name, 'cv2')
    self.assertEqual(restored.install_name, 'opencv-python')
    self.assertEqual(restored.name, 'cv2')
    self.assertEqual(str(restored), str(error))
    self.assertEqual(restored.__notes__, ['original note'])

  def test_eager_framework_sites_retain_a_native_import_failure(self):
    script = textwrap.dedent('''
      import builtins
      import traceback
      original = builtins.__import__

      def blocked_import(name, *args, **kwargs):
        if name == 'cv2' or name.startswith('cv2.'):
          raise ImportError('synthetic ABI mismatch', name='cv2', path='cv2.so')
        return original(name, *args, **kwargs)

      builtins.__import__ = blocked_import
      import crappy
      from crappy._exceptions import OptionalDependencyError
      from crappy.blocks.camera_processes import display, record
      from crappy.tool.image_processing import fields

      for proxy in (crappy.resources.speckle, crappy.resources.ve_markers,
                    crappy.resources.pad, display.cv2, record.cv2, fields.cv2):
        try:
          proxy.unavailable_feature
        except OptionalDependencyError as error:
          assert error.import_name == 'cv2', error
          assert error.install_name == 'opencv-python', error
          assert type(error.__cause__) is ImportError, error
          assert error.__cause__.path == 'cv2.so', error
          assert 'blocked_import' in ''.join(traceback.format_exception(error))
          assert 'not installed' not in str(error), error
        else:
          raise AssertionError('Unavailable dependency was silently accepted')
    ''')
    result = subprocess.run([sys.executable, '-c', script], text=True,
                            capture_output=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_import_causes_and_tracebacks_cross_process_boundaries(self):
    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = (
            'from tests.framework.test_optional_dependencies '
            'import TestOptionalModule; '
            f'TestOptionalModule()._check_import_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script], text=True,
                                capture_output=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def _check_import_context(self, method):
    context = get_context(method)
    queue = context.Queue()
    process = context.Process(target=report_import_failures, args=(queue,))
    parent = FailureManager('parent', 'prepare')
    parent._logger = Mock()
    try:
      process.start()
      for _ in range(2):
        report = FailureReport.from_wire(
            pickle.loads(queue.get(timeout=10)))
        self.assertIn('failing_import', report.formatted_traceback)
        parent.forward(report)
      process.join(5)
      self.assertEqual(process.exitcode, 0)
      with self.assertRaises(ExceptionGroup) as caught:
        parent.raise_if_failed('Dependency failures')
      optional, collection = exception_leaves(caught.exception)
      self.assertIs(type(optional), OptionalDependencyError)
      self.assertEqual(optional.install_name, 'backend-package')
      self.assertEqual(collection.entry.module, 'collection.device')
      for error in (optional, collection):
        self.assertIs(type(error.__cause__), ModuleNotFoundError)
        self.assertEqual(error.__cause__.name, 'transitive_dependency')
        self.assertIs(type(error.__cause__.__cause__), OSError)
        self.assertTrue(any('failing_import' in note
                            for note in error.__cause__.__notes__))
      self.assertIs(collection.cause, collection.__cause__)
      parent._logger.error.assert_not_called()
    finally:
      if process.pid is not None:
        if process.is_alive():
          process.terminate()
          process.join(2)
        if process.is_alive():
          process.kill()
          process.join(2)
      process.close()
      queue.close()
      queue.join_thread()
