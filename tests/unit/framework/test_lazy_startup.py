"""Selection completes before spawning; process-owned setup stays in workers."""

from tests.support.processes import run_python
import importlib
import logging
import multiprocessing as mp
import os
import pickle
import textwrap
import warnings
from collections.abc import Mapping
from unittest import TestCase
from unittest.mock import Mock, patch

from crappy import Block, img_link
from crappy.blocks import AutoDriveVideoExtenso, Camera, IOBlock, Machine
from crappy.blocks.meta_block import block as block_module
from crappy.blocks.vision import CameraSource
from crappy.blocks.vision.block import ConfigRequest
from crappy._exceptions import OptionalDependencyError
from crappy._global import OptionalModule
from crappy.tool.camera_config import resolve_configurator
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.assertions import exception_leaves
from tests.unit.framework.lazy_startup_fixtures import StartupActuator, StartupCamera, StartupConfig, StartupConsumer, StartupInOut, StartupProcess


class RecordingChoices(Mapping):
  def __init__(self, selected=StartupConfig, error=None):
    self.selected = selected
    self.error = error
    self.lookups = []

  def __getitem__(self, key):
    self.lookups.append(key)
    if self.error is not None:
      raise self.error
    return self.selected

  def __iter__(self):
    return iter(('pyqt', 'tkinter'))

  def __len__(self):
    return 2

  def __contains__(self, key):
    return key in ('pyqt', 'tkinter')


def image_generator(exx, eyy):
  return None  # Resolution must not acquire even a synthetic image.


class TestParentImportResolution(BlockTestBase):
  def setUp(self):
    super().setUp()
    warnings_filter = warnings.catch_warnings()
    warnings_filter.__enter__()
    self.addCleanup(warnings_filter.__exit__, None, None, None)
    warnings.simplefilter('ignore', UserWarning)

  @staticmethod
  def source(**kwargs):
    return CameraSource('', image_generator=image_generator, **kwargs)

  @staticmethod
  def request(source, choices, required=True):
    request = ConfigRequest('consumer', (), {}, choices, source.name,
                             required=required)
    source.config_requests_in.append(request)
    return request

  def test_factory_resolution_returns_real_classes_without_construction(self):
    choices = RecordingChoices()
    self.assertIs(resolve_configurator(choices, 'tkinter'), StartupConfig)
    self.assertEqual(choices.lookups, ['tkinter'])
    self.assertIs(resolve_configurator(StartupConfig, 'unused'), StartupConfig)
    with self.assertRaises(KeyError):
      resolve_configurator({}, 'pyqt')

  def test_enabled_defaults_resolve_only_selected_backend_and_stay_picklable(self):
    for cls in (Camera, CameraSource):
      for backend in ('pyqt', 'tkinter'):
        with self.subTest(cls=cls, backend=backend):
          choices = RecordingChoices()
          with patch.object(cls, 'configurator', choices):
            block = cls('', image_generator=image_generator,
                        config_backend=backend)
            if isinstance(block, CameraSource):
              self.assertEqual(choices.lookups, [])
              block.resolve_config_requests()
          self.assertIs(block.configurator, StartupConfig)
          self.assertEqual(choices.lookups, [backend])
          self.assertIs(pickle.loads(pickle.dumps(block.configurator)),
                        StartupConfig)
          self.assertIsNone(block._camera)
          self.assertIsNone(block._camera_class)

  def test_requested_configuration_uses_source_backend_and_skips_default(self):
    source = self.source(config_backend='tkinter')
    unused = RecordingChoices(error=AssertionError('Unused default'))
    source.configurator = unused
    choices = RecordingChoices()
    request = self.request(source, choices)
    source.resolve_config_requests()
    self.assertIs(request.configurator, StartupConfig)
    self.assertEqual(choices.lookups, ['tkinter'])
    self.assertEqual(unused.lookups, [])
    self.assertIs(source.configurator, unused)

  def test_disabled_configuration_does_not_index_any_choices(self):
    for cls in (Camera, CameraSource):
      choices = RecordingChoices(error=AssertionError('Disabled GUI'))
      with patch.object(cls, 'configurator', choices):
        block = cls('', image_generator=image_generator, config=False)
        if isinstance(block, CameraSource):
          self.request(block, choices, required=False)
          block.resolve_config_requests()
        self.assertIs(block.configurator, choices)
        self.assertEqual(choices.lookups, [])

  def test_disallowed_optional_requests_resolve_only_default(self):
    source = self.source(allow_downstream_config=False)
    default = RecordingChoices()
    source.configurator = default
    unused = RecordingChoices(error=AssertionError('Declined request'))
    request = self.request(source, unused, required=False)
    source.resolve_config_requests()
    self.assertEqual(default.lookups, ['pyqt'])
    self.assertEqual(unused.lookups, [])
    self.assertIs(request.configurator, unused)

  def test_unservable_required_requests_fail_without_resolving_a_window(self):
    for kwargs in ({'config': False}, {'allow_downstream_config': False}):
      with self.subTest(kwargs=kwargs):
        source = self.source(**kwargs)
        choices = RecordingChoices(error=AssertionError('Unusable window'))
        source.configurator = choices
        self.request(source, choices)
        with self.assertRaises(RuntimeError):
          source.resolve_config_requests()
        self.assertEqual(choices.lookups, [])

  def test_invalid_source_configurators_fail_during_resolution(self):
    for choices, error in (({}, ValueError), ({'pyqt': object}, TypeError)):
      source = self.source()
      source.configurator = choices
      with self.assertRaises(error):
        source.resolve_config_requests()

  def test_constructors_retain_real_picklable_drivers_without_opening(self):
    events = Mock()
    blocks = ((Machine([{'type': 'StartupActuator', 'events': events}]),
               '_actuator_classes', (StartupActuator,)),
              (AutoDriveVideoExtenso({'type': 'StartupActuator', 'events': events}),
               '_actuator_class', StartupActuator),
              (IOBlock('StartupInOut', events=events), '_inout_class', StartupInOut),
              (Camera('StartupCamera', config=False, events=events),
               '_camera_class', StartupCamera),
              (CameraSource('StartupCamera', config=False, events=events),
               '_camera_class', StartupCamera))
    for block, attribute, expected in blocks:
      with self.subTest(block=type(block).__name__):
        self.assertEqual(getattr(block, attribute), expected)
        self.assertEqual(pickle.loads(pickle.dumps(getattr(block, attribute))),
                         expected)
    events.put.assert_not_called()

  def test_scalar_driver_preparation_reuses_parent_selection(self):
    cases = ((Machine([{'type': 'StartupActuator', 'events': Mock()}]),
              'machine', 'resolve_actuator'),
             (AutoDriveVideoExtenso({'type': 'StartupActuator', 'events': Mock()}),
              'auto_drive_video_extenso', 'resolve_actuator'),
             (IOBlock('StartupInOut', cmd_labels=['cmd'], events=Mock()),
              'ioblock', 'resolve_inout'))
    for block, module_name, resolver in cases:
      block.inputs.append(Mock())
      with patch('crappy.blocks.' + module_name + '.' + resolver,
                 side_effect=AssertionError('Driver reselected in worker')):
        block.prepare()

  def test_resolution_sees_assembled_requests_and_precedes_manager_creation(self):
    source = self.source()
    consumer = StartupConsumer(Mock())
    img_link(source, consumer)

    def reached_manager():
      self.assertEqual(len(source.config_requests_in), 1)
      self.assertIs(source.config_requests_in[0].configurator, StartupConfig)
      raise RuntimeError('Stop before creating a process')

    with (patch.object(block_module, 'Manager', side_effect=reached_manager) as manager,
          patch.object(Block, 'start') as start):
      with self.assertRaises(ExceptionGroup):
        Block.prepare_all(log_level=logging.CRITICAL)
      manager.assert_called_once_with()
      start.assert_not_called()
    self.assertTrue(source.config_requests_in[0].connection.closed)
    self.assertTrue(consumer._config_requests_out[0].connection.closed)

  def test_resolution_failure_preserves_cause_and_cleans_unstarted_graph(self):
    source = self.source()
    consumer = StartupConsumer(Mock())
    missing = ImportError('selected backend missing')
    dependency = OptionalModule('missing_config_backend', cause=missing)

    class MissingChoices(RecordingChoices):
      def __getitem__(self, key):
        return dependency.resolve()

    consumer.configurator = MissingChoices()
    img_link(source, consumer)
    with (patch.object(block_module, 'Manager') as manager,
          patch.object(Block, 'start') as start):
      with self.assertRaises(ExceptionGroup) as caught:
        Block.prepare_all(log_level=logging.CRITICAL)
      manager.assert_not_called()
      start.assert_not_called()
    errors = exception_leaves(caught.exception)
    self.assertEqual(len(errors), 1)
    self.assertIsInstance(errors[0], OptionalDependencyError)
    self.assertIs(errors[0].__cause__, missing)
    self.assertIsNone(source.pid)
    self.assertIsNone(consumer.pid)
    self.assertIsNone(Block.shared_mgr)
    self.assertFalse(Block.prepared_all)
    self.assertTrue(source.config_requests_in[0].connection.closed)
    self.assertTrue(consumer._config_requests_out[0].connection.closed)

  def test_only_sources_finalize_and_see_all_their_downstream_requests(self):
    source = self.source()
    source.configurator = RecordingChoices(error=AssertionError('Unused default'))
    consumers = [StartupConsumer(Mock()), StartupConsumer(Mock())]
    for consumer in consumers:
      consumer.resolve_config_requests = Mock(
          side_effect=AssertionError('Consumer finalized as an image source'))
      img_link(source, consumer)
    resolve = source.resolve_config_requests

    def finalize():
      self.assertEqual(len(source.config_requests_in), 2)
      resolve()

    with (patch.object(source, 'resolve_config_requests', side_effect=finalize) as resolved,
          patch.object(block_module, 'Manager', side_effect=RuntimeError('test boundary'))):
      with self.assertRaises(ExceptionGroup):
        Block.prepare_all(log_level=logging.CRITICAL)
    resolved.assert_called_once_with()
    for consumer in consumers:
      consumer.resolve_config_requests.assert_not_called()
      self.assertTrue(consumer._config_requests_out[0].connection.closed)
    for request in source.config_requests_in:
      self.assertIs(request.configurator, StartupConfig)
      self.assertTrue(request.connection.closed)

  def test_shared_consumer_requests_use_each_sources_backend(self):
    sources = [self.source(config_backend=backend)
               for backend in ('pyqt', 'tkinter')]
    consumer = StartupConsumer(Mock())
    choices = RecordingChoices()
    consumer.configurator = choices
    for source in sources:
      source.configurator = RecordingChoices(error=AssertionError('Unused default'))
      img_link(source, consumer)

    def reached_manager():
      self.assertEqual(choices.lookups, ['pyqt', 'tkinter'])
      for source in sources:
        self.assertIs(source.config_requests_in[0].configurator, StartupConfig)
      raise RuntimeError('test boundary')

    with patch.object(block_module, 'Manager', side_effect=reached_manager) as manager:
      with self.assertRaises(ExceptionGroup):
        Block.prepare_all(log_level=logging.CRITICAL)
    manager.assert_called_once_with()
    self.assertTrue(all(request.connection.closed
                        for request in consumer._config_requests_out))

  def test_parent_resolution_preserves_initialization_errors_and_interrupts(self):
    for failure in (RuntimeError('backend initialization failed'),
                    KeyboardInterrupt(), SystemExit(3)):
      with self.subTest(failure=type(failure).__name__):
        source = self.source()
        consumer = StartupConsumer(Mock())
        img_link(source, consumer)
        source.resolve_config_requests = Mock(side_effect=failure)
        with (patch.object(block_module, 'Manager') as manager,
              patch.object(Block, 'start') as start):
          with self.assertRaises(BaseExceptionGroup) as caught:
            Block.prepare_all(log_level=logging.CRITICAL)
          manager.assert_not_called()
          start.assert_not_called()
        self.assertIs(exception_leaves(caught.exception)[0], failure)
        self.assertIsNone(source.pid)
        self.assertIsNone(consumer.pid)
        self.assertFalse(Block.prepared_all)

  def test_plain_blocks_have_no_import_or_configuration_preparation_hook(self):
    block = TestBlock()
    self.assertFalse(hasattr(block, '_resolve_imports'))
    self.assertFalse(hasattr(block, 'resolve_config_requests'))
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(block.prepared.wait(3))
    Block.launch_all()
    self.assertTrue(block.looped.is_set())

  def test_legacy_configurator_failures_are_reported_at_construction(self):
    for failure in (ImportError('missing backend'), RuntimeError('backend bug'),
                    KeyboardInterrupt(), SystemExit(3)):
      with self.subTest(failure=type(failure).__name__):
        choices = RecordingChoices(error=failure)
        with (patch.object(Camera, 'configurator', choices),
              patch.object(block_module, 'Manager') as manager,
              patch.object(Block, 'start') as start):
          with self.assertRaises(type(failure)) as caught:
            Camera('', image_generator=image_generator)
          self.assertIs(caught.exception, failure)
          self.assertEqual(choices.lookups, ['pyqt'])
          manager.assert_not_called()
          start.assert_not_called()


class TestWorkerImportResolution(TestCase):
  def test_camera_pipelines_under_all_supported_start_methods(self):
    for method in mp.get_all_start_methods():
      for legacy in (False, True):
        for collection in (False, True):
          with self.subTest(method=method, legacy=legacy, collection=collection):
            script = ('from tests.unit.framework.lazy_startup_fixtures import run_pipeline; '
                      f'run_pipeline({method!r}, {legacy!r}, {collection!r})')
            result = run_python(['-B', '-c', script], timeout=45, env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_scalar_driver_owners_under_all_supported_start_methods(self):
    for method in mp.get_all_start_methods():
      for collection in (False, True):
        with self.subTest(method=method, collection=collection):
          script = ('from tests.unit.framework.lazy_startup_fixtures import run_scalar_owners; '
                    f'run_scalar_owners({method!r}, {collection!r})')
          result = run_python(['-B', '-c', script], timeout=45)
          self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_disabled_sources_do_not_resolve_configuration_dependencies(self):
    script = '''
      import importlib.abc
      import sys
      import warnings
      class RejectConfiguration(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.startswith(('crappy.tool.camera_config.base',
                                  'crappy.tool.camera_config.pyqt',
                                  'crappy.tool.camera_config.tkinter')):
            raise AssertionError('Disabled configuration import: ' + fullname)
      sys.meta_path.insert(0, RejectConfiguration())
      warnings.simplefilter('ignore')
      from crappy.blocks import Camera
      from crappy.blocks.vision import CameraSource
      for cls in (Camera, CameraSource):
        block = cls('', config=False, image_generator=lambda *_: None)
        if isinstance(block, CameraSource):
          block.resolve_config_requests()
    '''
    result = run_python(['-B', '-c', textwrap.dedent(script)], timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_nested_backend_failure_aborts_before_readiness_or_frame_receive(self):
    process = StartupProcess(Mock())
    process.backend_name = 'missing_nested_processing_backend'
    process._failure_queue = Mock()
    process._block_name = 'test'
    process._log_level = None
    process._set_logger = Mock()
    process._logger = Mock()
    process._cam_barrier = Mock()
    process._stop_event = mp.Event()
    process._get_data = Mock()
    process.run()
    process._cam_barrier.wait.assert_not_called()
    process._cam_barrier.abort.assert_called_once_with()
    process._get_data.assert_not_called()
    self.assertEqual(process._failure_manager.snapshot()[0][0].phase, 'init')
    self.assertTrue(process._stop_event.is_set())

  def test_enabled_builtin_configuration_imports_only_selected_toolkit(self):
    for backend, other in (('pyqt', 'tkinter'), ('tkinter', 'pyqt')):
      with self.subTest(backend=backend):
        script = f'''
          import importlib.abc
          import multiprocessing as mp
          import pickle
          import sys
          import warnings
          class RejectUnusedToolkit(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
              forbidden = ('tkinter', '_tkinter') if '{other}' == 'tkinter' else ('PyQt6',)
              if fullname.split('.')[0] in forbidden:
                raise AssertionError('Unused toolkit: ' + fullname)
          sys.meta_path.insert(0, RejectUnusedToolkit())
          warnings.simplefilter('ignore')
          from crappy.blocks import Camera
          from crappy.blocks.vision import CameraSource
          from tests.support.dependencies import optional_import
          required = ('PyQt6.QtWidgets',) if '{backend}' == 'pyqt' else ('_tkinter', 'tkinter')
          if any(optional_import(name) is None for name in required):
            raise SystemExit(77)
          for cls in (Camera, CameraSource):
            block = cls('', image_generator=lambda *_: None,
                        config_backend='{backend}')
            if isinstance(block, CameraSource):
              block.resolve_config_requests()
            assert isinstance(block.configurator, type)
            assert pickle.loads(pickle.dumps(block.configurator)) is block.configurator
            assert block._camera is None
          assert not mp.active_children()
          if '{backend}' == 'pyqt':
            from PyQt6.QtWidgets import QApplication
            assert QApplication.instance() is None
          else:
            import tkinter
            assert tkinter._default_root is None
        '''
        result = run_python(['-B', '-c', textwrap.dedent(script)], timeout=25, env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        if result.returncode == 77:
          self.skipTest('The selected optional configuration toolkit is unavailable')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_tracker_missing_dependency_fails_before_receiving_frames(self):
    module = importlib.import_module('crappy.tool.image_processing.video_extenso.tracker')
    for name in ('cv2', 'threshold_otsu'):
      with self.subTest(name=name):
        pipe = Mock()
        tracker = module.Tracker(pipe, 'test', None, None,
                                 failure_queue=Mock(), stop_event=mp.Event(),
                                 block_name='test')
        missing = ImportError('missing ' + name)
        dependency = OptionalModule(name, cause=missing)
        # Isolate the missing dependency even when its peers are not installed.
        with (patch.object(module, 'cv2', Mock()),
              patch.object(module, 'threshold_otsu', Mock()),
              patch.object(module, name, dependency)):
          tracker.run()
        pipe.poll.assert_not_called()
        pipe.close.assert_called_once_with()
        self.assertEqual(tracker._failure_manager.snapshot()[0][0].phase, 'init')
        with self.assertRaises(ExceptionGroup) as caught:
          tracker._failure_manager.raise_if_failed('tracker failure')
        error, = exception_leaves(caught.exception)
        self.assertIsInstance(error, OptionalDependencyError)
        self.assertIs(error.__cause__, missing)

  def test_grapher_sets_binding_before_import_and_restores_opencv_plugin_path(self):
    module = importlib.import_module('crappy.blocks.grapher')
    from tests.support.assertions import with_failure_manager
    graph = with_failure_manager(module.Grapher(('t', 'x'), backend='PyQt6'))
    graph.inputs.append(Mock())
    backend = Mock()
    screen = backend.mkQApp.return_value.primaryScreen.return_value
    screen.logicalDotsPerInchX.return_value = 96
    screen.logicalDotsPerInchY.return_value = 96
    original = os.pathsep.join(('/custom/plugins', '/wheel/cv2/qt/plugins'))

    def load(name):
      self.assertEqual(name, 'pyqtgraph')
      self.assertEqual(os.environ['PYQTGRAPH_QT_LIB'], 'PyQt6')
      self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], '/custom/plugins')
      return backend

    with (patch.object(module, 'pg', OptionalModule('pyqtgraph', lazy_import=True)),
          patch('crappy._global.import_module', side_effect=load) as imports,
          patch.dict(os.environ, {'QT_QPA_PLATFORM_PLUGIN_PATH': original})):
      graph.prepare()
      imports.assert_called_once_with('pyqtgraph')
      self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], original)
      graph.finish()
    Block.reset()
