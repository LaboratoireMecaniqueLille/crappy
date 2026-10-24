# coding: utf-8

"""Lazy exports and backend selection without constructing GUI windows."""

from collections.abc import Mapping
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

from crappy._failure import FailureManager
from crappy.tool.camera_config import factory


class TestLazyConfiguratorMapping(unittest.TestCase):
  @staticmethod
  def choices():
    return factory.LazyConfiguratorMapping({
      'tkinter': ('.tkinter', 'TkinterCameraConfig'),
      'pyqt': ('.pyqt', 'PyQtCameraConfig')})

  def test_listing_membership_and_missing_keys_do_not_import(self):
    choices = self.choices()
    with patch.object(factory, 'import_module') as imports:
      self.assertIsInstance(choices, Mapping)
      self.assertEqual(list(choices), ['tkinter', 'pyqt'])
      self.assertEqual(list(choices.keys()), ['tkinter', 'pyqt'])
      self.assertEqual(len(choices), 2)
      self.assertIn('pyqt', choices)
      self.assertNotIn('invalid', choices)
      with self.assertRaises(KeyError) as caught:
        choices['invalid']
      self.assertEqual(caught.exception.args, ('invalid',))
      imports.assert_not_called()

  def test_indexing_loads_and_caches_only_the_selected_real_class(self):
    choices = self.choices()
    selected = type('SelectedConfig', (), {})
    module = SimpleNamespace(PyQtCameraConfig=selected)
    with patch.object(factory, 'import_module', return_value=module) as imports:
      self.assertIs(choices['pyqt'], selected)
      self.assertIs(choices['pyqt'], selected)
      self.assertIs(choices.get('pyqt'), selected)
      imports.assert_called_once_with('.pyqt', factory.__package__)
    self.assertEqual(choices._classes, {'pyqt': selected})

  def test_failed_resolution_is_not_cached_or_relabelled(self):
    failures = (ImportError('native loader failure'), AttributeError('export'),
                RuntimeError('module bug'), KeyboardInterrupt(),
                BaseExceptionGroup('interrupted', [ValueError(), SystemExit(2)]))
    for error in failures:
      with self.subTest(error=type(error).__name__):
        choices = self.choices()
        with patch.object(factory, 'import_module', side_effect=error):
          with self.assertRaises(type(error)) as caught:
            choices['pyqt']
        self.assertIs(caught.exception, error)
        self.assertEqual(choices._classes, {})
        selected = type('RecoveredConfig', (), {})
        with patch.object(factory, 'import_module', return_value=SimpleNamespace(
            PyQtCameraConfig=selected)):
          self.assertIs(choices['pyqt'], selected)

    choices = self.choices()
    with patch.object(factory, 'import_module', return_value=SimpleNamespace()):
      with self.assertRaises(AttributeError):
        choices['pyqt']
    self.assertEqual(choices._classes, {})

  def test_factory_preserves_custom_classes_mappings_and_forwarding(self):
    class CustomConfig:
      def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    manager = FailureManager('test configuration', 'configuration')
    arguments = (sentinel.camera, sentinel.queue, 20, 30, sentinel.transform,
                 manager, sentinel.extra)
    unused = Mock(side_effect=AssertionError('Wrong backend was selected'))
    for choices in (CustomConfig,
                    {'pyqt': CustomConfig, 'tkinter': unused}, self.choices()):
      with self.subTest(choices=type(choices).__name__):
        with patch.object(factory, 'import_module', return_value=SimpleNamespace(
            PyQtCameraConfig=CustomConfig)) as imports:
          result = factory.create_configurator(
            choices, arguments[0], 'pyqt', *arguments[1:], option=sentinel.option)
        self.assertIs(type(result), CustomConfig)
        self.assertEqual(result.args, arguments)
        self.assertEqual(result.kwargs, {'option': sentinel.option})
        self.assertEqual(imports.call_count,
                         int(isinstance(choices, factory.LazyConfiguratorMapping)))
    unused.assert_not_called()

    result = factory.create_configurator(
      CustomConfig, arguments[0], 'ignored', *arguments[1:])
    self.assertEqual(result.args, arguments)

  def test_pickling_unresolved_choices_does_not_import_implementations(self):
    choices = self.choices()
    with patch.object(factory, 'import_module') as imports:
      restored = pickle.loads(pickle.dumps(choices))
      imports.assert_not_called()
    self.assertEqual(restored._references, choices._references)
    self.assertEqual(restored._classes, {})


class TestLazyConfigurationImports(unittest.TestCase):
  def run_script(self, script):
    result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(script)],
                            capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_root_and_geometry_access_do_not_import_configuration_features(self):
    self.run_script('''
      import importlib.abc
      import pickle
      import sys

      forbidden = ('crappy.tool.camera_config.base',
                   'crappy.tool.camera_config.pyqt',
                   'crappy.tool.camera_config.tkinter',
                   'crappy.tool.camera_config.config_tools.spots_detector',
                   'crappy.tool.camera_config.config_tools.histogram_process')
      class RejectConfiguration(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if any(fullname == name or fullname.startswith(name + '.')
                 for name in forbidden):
            raise AssertionError('Premature configuration import: ' + fullname)
      sys.meta_path.insert(0, RejectConfiguration())

      import crappy
      from crappy.tool.camera_config import Box, Overlay, SpotsBoxes, Zoom
      from crappy.tool.camera_config import config_tools, factory
      from crappy.blocks.vision.block import ConfigRequest

      assert Box is config_tools.Box
      assert Overlay is config_tools.Overlay
      assert SpotsBoxes is config_tools.SpotsBoxes
      assert Zoom is config_tools.Zoom
      box = Box(1, 2, 3, 4)
      assert pickle.loads(pickle.dumps(box)).sorted() == box.sorted()
      assert pickle.loads(pickle.dumps(SpotsBoxes(box))).spot_1 == box
      assert pickle.loads(pickle.dumps(Zoom())) == Zoom()
      assert 'CameraConfig' not in vars(factory)
      assert 'CameraConfig' not in vars(sys.modules[ConfigRequest.__module__])

      classes = (crappy.blocks.Camera, crappy.blocks.DICVE,
                 crappy.blocks.DISCorrel, crappy.blocks.VideoExtenso,
                 crappy.blocks.vision.CameraSource,
                 crappy.blocks.vision.DICVEProcessor,
                 crappy.blocks.vision.DISCorrelProcessor,
                 crappy.blocks.vision.VideoExtensoProcessor)
      for cls in classes:
        choices = cls.configurator
        assert list(choices) == ['tkinter', 'pyqt'], cls
        assert len(choices) == 2
        assert 'pyqt' in choices and 'invalid' not in choices
        assert choices._classes == {}, cls
      assert not any(name.startswith(forbidden) for name in sys.modules)
    ''')

  def test_base_exports_do_not_import_gui_or_spot_detection(self):
    self.run_script('''
      import importlib.abc
      import sys

      forbidden = ('crappy.tool.camera_config.pyqt',
                   'crappy.tool.camera_config.tkinter',
                   'crappy.tool.camera_config.config_tools.spots_detector',
                   'crappy.tool.camera_config.config_tools.histogram_process')
      class RejectGUI(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.startswith(forbidden):
            raise AssertionError('Unrelated implementation imported: ' + fullname)
      sys.meta_path.insert(0, RejectGUI())
      from crappy.tool.camera_config import base
      for name in base.__all__:
        assert isinstance(getattr(base, name), type), name
      assert not any(name.startswith(forbidden) for name in sys.modules)
    ''')

  def test_generic_selection_does_not_load_another_backend_or_family(self):
    for backend, prefix in (('pyqt', 'PyQt'), ('tkinter', 'Tkinter')):
      with self.subTest(backend=backend):
        self.run_script(f'''
          import importlib.abc
          import pickle
          import sys
          from unittest.mock import patch

          opposite = {'tkinter' if backend == 'pyqt' else 'pyqt'!r}
          class RejectUnrelated(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
              if (fullname.startswith('crappy.tool.camera_config.' + opposite) or
                  (fullname.startswith('crappy.tool.camera_config.') and
                   fullname.rsplit('.', 1)[-1] in
                   ('dic_ve_config', 'dis_correl_config', 'video_extenso_config',
                    'spots_detector'))):
                raise AssertionError('Unrelated configurator import: ' + fullname)
          sys.meta_path.insert(0, RejectUnrelated())

          from crappy.blocks.vision import CameraSource
          from crappy.tool.camera_config import factory
          choices = CameraSource.configurator
          with patch.object(factory, 'import_module',
                            wraps=factory.import_module) as imports:
            selected = choices[{backend!r}]
            assert selected.__name__ == {prefix + 'CameraConfig'!r}
            assert choices[{backend!r}] is selected
            assert imports.call_count == 1
          assert isinstance(selected, type)
          assert set(choices._classes) == {{{backend!r}}}
          restored = pickle.loads(pickle.dumps(choices))
          assert restored[{backend!r}] is selected
          assert set(restored._classes) == {{{backend!r}}}
        ''')

  def test_all_builtin_mappings_select_the_expected_export(self):
    for backend, prefix in (('pyqt', 'PyQt'), ('tkinter', 'Tkinter')):
      with self.subTest(backend=backend):
        self.run_script(f'''
          import importlib
          import sys
          import crappy
          selected_package = importlib.import_module(
            'crappy.tool.camera_config.' + {backend!r})
          cases = (
            (crappy.blocks.Camera, 'CameraConfig'),
            (crappy.blocks.DICVE, 'DICVEConfig'),
            (crappy.blocks.DISCorrel, 'DISCorrelConfig'),
            (crappy.blocks.VideoExtenso, 'VideoExtensoConfig'),
            (crappy.blocks.vision.CameraSource, 'CameraConfig'),
            (crappy.blocks.vision.DICVEProcessor, 'DICVEConfig'),
            (crappy.blocks.vision.DISCorrelProcessor, 'DISCorrelConfig'),
            (crappy.blocks.vision.VideoExtensoProcessor, 'VideoExtensoConfig'))
          for block, family in cases:
            selected = block.configurator[{backend!r}]
            assert selected is getattr(selected_package, {prefix!r} + family)
            assert set(block.configurator._classes) == {{{backend!r}}}
          opposite = 'crappy.tool.camera_config.' + (
            'tkinter' if {backend!r} == 'pyqt' else 'pyqt')
          assert not any(name.startswith(opposite) for name in sys.modules)
        ''')

  def test_export_inventory_discovery_wildcards_and_identity(self):
    self.run_script('''
      import importlib
      from unittest.mock import patch
      import crappy.tool.camera_config as root

      inventories = {
        '': ('CameraConfig', 'Box', 'Overlay', 'SpotsBoxes', 'SpotsDetector',
             'Zoom', 'TkinterCameraConfig', 'TkinterCameraConfigBoxes',
             'TkinterDICVEConfig', 'TkinterDISCorrelConfig',
             'TkinterVideoExtensoConfig', 'create_configurator',
             'resolve_configurator'),
        '.base': ('ConfigurationLifecycle', 'CameraConfig', 'ConfigAction',
                  'CameraConfigBoxes', 'DICVEConfig', 'DISCorrelConfig',
                  'VideoExtensoConfig'),
        '.config_tools': ('Box', 'HistogramProcess', 'Overlay', 'SpotsBoxes',
                          'SpotsDetector', 'Zoom'),
        '.tkinter': ('TkinterCameraConfig', 'TkinterCameraConfigBoxes',
                     'TkinterDICVEConfig', 'TkinterDISCorrelConfig',
                     'TkinterVideoExtensoConfig'),
        '.pyqt': ('PyQtCameraConfig', 'PyQtCameraConfigBoxes', 'PyQtDICVEConfig',
                  'PyQtDISCorrelConfig', 'PyQtVideoExtensoConfig'),
      }
      for suffix, expected in inventories.items():
        module = importlib.import_module(root.__name__ + suffix)
        with patch.object(module, '_import_module',
                          wraps=module._import_module) as imports:
          assert set(module.__all__) == set(expected), suffix
          assert set(expected) <= set(dir(module)), suffix
          try:
            getattr(module, 'unknown_configuration_export')
          except AttributeError as error:
            assert module.__name__ in str(error), error
          else:
            raise AssertionError('Unknown export was accepted')
          imports.assert_not_called()

        for name in expected:
          initially_cached = name in vars(module)
          with patch.object(module, '_import_module',
                            wraps=module._import_module) as imports:
            value = getattr(module, name)
            assert getattr(module, name) is value
            assert vars(module)[name] is value
            assert imports.call_count == int(not initially_cached), name
          reference, attribute = module._EXPORTS[name]
          assert value is getattr(importlib.import_module(reference, module.__name__),
                                  attribute)
        namespace = {}
        exec('from ' + module.__name__ + ' import *', namespace)
        assert set(namespace) - {'__builtins__'} == set(expected)
        for name in expected:
          assert namespace[name] is getattr(module, name)
    ''')

  def test_failed_export_resolution_is_not_cached_or_relabelled(self):
    self.run_script('''
      import importlib
      from unittest.mock import patch

      cases = (('', 'Zoom'), ('.base', 'VideoExtensoConfig'),
               ('.config_tools', 'HistogramProcess'),
               ('.tkinter', 'TkinterDISCorrelConfig'),
               ('.pyqt', 'PyQtDISCorrelConfig'))
      for suffix, name in cases:
        module = importlib.import_module('crappy.tool.camera_config' + suffix)
        assert name not in vars(module), name
        for error in (ImportError('dependency'), RuntimeError('module bug'),
                      KeyboardInterrupt()):
          with patch.object(module, '_import_module', side_effect=error):
            try:
              getattr(module, name)
            except type(error) as caught:
              assert caught is error
            else:
              raise AssertionError('Export resolution swallowed a failure')
          assert name not in vars(module), name
        selected = getattr(module, name)
        assert isinstance(selected, type), name
        assert vars(module)[name] is selected
    ''')

  def test_disabled_configuration_and_request_pickling_do_not_resolve(self):
    self.run_script('''
      import importlib.abc
      import pickle
      import sys
      from unittest.mock import Mock, patch
      import numpy as np

      class RejectGUI(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.startswith(('crappy.tool.camera_config.pyqt',
                                  'crappy.tool.camera_config.tkinter')):
            raise AssertionError('Disabled configuration imported ' + fullname)
      sys.meta_path.insert(0, RejectGUI())
      from crappy import Block
      from crappy._failure import FailureManager
      from crappy.blocks.vision import CameraSource, DICVEProcessor, VisionBlock

      source = CameraSource(camera='', config=False,
                            image_generator=lambda x, y: np.zeros((4, 5),
                                                                  dtype=np.uint8),
                            img_shape=(4, 5), img_dtype='uint8')
      source._failure_manager = FailureManager(source.name, 'prepare')
      source.log = Mock()
      source.img_outputs.append(Mock())
      processor = DICVEProcessor(patches=[(0, 0, 1, 1)])
      request = processor.request_config(source.name)
      assert request.configurator._classes == {}
      restored = pickle.loads(pickle.dumps(request))
      assert restored.img_source == request.img_source
      assert list(restored.configurator) == ['tkinter', 'pyqt']
      assert restored.configurator._classes == {}
      request.connection = Mock()
      source.add_config_request_in(request)
      with patch.object(VisionBlock, 'prepare'):
        source.prepare()
      assert request.completed
      request.connection.send.assert_called_once_with(None)
      assert source.configurator._classes == {}
      assert processor.configurator._classes == {}
      source._camera.close()
      Block.reset()
    ''')
