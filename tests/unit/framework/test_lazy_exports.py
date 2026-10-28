# coding: utf-8

"""Public lazy exports, isolated imports, and string-based driver selection."""

import subprocess
import sys
import textwrap
from unittest import TestCase


def resolve_driver_in_worker(category, name, queue):
  """Importable spawn target: resolve a selected driver in a fresh worker."""

  from importlib import import_module

  module = import_module('crappy.' + category)
  registry = getattr(module, category + '_dict')
  cold = name not in registry
  selected = getattr(module, 'resolve_' + category)(name)
  assert selected is getattr(module, name) is registry[name]
  queue.put((cold, selected.__module__))
  queue.close()
  queue.join_thread()


class TestLazyExports(TestCase):
  def _run_script(self, script: str) -> None:
    result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(script)],
                            capture_output=True, text=True, timeout=45)
    if result.returncode == 77:
      self.skipTest('OpenCV is required for bundled resource wildcard exports')
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_root_and_generator_do_not_import_unused_features(self):
    self._run_script('''
      import importlib.abc
      import sys

      forbidden = {'scipy', 'skimage', 'cv2', 'SimpleITK', 'tables', 'PIL',
                   'PyQt6', 'tkinter', 'gi', 'serial', 'paho', 'ximea',
                   'pycuda', 'nidaqmx', 'pyqtgraph', 'matplotlib', 'graphviz'}
      class RejectUnused(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname == 'crappy.modifier' or fullname.startswith('crappy.modifier.'):
            raise AssertionError('Modifiers imported too early: ' + fullname)
          if fullname.split('.')[0] in forbidden:
            raise AssertionError('Unrelated dependency: ' + fullname)
      sys.meta_path.insert(0, RejectUnused())
      import crappy
      assert 'crappy.blocks.generator_path' not in sys.modules
      assert 'Path' not in vars(crappy)
      assert 'Path' in dir(crappy) and 'Path' in crappy.__all__
      assert 'crappy.blocks.generator_path' not in sys.modules
      from crappy.blocks import Generator
      from crappy.blocks.generator import Generator as implementation
      assert Generator is implementation is crappy.blocks.Generator
      assert not forbidden.intersection(sys.modules)
      for name in ('camera', 'actuator', 'inout', 'lamcube', 'modifier', 'tool'):
        assert 'crappy.' + name not in sys.modules
      assert 'crappy.blocks.vision.block' not in sys.modules
      assert 'crappy.blocks.camera' not in sys.modules
      assert 'crappy._resources' not in sys.modules
      for name in ('start', 'stop', 'prepare', 'launch', 'renice', 'reset'):
        assert getattr(crappy, name).__self__ is crappy.Block
    ''')

  def test_path_alias_and_builtin_exports_resolve_consistently(self):
    for first in ('alias', 'package', 'generator', 'direct'):
      with self.subTest(first=first):
        self._run_script(f'''
          import pickle
          import sys
          import crappy

          assert 'crappy.blocks.generator_path' not in sys.modules
          assert 'numpy' not in sys.modules
          if '{first}' == 'alias':
            from crappy import Path
          elif '{first}' == 'package':
            from crappy.blocks.generator_path import meta_path
            Path = meta_path.Path
          elif '{first}' == 'generator':
            from crappy.blocks import Generator
            Path = crappy.Path
          else:
            from crappy.blocks.generator_path.meta_path.path import Path
          from crappy.blocks.generator_path.meta_path import paths_dict
          assert Path is crappy.Path is crappy.blocks.generator_path.meta_path.Path
          assert pickle.loads(pickle.dumps(Path)) is Path
          assert paths_dict is Path.classes
          for name in 'Constant Cyclic Ramp CyclicRamp Integrator Sine Conditional Custom'.split():
            selected = getattr(crappy.blocks.generator_path, name)
            assert issubclass(selected, Path)
            assert paths_dict[name] is selected
        ''')

  def test_modifier_package_and_alias_resolve_the_same_real_class(self):
    entry_points = (
      'selected = crappy.modifier.Modifier',
      'selected = crappy.Modifier',
      'from crappy import Modifier as selected',
      'from crappy import modifier; selected = modifier.Modifier',
      'import crappy.modifier; selected = crappy.modifier.Modifier',
      'from crappy.modifier.meta_modifier import Modifier as selected',
    )
    for entry_point in entry_points:
      with self.subTest(entry_point=entry_point):
        self._run_script(f'''
          import pickle
          import sys
          import crappy

          assert 'crappy.modifier' not in sys.modules
          for name in ('modifier', 'Modifier'):
            assert name not in vars(crappy)
            assert name in crappy.__all__ and name in dir(crappy)
          assert 'crappy.modifier' not in sys.modules

          {entry_point}
          from crappy import Modifier, modifier
          from crappy.modifier.meta_modifier import Modifier as implementation
          assert selected is Modifier is modifier.Modifier is implementation
          assert isinstance(selected, type)
          assert pickle.loads(pickle.dumps(selected)) is selected
          assert modifier.modifier_dict is selected.classes
          assert set('Demux Diff DownSampler Integrate Mean Median MovingAvg MovingMed Offset TrigOnChange TrigOnValue Moving_avg Moving_med Trig_on_change Trig_on_value'.split()).issubset(selected.classes)
        ''')

  def test_modifier_resolution_failures_remain_retriable(self):
    for name in ('modifier', 'Modifier'):
      with self.subTest(name=name):
        self._run_script(f'''
          from unittest.mock import patch
          import crappy

          for failure in (ImportError('missing'), RuntimeError('broken'),
                          KeyboardInterrupt()):
            with patch.object(crappy, '_import_module', side_effect=failure):
              try:
                getattr(crappy, '{name}')
              except BaseException as error:
                assert error is failure
              else:
                raise AssertionError('Import failure was swallowed')
          selected = getattr(crappy, '{name}')
          assert selected is getattr(crappy, '{name}')
        ''')

  def test_category_discovery_does_not_import_driver_implementations(self):
    self._run_script('''
      import importlib
      import sys
      from unittest.mock import patch

      with patch('subprocess.run', side_effect=AssertionError('Eager probe')):
        for category, base_name in (('camera', 'Camera'),
                                    ('actuator', 'Actuator'),
                                    ('inout', 'InOut')):
          module = importlib.import_module('crappy.' + category)
          registry = getattr(module, category + '_dict')
          assert hasattr(registry, 'items')
          assert registry is getattr(module, base_name).classes
          assert registry == {}, registry
          names = getattr(module, category + '_names')()
          assert set(module._DRIVERS) == names
          assert set(module.__all__).issubset(dir(module))
          for relative, _ in module._DRIVERS.values():
            assert importlib.util.resolve_name(relative, module.__name__) not in sys.modules
    ''')

  def test_one_driver_does_not_import_its_siblings(self):
    for category, selected in (('camera', 'FakeCamera'),
                                ('actuator', 'FakeStepperMotor'),
                                ('inout', 'FakeInOut')):
      with self.subTest(category=category):
        self._run_script(f'''
          import importlib
          import sys
          from unittest.mock import patch
          module = importlib.import_module('crappy.{category}')
          with patch('subprocess.run', side_effect=AssertionError('Unused probe')):
            selected = getattr(module, '{selected}')
          assert isinstance(selected, type)
          assert selected is getattr(module, 'resolve_{category}')('{selected}')
          assert selected is getattr(module, '{category}_dict')['{selected}']
          for name, (relative, _) in module._DRIVERS.items():
            if name != '{selected}':
              assert importlib.util.resolve_name(relative, module.__name__) not in sys.modules
        ''')

  def test_unloaded_shipped_names_are_reserved(self):
    self._run_script('''
      import importlib
      import sys
      import crappy
      from crappy._exceptions import DefinitionError

      categories = [('crappy.blocks', crappy.Block, crappy.blocks._EXPORTS),
                    ('crappy.blocks.vision', crappy.Block,
                     crappy.blocks.vision._EXPORTS)]
      for category, base_name in (('camera', 'Camera'),
                                  ('actuator', 'Actuator'),
                                  ('inout', 'InOut')):
        module = importlib.import_module('crappy.' + category)
        categories.append((module.__name__, getattr(module, base_name),
                           module._DRIVERS))
      for package, base, exports in categories:
        for name, (relative, attribute) in exports.items():
          if attribute is None:
            continue
          assert name not in base.classes, name
          try:
            type(name, (base,), {'__module__': 'custom_drivers'})
          except DefinitionError:
            pass
          else:
            raise AssertionError('Accepted reserved name: ' + name)
          assert name not in base.classes
          assert importlib.util.resolve_name(relative, package) not in sys.modules
        custom = type('Custom' + package.replace('.', ''), (base,),
                      {'__module__': 'custom_drivers'})
        assert base.classes[custom.__name__] is custom
        try:
          type(custom.__name__, (base,), {'__module__': 'custom_drivers'})
        except DefinitionError:
          pass
        else:
          raise AssertionError('Accepted duplicate custom class')
    ''')

  def test_supported_exports_wildcards_identity_and_unknown_attributes(self):
    self._run_script('''
      import importlib
      import pickle
      try:
        import cv2
      except ImportError:
        raise SystemExit(77)
      expected = {
        'crappy': 'actuator blocks camera inout lamcube links modifier tool __version__ OptionalModule docs resources link img_link display_graph Block VisionBlock Actuator Camera InOut Modifier Path stop prepare launch start renice reset',
        'crappy.blocks': 'AutoDriveVideoExtenso Block Button Camera Canvas Dashboard DICVE DISCorrel FakeMachine Generator Grapher HDFRecorder IOBlock LinkReader Machine MeanBlock Multiplexer Pause PID Recorder Scheduler Sink StopBlock StopButton Synchronizer VideoExtenso generator_path schedulers camera_processes vision',
        'crappy.blocks.vision': 'VisionBlock CameraSource DICVEProcessor DISCorrelProcessor ImageDisplayer ImageRecorder VideoExtensoProcessor',
        'crappy.blocks.camera_processes': 'CameraProcess DISCorrelProcess Displayer DICVEProcess ImageSaver VideoExtensoProcess',
        'crappy.tool': 'camera_config image_processing ApplyStrainToImage',
        'crappy.tool.image_processing': 'DICVETool LostPatchError DISCorrelTool VideoExtensoTool LostSpotError',
        'crappy.tool.image_processing.video_extenso': 'LostSpotError VideoExtensoTool',
        'crappy.camera': 'FakeCamera FileReader Camera camera_setting moved_to_collection camera_dict',
        'crappy.actuator': 'FakeDCMotor FakeStepperMotor Actuator moved_to_collection actuator_dict',
        'crappy.inout': 'FakeInOut InOut moved_to_collection inout_dict',
      }
      for package, names in expected.items():
        module = importlib.import_module(package)
        names = set(names.split())
        assert names.issubset(module.__all__), package
        assert names.issubset(dir(module)), package
        # Whole-package wildcards elsewhere would also load excluded drivers
        # or GPU objects. Exercise those exports by their selected names.
        namespace = {name: getattr(module, name) for name in names}
        if package in ('crappy', 'crappy.blocks.vision',
                       'crappy.tool.image_processing.video_extenso'):
          namespace = {}
          exec('from ' + package + ' import *', namespace)
          assert set(namespace) - {'__builtins__'} == set(module.__all__)
        for name in names:
          value = getattr(module, name)
          assert value is namespace[name]
          if isinstance(value, type):
            assert pickle.loads(pickle.dumps(value)) is value
        try:
          getattr(module, 'nonexistent_lazy_export')
        except AttributeError:
          pass
        else:
          raise AssertionError('Unknown export was accepted')
      import crappy
      assert crappy.Camera is crappy.camera.Camera
      assert crappy.Actuator is crappy.actuator.Actuator
      assert crappy.InOut is crappy.inout.InOut
      assert crappy.Modifier is crappy.modifier.Modifier
      assert crappy.VisionBlock is crappy.blocks.vision.VisionBlock
      assert crappy.Path is crappy.blocks.generator_path.meta_path.Path
      assert crappy.link is crappy.links.link
      assert crappy.img_link is crappy.links.img_link
      assert crappy.display_graph.__self__ is crappy.links.link_graph
    ''')

  def test_export_import_failures_are_not_cached_or_reclassified(self):
    for category, name in (('camera', 'FakeCamera'),
                            ('actuator', 'FakeStepperMotor'),
                            ('inout', 'FakeInOut'),
                            ('blocks', 'Generator'),
                            ('tool', 'ApplyStrainToImage')):
      with self.subTest(category=category):
        self._run_script(f'''
          import importlib
          from unittest.mock import patch
          module = importlib.import_module('crappy.{category}')
          for failure in (ImportError('missing'), RuntimeError('broken'),
                          KeyboardInterrupt()):
            with patch.object(module, '_import_module', side_effect=failure):
              try:
                getattr(module, '{name}')
              except BaseException as error:
                assert error is failure
              else:
                raise AssertionError('Import error was swallowed')
          selected = getattr(module, '{name}')
          assert selected is getattr(module, '{name}')
        ''')

  def test_string_constructors_import_only_selected_builtins(self):
    cases = (
      ('camera', 'FakeCamera', "crappy.blocks.Camera('FakeCamera', config=False)"),
      ('camera', 'FakeCamera', "crappy.blocks.vision.CameraSource('FakeCamera', config=False)"),
      ('inout', 'FakeInOut', "crappy.blocks.IOBlock('FakeInOut')"),
      ('actuator', 'FakeDCMotor', "crappy.blocks.Machine([{'type': 'FakeDCMotor'}])"),
      ('actuator', 'FakeDCMotor', "crappy.blocks.AutoDriveVideoExtenso({'type': 'FakeDCMotor'})"),
    )
    for category, name, constructor in cases:
      with self.subTest(constructor=constructor):
        self._run_script(f'''
          import importlib
          import sys
          import warnings
          import crappy
          warnings.simplefilter('ignore')
          module = importlib.import_module('crappy.{category}')
          assert '{name}' not in getattr(module, '{category}_dict')
          block = {constructor}
          assert '{name}' in getattr(module, '{category}_dict')
          for sibling, (relative, _) in module._DRIVERS.items():
            if sibling != '{name}':
              assert importlib.util.resolve_name(relative, module.__name__) not in sys.modules
          crappy.reset()
        ''')

  def test_unknown_driver_diagnostics_list_unloaded_builtins(self):
    for category, constructor in (
        ('camera', "crappy.blocks.Camera('UnknownDriver', config=False)"),
        ('camera', "crappy.blocks.vision.CameraSource('UnknownDriver', config=False)"),
        ('inout', "crappy.blocks.IOBlock('UnknownDriver')"),
        ('actuator', "crappy.blocks.Machine([{'type': 'UnknownDriver'}])")):
      with self.subTest(constructor=constructor):
        self._run_script(f'''
          import importlib
          import sys
          import crappy
          module = importlib.import_module('crappy.{category}')
          try:
            {constructor}
          except ValueError as error:
            assert all(name in str(error) for name in module._DRIVERS), error
          else:
            raise AssertionError('Unknown driver accepted')
          for relative, _ in module._DRIVERS.values():
            assert importlib.util.resolve_name(relative, module.__name__) not in sys.modules
        ''')


  def test_driver_resolution_in_fresh_workers(self):
    self._run_script('''
      import importlib
      import multiprocessing as mp
      from tests.unit.framework.test_lazy_exports import resolve_driver_in_worker

      for category, name in (('camera', 'FakeCamera'),
                              ('actuator', 'FakeDCMotor'),
                              ('inout', 'FakeInOut')):
        module = importlib.import_module('crappy.' + category)
        selected = getattr(module, 'resolve_' + category)(name)
        for method in mp.get_all_start_methods():
          context = mp.get_context(method)
          queue = context.Queue()
          process = context.Process(target=resolve_driver_in_worker,
                                    args=(category, name, queue))
          try:
            process.start()
            cold, implementation = queue.get(timeout=10)
            process.join(timeout=10)
            assert process.exitcode == 0, (method, process.exitcode)
            assert cold is (method != 'fork'), (method, category, cold)
            assert implementation == selected.__module__
          finally:
            if process.is_alive():
              process.terminate()
              process.join(timeout=5)
            queue.close()
            queue.join_thread()
    ''')

  def test_removed_deprecated_block_api_is_not_exported(self):
    self._run_script('''
      import crappy
      assert 'Camera' not in crappy.Block.classes
      for name in ('recv_all', 'poll', 'recv_all_last', 'get_last',
                   'get_all_last', 'recv_all_delay', 'drop'):
        assert not hasattr(crappy.Block, name)
      for name in ('AutoDrive', 'Client_server', 'Displayer', 'DISVE',
                   'Drawing', 'Fake_machine', 'GUI', 'Hdf_recorder',
                   'Mean_block', 'Multiplex', 'Reader', 'Video_extenso'):
        assert name not in crappy.blocks.__all__
        assert name not in crappy.Block.classes
        assert not hasattr(crappy.blocks, name)
    ''')
