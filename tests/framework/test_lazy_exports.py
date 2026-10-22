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
  selected = getattr(module, '_resolve_' + category)(name)
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

  def test_path_alias_is_deferred_but_builtin_paths_remain_grouped(self):
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
          assert vars(crappy)['Path'] is Path
          assert pickle.loads(pickle.dumps(Path)) is Path
          assert paths_dict is Path.classes
          assert set(paths_dict) == set('Constant Cyclic Ramp CyclicRamp Integrator Sine Conditional Custom'.split())
          for module in ('constant', 'cyclic', 'ramp', 'cyclic_ramp',
                         'integrator', 'sine', 'conditional', 'custom'):
            assert 'crappy.blocks.generator_path.' + module in sys.modules
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
          assert vars(crappy)['Modifier'] is selected
          assert vars(crappy)['modifier'] is modifier
          assert modifier.modifier_dict is selected.classes
          assert set(selected.classes) == set('Demux Diff DownSampler Integrate Mean Median MovingAvg MovingMed Offset TrigOnChange TrigOnValue Moving_avg Moving_med Trig_on_change Trig_on_value'.split())
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
            assert '{name}' not in vars(crappy)
          selected = getattr(crappy, '{name}')
          assert selected is vars(crappy)['{name}']
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
          assert type(registry) is dict
          assert registry is getattr(module, base_name).classes
          assert registry == {}, registry
          names = getattr(module, '_' + category + '_names')()
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
          assert selected is getattr(module, '_resolve_{category}')('{selected}')
          assert selected is getattr(module, '{category}_dict')['{selected}']
          assert selected is vars(module)['{selected}']
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

  def test_frozen_exports_wildcards_identity_and_unknown_attributes(self):
    self._run_script('''
      import importlib
      import pickle
      try:
        import cv2
      except ImportError:
        raise SystemExit(77)
      expected = {
        'crappy': 'actuator blocks camera inout lamcube links modifier tool __version__ OptionalModule docs resources link img_link display_graph Block VisionBlock Actuator Camera InOut Modifier Path stop prepare launch start renice reset',
        'crappy.blocks': 'AutoDriveVideoExtenso Block Button Camera Canvas ClientServer Dashboard DICVE DISCorrel FakeMachine Generator GPUCorrel GPUVE Grapher HDFRecorder IOBlock LinkReader Machine MeanBlock Multiplexer Pause PID Recorder Scheduler Sink StopBlock StopButton Synchronizer UController VideoExtenso generator_path schedulers camera_processes vision AutoDrive Client_server Displayer DISVE Drawing Fake_machine GUI Hdf_recorder Mean_block Multiplex Reader Video_extenso',
        'crappy.blocks.vision': 'VisionBlock CameraSource DICVEProcessor DISCorrelProcessor ImageDisplayer ImageRecorder VideoExtensoProcessor',
        'crappy.blocks.camera_processes': 'CameraProcess DISCorrelProcess Displayer DICVEProcess GPUCorrelProcess GPUVEProcess ImageSaver VideoExtensoProcess',
        'crappy.tool': 'bindings camera_config image_processing ApplyStrainToImage',
        'crappy.tool.image_processing': 'DICVETool LostPatchError DISCorrelTool GPUCorrelTool VideoExtensoTool LostSpotError',
        'crappy.tool.image_processing.video_extenso': 'LostSpotError VideoExtensoTool',
        'crappy.camera': 'FakeCamera FileReader Webcam RaspberryPiCamera2 XiAPI Camera camera_setting moved_to_collection CameraGstreamer CameraOpencv camera_dict',
        'crappy.actuator': 'FakeDCMotor FakeStepperMotor JVLMac140 ServoStar300 Phidget4AStepper PololuTic Actuator moved_to_collection actuator_dict',
        'crappy.inout': 'ADS1115 DAQmx FakeInOut GPIOPWM GPIOSwitch LabjackT7 T7Streamer MPRLS NAU7802 NIDAQmx PhidgetWheatstoneBridge InOut moved_to_collection inout_dict',
        'crappy.lamcube': 'Biaxe Biotens',
      }
      for package, names in expected.items():
        module = importlib.import_module(package)
        names = set(names.split())
        assert names == set(module.__all__), package
        assert names.issubset(dir(module)), package
        namespace = {}
        exec('from ' + package + ' import *', namespace)
        assert set(namespace) - {'__builtins__'} == names
        for name in names:
          value = getattr(module, name)
          assert value is namespace[name] is vars(module)[name]
          if isinstance(value, type):
            assert pickle.loads(pickle.dumps(value)) is value
        try:
          getattr(module, 'nonexistent_lazy_export')
        except AttributeError as error:
          assert package in str(error)
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
            assert '{name}' not in vars(module)
          selected = getattr(module, '{name}')
          assert selected is vars(module)['{name}']
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

  def test_lab_actuator_resolution_and_direct_imports(self):
    for name, sibling in (('Biaxe', 'Biotens'), ('Biotens', 'Biaxe')):
      for first in ('attribute', 'leaf', 'constructor'):
        with self.subTest(name=name, first=first):
          self._run_script(f'''
            import importlib
            import sys
            import crappy
            if '{first}' == 'attribute':
              selected = getattr(crappy.lamcube, '{name}')
            elif '{first}' == 'leaf':
              selected = getattr(importlib.import_module('crappy.lamcube.' + '{name}'.lower()), '{name}')
            else:
              crappy.blocks.Machine([{{'type': '{name}'}}])
              selected = crappy.actuator.actuator_dict['{name}']
            assert crappy.actuator._resolve_actuator('{name}') is selected
            assert getattr(crappy.lamcube, '{name}') is selected
            assert 'crappy.lamcube.' + '{sibling}'.lower() not in sys.modules
          ''')

  def test_platform_camera_probe_runs_only_on_selected_variant(self):
    for platform, available, variant in (('Linux', True, 'v4l2'),
                                         ('Linux', False, 'basic'),
                                         ('Windows', True, 'basic')):
      with self.subTest(platform=platform, available=available):
        self._run_script(f'''
          import sys
          from unittest.mock import patch
          with patch('platform.system', return_value='{platform}'), \\
               patch('subprocess.run', side_effect={"None" if available else "FileNotFoundError('v4l2-ctl')"}) as probe:
            import crappy.camera as cameras
            assert probe.call_count == 0
            cameras._camera_names()
            assert probe.call_count == 0
            first = cameras.CameraOpencv
            expected = 1 if '{platform}' == 'Linux' else 0
            assert probe.call_count == expected
            assert first.__module__.endswith('opencv_camera_{variant}')
            assert 'crappy.camera.gstreamer_camera_{variant}' not in sys.modules
            second = cameras.CameraGstreamer
            assert second.__module__.endswith('gstreamer_camera_{variant}')
            assert probe.call_count == expected
            assert cameras.CameraOpencv is first
        ''')

  def test_direct_platform_camera_imports_keep_identity(self):
    for variant in ('basic', 'v4l2'):
      self._run_script(f'''
        import importlib
        from unittest.mock import patch
        with patch('subprocess.run', side_effect=AssertionError('Unused probe')):
          implementation = importlib.import_module('crappy.camera.opencv_camera_{variant}')
          import crappy.camera as cameras
          assert cameras.CameraOpencv is implementation.CameraOpencv
          assert cameras._resolve_camera('CameraOpencv') is implementation.CameraOpencv
      ''')

  def test_driver_resolution_in_fresh_workers(self):
    self._run_script('''
      import importlib
      import multiprocessing as mp
      from tests.framework.test_lazy_exports import resolve_driver_in_worker

      for category, name in (('camera', 'FakeCamera'),
                              ('actuator', 'FakeDCMotor'),
                              ('inout', 'FakeInOut')):
        module = importlib.import_module('crappy.' + category)
        selected = getattr(module, '_resolve_' + category)(name)
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

  def test_deprecated_block_methods_remain_eager(self):
    self._run_script('''
      import crappy
      assert 'Camera' not in crappy.Block.classes
      for name in ('recv_all', 'poll', 'recv_all_last', 'get_last',
                   'get_all_last', 'recv_all_delay', 'drop'):
        method = getattr(crappy.Block, name)
        try:
          method(None)
        except NotImplementedError:
          pass
        else:
          raise AssertionError('Lost deprecation error: ' + name)
      for name in ('AutoDrive', 'Client_server', 'Displayer', 'DISVE',
                   'Drawing', 'Fake_machine', 'GUI', 'Hdf_recorder',
                   'Mean_block', 'Multiplex', 'Reader', 'Video_extenso'):
        assert getattr(crappy.blocks, name) is crappy.Block.classes[name]
    ''')
