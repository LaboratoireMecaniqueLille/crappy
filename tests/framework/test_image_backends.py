# coding: utf-8

"""Backend import boundaries, exercised without hardware or GUI windows."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

import crappy._global as common
import crappy.camera.file_reader as reader_module
import crappy.blocks.camera_processes.record as saver_module
import crappy.blocks.camera_processes.display as display_module
import crappy.blocks.vision.record as recorder_module
import crappy.blocks.vision.display as displayer_module
from crappy.blocks.camera import Camera
from crappy._exceptions import OptionalDependencyError
from crappy._global import OptionalModule

from tests.block.block_test_base import BlockTestBase


RECORDERS = ((saver_module, saver_module.ImageSaver),
             (recorder_module, recorder_module.ImageRecorder))
DISPLAYERS = ((display_module, display_module.Displayer),
              (displayer_module, displayer_module.ImageDisplayer))
DEPENDENCIES = {'Sitk': ('SimpleITK', 'SimpleITK'),
                'Image': ('PIL.Image', 'Pillow'),
                'ExifTags': ('PIL.ExifTags', 'Pillow'),
                'cv2': ('cv2', 'opencv-python'),
                'plt': ('matplotlib.pyplot', 'matplotlib')}


@contextmanager
def fresh_dependencies(module):
  """Simulates the unresolved dependency declarations of a fresh worker."""

  with ExitStack() as stack:
    dependencies = {}
    for name, (import_name, install_name) in DEPENDENCIES.items():
      if name in vars(module):
        dependency = OptionalModule(import_name, install_name=install_name,
                                     lazy_import=True)
        dependencies[name] = dependency
        stack.enter_context(patch.object(module, name, dependency))
    yield dependencies


def fake_modules():
  """Provides real module objects with deterministic image/GUI stand-ins."""

  modules = {name: ModuleType(name) for name, _ in DEPENDENCIES.values()}
  sitk = modules['SimpleITK']
  sitk.ReadImage = Mock(return_value=np.zeros((2, 3), dtype=np.uint8))
  sitk.GetArrayFromImage = lambda image: image
  sitk.GetImageFromArray = lambda image, **kwargs: image
  sitk.WriteImage = Mock()
  image = modules['PIL.Image']
  image.init = Mock()
  image.Exif = dict
  image.fromarray = Mock(return_value=Mock())
  modules['PIL.ExifTags'].TAGS = {42016: 'ImageUniqueID'}
  cv2 = modules['cv2']
  cv2.imread = Mock(return_value=np.zeros((2, 3), dtype=np.uint8))
  cv2.imwrite = Mock()
  cv2.WINDOW_NORMAL = 0
  cv2.WINDOW_KEEPRATIO = 0
  cv2.namedWindow = Mock()
  cv2.imshow = Mock()
  cv2.waitKey = Mock()
  plt = modules['matplotlib.pyplot']
  plt.ion = Mock()
  plt.subplots = Mock(return_value=(Mock(), Mock()))
  plt.pause = Mock()
  plt.show = Mock()
  return modules


class TestImageBackends(BlockTestBase):
  def setUp(self):
    super().setUp()
    # Hold strong references until the Block harness has reset global state.
    self.components = []

  def recorder(self, cls, backend, folder):
    component = cls(save_backend=backend, save_folder=folder)
    self.components.append(component)
    component.log = Mock()
    return component

  def displayer(self, cls, backend):
    component = cls(title='Backend test', framerate=5, backend=backend)
    self.components.append(component)
    component.log = Mock()
    return component

  @staticmethod
  def prepare_component(component):
    """Runs the actual hook without threads, windows, or shared memory."""

    if isinstance(component, (recorder_module.ImageRecorder,
                              displayer_module.ImageDisplayer)):
      link = Mock()
      link.name = 'image'
      component.add_img_input(link)
      with patch.object(recorder_module.VisionBlock, 'prepare'):
        component.prepare()
    else:
      with patch.object(display_module, 'Thread'):
        component.init()

  @staticmethod
  def feed_component(component):
    metadata = {'ImageUniqueID': 1, 't(s)': 0.0}
    image = np.zeros((2, 3), dtype=np.uint8)
    if isinstance(component, (recorder_module.ImageRecorder,
                              displayer_module.ImageDisplayer)):
      component.receive_imgs = Mock(return_value=['image'])
      component.last_received['image'].metadata = metadata
      component.last_received['image'].id = 1
      component.last_received['image'].img = image
      component.send = Mock()
    else:
      component.metadata = metadata
      component.img = image

  def test_feature_imports_leave_their_optional_dependencies_unresolved(self):
    script = '''
      import builtins
      import importlib

      targets = {
        'crappy.camera.file_reader': ('Sitk', 'cv2'),
        'crappy.blocks.camera_processes.record':
          ('Sitk', 'Image', 'ExifTags', 'cv2'),
        'crappy.blocks.camera_processes.display': ('cv2', 'plt'),
        'crappy.blocks.vision.record': ('Sitk', 'Image', 'ExifTags', 'cv2'),
        'crappy.blocks.vision.display': ('cv2', 'plt'),
      }
      original_import = builtins.__import__
      def guarded_import(name, globals=None, *args, **kwargs):
        if (globals and globals.get('__name__') in targets and
            name.split('.')[0] in {'SimpleITK', 'PIL', 'cv2', 'matplotlib'}):
          raise AssertionError(f'Eager backend import: {name}')
        return original_import(name, globals, *args, **kwargs)
      builtins.__import__ = guarded_import
      for name, attributes in targets.items():
        module = importlib.import_module(name)
        for attribute in attributes:
          dependency = vars(module)[attribute]
          assert dependency._lazy, (name, attribute)
          assert dependency._module is None, (name, attribute)
    '''
    result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(script)],
                            capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_explicit_recorders_probe_only_the_requested_backend(self):
    expected = {'sitk': ['SimpleITK'], 'pil': ['PIL.Image', 'PIL.ExifTags'],
                'cv2': ['cv2'], 'npy': []}
    for module, cls in RECORDERS:
      for backend, names in expected.items():
        with self.subTest(cls=cls.__name__, backend=backend):
          modules = fake_modules()
          with (fresh_dependencies(module),
                patch.object(common, 'import_module',
                             side_effect=modules.__getitem__) as imports):
            component = self.recorder(cls, backend, 'unused')
          self.assertEqual(component._save_backend, backend)
          self.assertEqual([call.args[0] for call in imports.call_args_list],
                           names)
          modules['PIL.Image'].init.assert_not_called()

  def test_automatic_recorders_stop_at_the_first_usable_backend(self):
    cases = (
      ((), 'sitk', ['SimpleITK']),
      (('SimpleITK',), 'pil', ['SimpleITK', 'PIL.Image', 'PIL.ExifTags']),
      (('SimpleITK', 'PIL.Image'), 'cv2', ['SimpleITK', 'PIL.Image', 'cv2']),
      (('SimpleITK', 'PIL.ExifTags'), 'cv2',
       ['SimpleITK', 'PIL.Image', 'PIL.ExifTags', 'cv2']),
      (('SimpleITK', 'PIL.Image', 'cv2'), 'npy',
       ['SimpleITK', 'PIL.Image', 'cv2']),
    )
    for module, cls in RECORDERS:
      for missing, expected, names in cases:
        with self.subTest(cls=cls.__name__, missing=missing):
          modules = fake_modules()

          def load(name):
            if name in missing:
              raise ImportError(f'{name} cannot initialize')
            return modules[name]

          with (fresh_dependencies(module),
                patch.object(common, 'import_module', side_effect=load) as imports):
            component = self.recorder(cls, None, 'unused')
          self.assertEqual(component._save_backend, expected)
          self.assertEqual([call.args[0] for call in imports.call_args_list], names)

  def test_explicit_recorders_reject_missing_dependencies(self):
    for module, cls in RECORDERS:
      for backend, missing in (('sitk', 'SimpleITK'), ('pil', 'PIL.Image'),
                               ('pil', 'PIL.ExifTags'), ('cv2', 'cv2')):
        with self.subTest(cls=cls.__name__, backend=backend, missing=missing):
          modules = fake_modules()

          def load(name):
            if name == missing:
              raise ModuleNotFoundError(name, name=name)
            return modules[name]

          with (fresh_dependencies(module),
                patch.object(common, 'import_module', side_effect=load)):
            with self.assertRaisesRegex(ModuleNotFoundError, backend):
              self.recorder(cls, backend, 'unused')

  def test_recorders_resolve_in_fresh_workers_before_the_first_frame(self):
    expected = {'sitk': ['SimpleITK'], 'pil': ['PIL.Image', 'PIL.ExifTags'],
                'cv2': ['cv2'], 'npy': []}
    for module, cls in RECORDERS:
      for backend, names in expected.items():
        with self.subTest(cls=cls.__name__, backend=backend):
          modules = fake_modules()
          with (TemporaryDirectory() as folder,
                fresh_dependencies(module),
                patch.object(common, 'import_module',
                             side_effect=modules.__getitem__)):
            component = self.recorder(cls, backend, folder)
            # The child has its own unresolved module-level placeholders.
            with (fresh_dependencies(module),
                  patch.object(common, 'import_module',
                               side_effect=modules.__getitem__) as imports):
              self.prepare_component(component)
              self.assertEqual([call.args[0] for call in imports.call_args_list],
                               names)
              if backend == 'pil':
                modules['PIL.Image'].init.assert_called_once_with()
                self.assertEqual(component._pil_tags_inv, {'ImageUniqueID': 42016})
              self.assertFalse(any(isinstance(value, ModuleType)
                                   for value in vars(component).values()))
              self.feed_component(component)
              imports.side_effect = AssertionError('Import in recording loop')
              component.loop()

  def test_recorders_fail_before_creating_folders_if_worker_imports_fail(self):
    for module, cls in RECORDERS:
      for backend in ('sitk', 'pil', 'cv2'):
        with self.subTest(cls=cls.__name__, backend=backend):
          modules = fake_modules()
          with (TemporaryDirectory() as folder,
                fresh_dependencies(module),
                patch.object(common, 'import_module',
                             side_effect=modules.__getitem__)):
            destination = Path(folder) / 'recording'
            component = self.recorder(cls, backend, destination)
            cause = ImportError('worker backend cannot load')
            with (fresh_dependencies(module),
                  patch.object(common, 'import_module', side_effect=cause)):
              with self.assertRaises(OptionalDependencyError) as caught:
                self.prepare_component(component)
              self.assertIs(caught.exception.__cause__, cause)
              self.assertFalse(destination.exists())

  def test_real_recording_backends_round_trip_png_images(self):
    """Checks actual codecs as well as the import-boundary stand-ins."""

    for module, cls in RECORDERS:
      for backend in ('sitk', 'pil', 'cv2'):
        with self.subTest(cls=cls.__name__, backend=backend):
          dependencies = {'sitk': (module.Sitk,),
                          'pil': (module.Image, module.ExifTags),
                          'cv2': (module.cv2,)}
          if not all(item.is_available() for item in dependencies[backend]):
            self.skipTest(f'{backend} is unavailable')
          with TemporaryDirectory() as folder:
            component = self.recorder(cls, backend, folder)
            component._img_extension = 'png'
            self.prepare_component(component)
            self.feed_component(component)
            image = np.arange(6, dtype=np.uint8).reshape(2, 3)
            if isinstance(component, recorder_module.ImageRecorder):
              component.last_received['image'].img = image
            else:
              component.img = image
            with patch.object(common, 'import_module',
                              side_effect=AssertionError('Import in first frame')):
              component.loop()
            path = Path(folder) / '000001_0.000.png'
            self.assertTrue(path.is_file())
            if backend == 'pil':
              with module.Image.open(path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), image)
                self.assertEqual(saved.getexif()[42016], '1')
            else:
              reader = reader_module.FileReader()
              reader.open(folder, reader_backend=backend)
              np.testing.assert_array_equal(reader.get_image()[1], image)

  def test_reader_resolves_at_open_and_reuses_the_backend_for_frames(self):
    cases = ((None, (), 'sitk', ['SimpleITK']),
             (None, ('SimpleITK',), 'cv2', ['SimpleITK', 'cv2']),
             ('sitk', (), 'sitk', ['SimpleITK']),
             ('cv2', (), 'cv2', ['cv2']))
    for backend, missing, expected, names in cases:
      with self.subTest(backend=backend, missing=missing):
        modules = fake_modules()

        def load(name):
          if name in missing:
            raise ImportError(name)
          return modules[name]

        with (TemporaryDirectory() as folder,
              fresh_dependencies(reader_module),
              patch.object(common, 'import_module', side_effect=load) as imports):
          (Path(folder) / '000001_0.000.png').touch()
          reader = reader_module.FileReader()
          self.assertEqual(imports.call_count, 0)
          reader.open(folder, reader_backend=backend)
          self.assertEqual(reader._backend, expected)
          self.assertEqual([call.args[0] for call in imports.call_args_list], names)
          imports.side_effect = AssertionError('Import in reading loop')
          _, image = reader.get_image()
          self.assertEqual(image.shape, (2, 3))

  def test_reader_reports_missing_backends(self):
    for backend, names in ((None, ['SimpleITK', 'cv2']),
                            ('sitk', ['SimpleITK']), ('cv2', ['cv2'])):
      with self.subTest(backend=backend):
        with (fresh_dependencies(reader_module),
              patch.object(common, 'import_module',
                           side_effect=ImportError('missing')) as imports):
          with self.assertRaises(ModuleNotFoundError):
            reader_module.FileReader().open(Path.cwd(), reader_backend=backend)
        self.assertEqual([call.args[0] for call in imports.call_args_list], names)

  def test_automatic_displayers_preserve_preference_and_failure_causes(self):
    for module, cls in DISPLAYERS:
      for missing, names, backend in (((), ['cv2'], 'cv2'),
                                      (('cv2',), ['cv2', 'matplotlib.pyplot'], 'mpl'),
                                      (('cv2', 'matplotlib.pyplot'),
                                       ['cv2', 'matplotlib.pyplot'], None)):
        with self.subTest(cls=cls.__name__, missing=missing):
          modules = fake_modules()
          cause = ImportError('native loader failure')

          def load(name):
            if name in missing:
              raise cause
            return modules[name]

          with (fresh_dependencies(module),
                patch.object(common, 'import_module', side_effect=load) as imports):
            if backend is None:
              with self.assertRaises(ModuleNotFoundError) as caught:
                self.displayer(cls, None)
              self.assertIsInstance(caught.exception.__cause__, OptionalDependencyError)
              self.assertIs(caught.exception.__cause__.__cause__, cause)
            else:
              component = self.displayer(cls, None)
              self.assertEqual(component._backend, backend)
          self.assertEqual([call.args[0] for call in imports.call_args_list], names)

  def test_explicit_displayers_resolve_only_during_worker_preparation(self):
    for module, cls in DISPLAYERS:
      for backend, name in (('cv2', 'cv2'), ('mpl', 'matplotlib.pyplot')):
        with self.subTest(cls=cls.__name__, backend=backend):
          modules = fake_modules()
          with (fresh_dependencies(module),
                patch.object(common, 'import_module',
                             side_effect=modules.__getitem__) as imports):
            component = self.displayer(cls, backend)
            self.assertEqual(imports.call_count, 0)
            self.prepare_component(component)
            self.assertEqual([call.args[0] for call in imports.call_args_list], [name])
            self.feed_component(component)
            imports.side_effect = AssertionError('Import in display loop')
            component.loop()

  def test_missing_display_backend_fails_before_threads_or_windows_start(self):
    for module, cls in DISPLAYERS:
      for backend, name in (('cv2', 'cv2'), ('mpl', 'matplotlib.pyplot')):
        with self.subTest(cls=cls.__name__, backend=backend):
          cause = ImportError('selected backend cannot load')
          with (fresh_dependencies(module),
                patch.object(common, 'import_module', side_effect=cause) as imports):
            component = self.displayer(cls, backend)
            with (patch.object(component, '_prepare_cv2') as cv_window,
                  patch.object(component, '_prepare_mpl') as mpl_window,
                  patch.object(display_module, 'Thread') as thread):
              with self.assertRaises(OptionalDependencyError) as caught:
                if isinstance(component, displayer_module.ImageDisplayer):
                  self.prepare_component(component)
                else:
                  component.init()
              self.assertIs(caught.exception.__cause__, cause)
              self.assertEqual(caught.exception.import_name, name)
              cv_window.assert_not_called()
              mpl_window.assert_not_called()
              thread.assert_not_called()
          self.assertEqual([call.args[0] for call in imports.call_args_list], [name])

  def test_invalid_backend_names_do_not_probe_dependencies(self):
    with patch.object(common, 'import_module',
                      side_effect=AssertionError('Unnecessary backend probe')):
      for backend in ('invalid', [], {}, 1):
        with self.subTest(backend=backend):
          with self.assertRaises(ValueError):
            Camera('', config=False, image_generator=lambda x, y: None,
                   save_backend=backend)
          with self.assertRaises(ValueError):
            Camera('', config=False, image_generator=lambda x, y: None,
                   displayer_backend=backend)
          with self.assertRaises(ValueError):
            self.recorder(recorder_module.ImageRecorder, backend, 'unused')
          with self.assertRaises(ValueError):
            self.displayer(displayer_module.ImageDisplayer, backend)
          with self.assertRaises(ValueError):
            reader_module.FileReader().open('unused', reader_backend=backend)

  def test_unrelated_initialization_errors_and_interruptions_propagate(self):
    for error in (RuntimeError('initialization failed'), StopIteration(),
                  KeyboardInterrupt(),
                  BaseExceptionGroup('interrupted initialization',
                                     [ValueError(), SystemExit(2)])):
      for module, cls in (*RECORDERS, *DISPLAYERS, (reader_module, None)):
        with self.subTest(module=module.__name__, error=type(error).__name__):
          with (fresh_dependencies(module),
                patch.object(common, 'import_module', side_effect=error)):
            with self.assertRaises(type(error)) as caught:
              if cls is None:
                reader_module.FileReader().open(Path.cwd())
              elif (module, cls) in RECORDERS:
                self.recorder(cls, None, 'unused')
              else:
                self.displayer(cls, None)
            self.assertIs(caught.exception, error)
