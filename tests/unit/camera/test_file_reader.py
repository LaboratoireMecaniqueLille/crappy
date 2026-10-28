# coding: utf-8

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from tests.support.clock import Clock

from crappy._exceptions import ReaderStop
from crappy._global import OptionalModule
from crappy.camera.file_reader import FileReader
import crappy.camera.file_reader as file_reader_module


class FakeCV2:
  """Small image-reading backend stand-in."""

  def __init__(self, clock=None, delay=0) -> None:
    self.paths = list()
    self.clock, self.delay = clock, delay

  @staticmethod
  def is_available() -> bool:
    return True

  def imread(self, path: str, flags: int):
    self.paths.append((path, flags))
    if self.clock is not None:
      self.clock.advance(self.delay)
    return np.array([[float(Path(path).stem.split('_')[1])]])


class FakeSitk:
  """Small SimpleITK-like backend stand-in."""

  def __init__(self, clock=None, delay=0) -> None:
    self.paths = list()
    self.clock, self.delay = clock, delay

  @staticmethod
  def is_available() -> bool:
    return True

  def ReadImage(self, path):
    self.paths.append(path)
    if self.clock is not None:
      self.clock.advance(self.delay)
    return path

  @staticmethod
  def GetArrayFromImage(path):
    return np.array([[float(Path(path).stem.split('_')[1])]])


class TestFileReader(TestCase):
  """Unit tests for the FileReader camera."""

  def setUp(self):
    self.clock = Clock()
    self.enterContext(patch.object(file_reader_module, 'time', self.clock))
    self.enterContext(patch.object(file_reader_module, 'sleep', self.clock.sleep))

  def make_folder(self, names: tuple[str, ...]) -> TemporaryDirectory:
    """Creates a temporary folder populated with fake image files."""

    tmp = TemporaryDirectory()
    folder = Path(tmp.name)
    for name in names:
      (folder / name).write_bytes(b'fake')
    return tmp

  def test_default_reader_delivers_frames_then_stops(self):
    backend = FakeCV2()
    with self.make_folder(('000001_0.000.png',)) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2')
        _, image = reader.get_image()
        np.testing.assert_array_equal(image, [[0]])
        with self.assertRaises(ReaderStop):
          reader.get_image()

  def test_open_selects_sitk_by_default_when_available(self) -> None:
    """Checks automatic backend selection priority."""

    fake_sitk = FakeSitk()
    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module, 'Sitk', fake_sitk):
        with patch.object(file_reader_module, 'cv2', fake_cv2):
          reader = FileReader()
          reader.open(tmp)
          reader.get_image()

    self.assertTrue(fake_sitk.paths)

    self.assertFalse(fake_cv2.paths)

  def test_open_selects_cv2_when_sitk_is_unavailable(self) -> None:
    """Checks fallback automatic backend selection."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module,
                        'Sitk',
                        OptionalModule('SimpleITK')):
        with patch.object(file_reader_module, 'cv2', fake_cv2):
          reader = FileReader()
          reader.open(tmp)
          reader.get_image()

    self.assertTrue(fake_cv2.paths)

  def test_open_raises_when_no_backend_is_available(self) -> None:
    """Checks automatic backend selection failure."""

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module,
                        'Sitk',
                        OptionalModule('SimpleITK')):
        with patch.object(file_reader_module,
                          'cv2',
                          OptionalModule('opencv-python')):
          with self.assertRaises(ModuleNotFoundError):
            FileReader().open(tmp)

  def test_requested_unavailable_backend_is_rejected(self) -> None:
    """Checks explicit backend availability validation."""

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module,
                        'cv2',
                        OptionalModule('opencv-python')):
        with self.assertRaises(ModuleNotFoundError):
          FileReader().open(tmp, reader_backend='cv2')

  def test_invalid_backend_name_is_rejected(self) -> None:
    """Checks backend name validation."""

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with self.assertRaises(ValueError):
        FileReader().open(tmp, reader_backend='bad')

  def test_missing_folder_is_rejected(self) -> None:
    """Checks folder validation."""

    with patch.object(file_reader_module, 'cv2', FakeCV2()):
      with self.assertRaises(FileNotFoundError):
        FileReader().open('/definitely/missing/folder', reader_backend='cv2')

  def test_invalid_paths_are_rejected_before_loading_backends(self) -> None:
    """Reader arguments are checked before importing optional libraries."""

    with TemporaryDirectory() as folder:
      existing = Path(folder) / 'file.txt'
      existing.touch()
      cases = ((None, TypeError), (1, TypeError), (b'folder', TypeError),
               ('', ValueError), (' ', ValueError), (Path(' '), ValueError),
               (Path(folder) / 'missing', FileNotFoundError),
               (existing, FileNotFoundError))
      with patch.object(file_reader_module.Sitk, 'is_available') as sitk, \
           patch.object(file_reader_module.cv2, 'is_available') as cv2:
        for value, error in cases:
          with self.subTest(value=value), self.assertRaises(error):
            FileReader().open(value)
        with self.assertRaises(TypeError):
          FileReader().open(folder, stop_at_end=1)
        sitk.assert_not_called()
        cv2.assert_not_called()

  def test_folder_without_matching_images_is_rejected(self) -> None:
    """Checks filename pattern validation."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('metadata.csv', 'image.png')) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        with self.assertRaises(FileNotFoundError):
          FileReader().open(tmp, reader_backend='cv2')

  def test_images_are_sorted_by_timestamp(self):
    backend = FakeCV2()
    with self.make_folder(('000002_2.000.png', '000001_1.000.png', 'not_an_image.png')) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2')
        first_t, first = reader.get_image()
        second_t, second = reader.get_image()
    self.assertEqual((first_t, second_t), (1, 2))
    np.testing.assert_array_equal(first, [[1]])
    np.testing.assert_array_equal(second, [[2]])

  def test_get_image_reads_with_cv2_and_sleeps_until_timestamp(self):
    backend = FakeCV2(self.clock, delay=0.2)
    self.clock.now = 100
    with self.make_folder(('000001_1.000.png',)) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2')
        timestamp, image = reader.get_image()
    self.assertAlmostEqual(timestamp, 101)
    np.testing.assert_array_equal(image, [[1]])
    self.assertAlmostEqual(sum(self.clock.sleeps), 0.8)
    self.assertEqual(backend.paths[0][1], 0)

  def test_get_image_reads_with_sitk(self):
    backend = FakeSitk(self.clock, delay=1.5)
    self.clock.now = 100
    with self.make_folder(('000001_1.000.png',)) as folder:
      with patch.object(file_reader_module, 'Sitk', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='sitk')
        timestamp, image = reader.get_image()
    self.assertEqual(timestamp, 101.5)
    np.testing.assert_array_equal(image, [[1]])
    self.assertFalse(self.clock.sleeps)

  def test_get_image_raises_reader_stop_when_exhausted(self) -> None:
    """Checks default exhaustion behavior."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        reader = FileReader()
        reader.open(tmp, reader_backend='cv2')
        with patch.object(file_reader_module, 'time', return_value=0):
          reader.get_image()
        with self.assertRaises(ReaderStop):
          reader.get_image()

  def test_get_image_idles_after_exhaustion_when_requested(self):
    backend = FakeCV2()
    with self.make_folder(('000001_0.000.png',)) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2', stop_at_end=False)
        reader.get_image()
        self.assertIsNone(reader.get_image())
        self.assertIsNone(reader.get_image())
    self.assertTrue(self.clock.sleeps)
    self.assertTrue(all(delay > 0 for delay in self.clock.sleeps))
    self.assertEqual(len(backend.paths), 1)

  def test_open_resets_idle_state_after_non_stopping_exhaustion(self):
    backend = FakeCV2()
    with self.make_folder(('000001_0.000.png',)) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2', stop_at_end=False)
        reader.get_image()
        self.assertIsNone(reader.get_image())
        self.clock.now = 10
        reader.open(folder, reader_backend='cv2', stop_at_end=False)
        timestamp, image = reader.get_image()
    self.assertEqual(timestamp, 10)
    np.testing.assert_array_equal(image, [[0]])
    self.assertEqual(len(backend.paths), 2)

  def test_stopped_reader_waits_without_consuming_images(self):
    backend = FakeCV2()
    with self.make_folder(('000001_0.000.png',)) as folder:
      with patch.object(file_reader_module, 'cv2', backend):
        reader = FileReader()
        reader.open(folder, reader_backend='cv2', stop_at_end=False)
        reader.get_image()
        self.assertIsNone(reader.get_image())
        (Path(folder) / '000002_1.000.png').write_bytes(b'fake')
        for _ in range(2):
          self.assertIsNone(reader.get_image())
    self.assertEqual(len(backend.paths), 1)
    self.assertGreater(sum(self.clock.sleeps), 0)
