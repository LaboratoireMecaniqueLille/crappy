# coding: utf-8

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np

from crappy._global import OptionalModule, ReaderStop
from crappy.camera.file_reader import FileReader
import crappy.camera.file_reader as file_reader_module


class FakeCV2:
  """Small image-reading backend stand-in."""

  def __init__(self) -> None:
    self.paths = list()

  def imread(self, path: str, flags: int):
    self.paths.append((path, flags))
    return np.array([[float(Path(path).stem.split('_')[1])]])


class FakeSitk:
  """Small SimpleITK-like backend stand-in."""

  def __init__(self) -> None:
    self.paths = list()

  def ReadImage(self, path):
    self.paths.append(path)
    return path

  @staticmethod
  def GetArrayFromImage(path):
    return np.array([[float(Path(path).stem.split('_')[1])]])


class TestFileReader(TestCase):
  """Unit tests for the FileReader camera."""

  def make_folder(self, names: tuple[str, ...]) -> TemporaryDirectory:
    """Creates a temporary folder populated with fake image files."""

    tmp = TemporaryDirectory()
    folder = Path(tmp.name)
    for name in names:
      (folder / name).write_bytes(b'fake')
    return tmp

  def test_constructor_initializes_state(self) -> None:
    """Checks initial FileReader attributes."""

    reader = FileReader()

    self.assertIsNone(reader._images)
    self.assertTrue(reader._stop_at_end)
    self.assertIsNone(reader._backend)
    self.assertIsNone(reader._t0)
    self.assertFalse(reader._stopped)

  def test_open_selects_sitk_by_default_when_available(self) -> None:
    """Checks automatic backend selection priority."""

    fake_sitk = FakeSitk()
    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module, 'Sitk', fake_sitk):
        with patch.object(file_reader_module, 'cv2', fake_cv2):
          reader = FileReader()
          reader.open(tmp)

    self.assertEqual(reader._backend, 'sitk')

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

    self.assertEqual(reader._backend, 'cv2')

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

  def test_folder_without_matching_images_is_rejected(self) -> None:
    """Checks filename pattern validation."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('metadata.csv', 'image.png')) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        with self.assertRaises(FileNotFoundError):
          FileReader().open(tmp, reader_backend='cv2')

  def test_images_are_sorted_by_timestamp(self) -> None:
    """Checks image ordering independent from filename order."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000002_2.000.png',
                           '000001_1.000.png',
                           'not_an_image.png')) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        reader = FileReader()
        reader.open(tmp, reader_backend='cv2')
        with patch.object(file_reader_module,
                          'time',
                          side_effect=[0, 2, 2, 2, 2]):
          first_t, first = reader.get_image()
          second_t, second = reader.get_image()

    self.assertEqual(first_t, 2)
    self.assertEqual(second_t, 2)
    np.testing.assert_array_equal(first, np.array([[1.0]]))
    np.testing.assert_array_equal(second, np.array([[2.0]]))
    self.assertTrue(fake_cv2.paths[0][0].endswith('000001_1.000.png'))
    self.assertTrue(fake_cv2.paths[1][0].endswith('000002_2.000.png'))

  def test_get_image_reads_with_cv2_and_sleeps_until_timestamp(self) -> None:
    """Checks cv2 reading and replay timing."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_1.000.png',)) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        reader = FileReader()
        reader.open(tmp, reader_backend='cv2')
        with patch.object(file_reader_module,
                          'time',
                          side_effect=[100.0, 100.2, 101.0]):
          with patch.object(file_reader_module, 'sleep') as mocked_sleep:
            t, img = reader.get_image()

    self.assertEqual(t, 101.0)
    np.testing.assert_array_equal(img, np.array([[1.0]]))
    mocked_sleep.assert_called_once()
    self.assertAlmostEqual(mocked_sleep.call_args.args[0], 0.8)
    self.assertEqual(fake_cv2.paths[0][1], 0)

  def test_get_image_reads_with_sitk(self) -> None:
    """Checks SimpleITK reading path."""

    fake_sitk = FakeSitk()

    with self.make_folder(('000001_1.000.png',)) as tmp:
      with patch.object(file_reader_module, 'Sitk', fake_sitk):
        reader = FileReader()
        reader.open(tmp, reader_backend='sitk')
        with patch.object(file_reader_module,
                          'time',
                          side_effect=[100.0, 101.5, 101.5]):
          with patch.object(file_reader_module, 'sleep') as mocked_sleep:
            _, img = reader.get_image()

    np.testing.assert_array_equal(img, np.array([[1.0]]))
    mocked_sleep.assert_not_called()
    self.assertTrue(str(fake_sitk.paths[0]).endswith('000001_1.000.png'))

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

  def test_get_image_idles_after_exhaustion_when_requested(self) -> None:
    """Checks non-stopping exhaustion behavior."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        reader = FileReader()
        reader.open(tmp, reader_backend='cv2', stop_at_end=False)
        with patch.object(file_reader_module, 'time', return_value=0):
          reader.get_image()
        self.assertIsNone(reader.get_image())
        with patch.object(file_reader_module, 'sleep') as mocked_sleep:
          self.assertIsNone(reader.get_image())

    self.assertTrue(reader._stopped)
    mocked_sleep.assert_called_once_with(0.1)

  def test_open_resets_idle_state_after_non_stopping_exhaustion(self) -> None:
    """Checks reopening a reader after it idled at the end."""

    fake_cv2 = FakeCV2()

    with self.make_folder(('000001_0.000.png',)) as tmp:
      with patch.object(file_reader_module, 'cv2', fake_cv2):
        reader = FileReader()
        reader.open(tmp, reader_backend='cv2', stop_at_end=False)
        with patch.object(file_reader_module, 'time', return_value=0):
          reader.get_image()
        reader.get_image()
        self.assertTrue(reader._stopped)

        reader.open(tmp, reader_backend='cv2', stop_at_end=False)

    self.assertFalse(reader._stopped)
    self.assertIsNone(reader._t0)

  def test_stopped_reader_waits_without_consuming_images(self) -> None:
    """Checks idle behavior once the reader has stopped."""

    reader = FileReader()
    reader._stopped = True
    reader._images = iter([Path('000001_0.000.png')])
    reader._backend = 'cv2'

    with patch.object(file_reader_module, 'sleep') as mocked_sleep:
      self.assertIsNone(reader.get_image())

    mocked_sleep.assert_has_calls([call(0.1)])
