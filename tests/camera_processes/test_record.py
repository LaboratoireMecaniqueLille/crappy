# coding: utf-8

from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

import numpy as np

import crappy.blocks.camera_processes.record as record_module
from crappy._global import OptionalModule
from crappy.blocks.camera_processes.record import ImageSaver, TAGS_INV

from tests.camera_process.camera_process_test_base import (CameraProcessTestBase,
                                                           TestLink)


class TestImageSaver(CameraProcessTestBase):
  """Unit tests for the ImageSaver CameraProcess."""

  def test_constructor_validates_arguments(self) -> None:
    """Checks early validation of user-facing settings."""

    with self.assertRaises(ValueError):
      ImageSaver(save_backend='bad')

    for save_period in (0, -1, 1.5, '2'):
      with self.subTest(save_period=save_period):
        with self.assertRaises(ValueError):
          ImageSaver(save_period=save_period)

  def test_init_creates_folder_and_suffixes_existing_recording(self) -> None:
    """Checks save-folder creation and collision avoidance."""

    with TemporaryDirectory() as tmp:
      folder = Path(tmp) / 'images'
      saver = ImageSaver(save_folder=folder, save_backend='npy')
      saver.init()

      self.assertEqual(saver._save_folder, folder)
      self.assertTrue(folder.exists())

      (folder / 'metadata.csv').write_text('already used\n')
      saver = ImageSaver(save_folder=folder, save_backend='npy')
      saver.init()

      self.assertEqual(saver._save_folder, Path(tmp) / 'images_00001')
      self.assertTrue(saver._save_folder.exists())

  def test_get_data_respects_save_period(self) -> None:
    """Checks frame skipping and copied data for periodic recording."""

    self._process = ImageSaver(save_period=3, save_backend='npy')
    shared = self.make_shared(process=self._process,
                              shape=(2, 2),
                              dtype=np.uint16)

    img0 = np.arange(4, dtype=np.uint16).reshape(2, 2)
    self.write_image(shared, img0, {'ImageUniqueID': 0, 't(s)': 0.0})

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata['ImageUniqueID'], 0)
    np.testing.assert_array_equal(self._process.img, img0)

    img1 = img0 + 10
    self.write_image(shared, img1, {'ImageUniqueID': 1, 't(s)': 0.1})

    self.assertFalse(self._process._get_data())
    self.assertEqual(self._process.metadata['ImageUniqueID'], 0)
    np.testing.assert_array_equal(self._process.img, img0)

    img3 = img0 + 30
    self.write_image(shared, img3, {'ImageUniqueID': 3, 't(s)': 0.3})

    self.assertTrue(self._process._get_data())
    self.assertEqual(self._process.metadata['ImageUniqueID'], 3)
    np.testing.assert_array_equal(self._process.img, img3)

  def test_loop_saves_npy_metadata_and_optional_message(self) -> None:
    """Checks npy file writing, metadata CSV, and downstream message."""

    with TemporaryDirectory() as tmp:
      link = TestLink()
      saver = ImageSaver(save_folder=tmp,
                         save_backend='npy',
                         send_msg=True)
      self._process = saver
      self.set_test_logger(saver)
      saver._outputs = [link]
      saver.init()
      saver.metadata = {
        'ImageUniqueID': 5,
        't(s)': 1.234,
        'DateTimeOriginal': '2020:01:01 00:00:00',
      }
      saver.img = np.arange(6, dtype=np.uint8).reshape(2, 3)

      saver.loop()

      saved = Path(tmp) / '000005_1.234.npy'
      self.assertTrue(saved.exists())
      np.testing.assert_array_equal(np.load(saved), saver.img)

      with open(Path(tmp) / 'metadata.csv', newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))

      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]['ImageUniqueID'], '5')
      self.assertEqual(rows[0]['t(s)'], '1.234')
      self.assertEqual(rows[0]['DateTimeOriginal'], '2020:01:01 00:00:00')

      self.assertTrue(link.sent.is_set())
      self.assertEqual(link.sent_values[-1], {
        't(s)': 1.234,
        'img_index': 5,
        'meta': saver.metadata,
      })

  @unittest.skipIf(isinstance(record_module.PIL, OptionalModule),
                   "Pillow is not available")
  def test_loop_saves_pil_images_with_exif_metadata(self) -> None:
    """Checks PIL saving for common formats and EXIF object generation."""

    metadata = {
      'ImageUniqueID': np.uint16(7),
      't(s)': np.float64(1.234),
      'DateTimeOriginal': '2020:01:01 00:00:00',
      'SubsecTimeOriginal': np.float64(0.234567),
      'UnknownField': object(),
    }

    with TemporaryDirectory() as tmp:
      for extension in ('png', 'jpg', 'tiff'):
        with self.subTest(extension=extension):
          folder = Path(tmp) / extension
          saver = ImageSaver(img_extension=extension,
                             save_folder=folder,
                             save_backend='pil')
          saver.init()
          saver.metadata = metadata.copy()
          saver.img = np.arange(12, dtype=np.uint8).reshape(3, 4)

          saver.loop()

          saved = folder / f'000007_1.234.{extension}'
          self.assertTrue(saved.exists())

          with record_module.PIL.Image.open(saved) as img:
            exif = img.getexif()
          self.assertEqual(exif.get(TAGS_INV['DateTimeOriginal']),
                           '2020:01:01 00:00:00')
          self.assertEqual(exif.get(TAGS_INV['SubsecTimeOriginal']),
                           '0.234567')
          self.assertEqual(exif.get(TAGS_INV['ImageUniqueID']), '7')

  @unittest.skipIf(isinstance(record_module.PIL, OptionalModule),
                   "Pillow is not available")
  def test_pil_exif_converts_numpy_scalars_and_skips_unknown_keys(self) -> None:
    """Checks EXIF construction for normal Crappy metadata values."""

    saver = ImageSaver(save_backend='pil')
    self._process = saver
    self.set_test_logger(saver)
    saver.metadata = {
      'DateTimeOriginal': '2020:01:01 00:00:00',
      'ImageUniqueID': np.uint16(7),
      'SubsecTimeOriginal': np.float64(0.234567),
      'UnknownField': 'ignored',
      't(s)': 1.0,
    }

    exif = saver._pil_exif()

    self.assertEqual(exif.get(TAGS_INV['DateTimeOriginal']),
                     '2020:01:01 00:00:00')
    self.assertEqual(exif.get(TAGS_INV['ImageUniqueID']), '7')
    self.assertEqual(exif.get(TAGS_INV['SubsecTimeOriginal']), '0.234567')
