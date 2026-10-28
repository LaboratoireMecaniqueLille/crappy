# coding: utf-8

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch, sentinel
import csv
import unittest

import numpy as np

import crappy.blocks.vision.record as record_module
from crappy._global import OptionalModule
from crappy.blocks.vision import ImageRecorder

from tests.support.vision import VisionTestBase


class TestImageRecorder(VisionTestBase):
  """Unit tests for the ImageRecorder VisionBlock."""

  def make_recorder(self, **kwargs) -> ImageRecorder:
    """Creates and tracks a recorder using the dependency-free backend."""

    options = {'save_backend': 'npy'}
    options.update(kwargs)
    recorder = ImageRecorder(**options)
    self.track_block(recorder)
    return recorder

  @staticmethod
  def add_image_input(recorder: ImageRecorder,
                      name: str = 'record-image') -> Mock:
    """Registers a minimal input ImageLink double."""

    link = Mock()
    link.name = name
    recorder.add_img_input(link)
    return link

  def feed_image(self,
                 recorder: ImageRecorder,
                 image: np.ndarray,
                 metadata,
                 transport_id: int,
                 name: str = 'record-image') -> None:
    """Installs a received image and makes receive_imgs report it."""

    if name not in recorder.last_received:
      self.add_image_input(recorder, name)
    received = recorder.last_received[name]
    received.id = transport_id
    received.metadata = metadata
    received.img = image
    recorder.receive_imgs = Mock(return_value=[name])

  def test_constructor_sets_paths_backend_and_period(self) -> None:
    """Default-directory recording honors the requested sampling period."""

    with TemporaryDirectory() as tmp, patch.object(Path, 'cwd', return_value=Path(tmp)):
      recorder = self.make_recorder(save_period=3, img_extension='')
      self.add_image_input(recorder)
      with patch.object(record_module.VisionBlock, 'prepare'):
        recorder.prepare()
      image = np.arange(6, dtype=np.uint8).reshape(2, 3)
      for index in (0, 1, 3):
        self.feed_image(recorder, image + index,
                        {'ImageUniqueID': index, 't(s)': float(index)},
                        transport_id=index)
        recorder.loop()

      folder = Path(tmp) / 'Crappy_images'
      saved = sorted(folder.glob('*.npy'))
      self.assertEqual(len(saved), 2)
      np.testing.assert_array_equal(np.load(saved[0]), image)
      np.testing.assert_array_equal(np.load(saved[1]), image + 3)
      self.assertEqual(recorder.freq, 100)

  def test_constructor_validates_user_arguments(self) -> None:
    """Checks extension, directory, period, and backend validation."""

    invalid = (
      {'img_extension': '', 'save_backend': None},
      {'save_folder': ''},
      {'save_folder': 1},
      {'save_period': 0},
      {'save_period': -1},
      {'save_period': 1.5},
      {'save_backend': 'invalid'},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises(ValueError):
          ImageRecorder(**options)

  def test_constructor_selects_available_backend_or_numpy_fallback(self) -> None:
    """Automatic selection writes with the first usable backend."""

    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    for backend in ('sitk', 'pil', 'npy'):
      with self.subTest(backend=backend), TemporaryDirectory() as tmp:
        sitk = Mock() if backend == 'sitk' else OptionalModule('missing-sitk')
        pil = Mock() if backend == 'pil' else OptionalModule('missing-pil')
        with (patch.object(record_module, 'Sitk', sitk),
              patch.object(record_module, 'Image', pil),
              patch.object(record_module, 'ExifTags', Mock()),
              patch.object(record_module, 'cv2', OptionalModule('missing-cv2'))):
          recorder = ImageRecorder(save_folder=tmp)
          self.track_block(recorder)
          self.feed_image(recorder, image, {'ImageUniqueID': 1, 't(s)': 0.25}, 0)
          recorder.loop()

          if backend == 'sitk':
            np.testing.assert_array_equal(sitk.GetImageFromArray.call_args.args[0], image)
            self.assertTrue(sitk.WriteImage.called)
          elif backend == 'pil':
            np.testing.assert_array_equal(pil.fromarray.call_args.args[0], image)
            self.assertTrue(pil.fromarray.return_value.save.called)
          else:
            saved, = Path(tmp).glob('*.npy')
            np.testing.assert_array_equal(np.load(saved), image)

  def test_constructor_rejects_explicit_unavailable_backend(self) -> None:
    """Checks clear errors for missing requested optional dependencies."""

    mapping = {'sitk': 'Sitk', 'pil': 'Image', 'cv2': 'cv2'}
    for backend, attribute in mapping.items():
      with self.subTest(backend=backend):
        with patch.object(record_module, attribute,
                          OptionalModule(f'missing-{backend}')):
          with self.assertRaises(ModuleNotFoundError):
            ImageRecorder(save_backend=backend)

  def test_prepare_validates_supported_topology(self) -> None:
    """Checks the one-image-input, no-other-input topology."""

    recorder = self.make_recorder()
    with self.assertRaises(IOError):
      recorder.prepare()

    self.add_image_input(recorder)
    recorder.img_outputs.append(Mock())
    with self.assertRaises(IOError):
      recorder.prepare()

    recorder.img_outputs.clear()
    recorder.inputs.append(Mock())
    with self.assertRaises(IOError):
      recorder.prepare()

    recorder.inputs.clear()
    self.add_image_input(recorder, 'second-image')
    with self.assertRaises(IOError):
      recorder.prepare()

  def test_prepare_creates_folder_and_calls_inherited_preparation(self) -> None:
    """Checks directory creation before image-buffer attachment."""

    with TemporaryDirectory() as tmp:
      folder = Path(tmp) / 'nested' / 'images'
      recorder = self.make_recorder(save_folder=folder)
      self.add_image_input(recorder)

      with patch.object(record_module.VisionBlock,
                        'prepare') as inherited:
        recorder.prepare()

      self.assertTrue(folder.is_dir())
      inherited.assert_called_once_with()

  def test_prepare_suffixes_existing_recording_folder(self) -> None:
    """A previous recording and occupied sibling directories stay intact."""

    with TemporaryDirectory() as tmp:
      root = Path(tmp)
      folder = root / 'images'
      folder.mkdir()
      metadata_path = folder / 'metadata.csv'
      metadata_path.write_text('already used\n')
      occupied = root / 'images_00001'
      occupied.mkdir()
      (occupied / 'keep.txt').write_text('keep')
      recorder = self.make_recorder(save_folder=folder)
      self.add_image_input(recorder)
      with patch.object(record_module.VisionBlock, 'prepare'):
        recorder.prepare()
      image = np.ones((2, 3), dtype=np.uint8)
      self.feed_image(recorder, image, {'ImageUniqueID': 1, 't(s)': 0.25}, 0)
      recorder.loop()

      saved, = root.rglob('*.npy')
      self.assertNotIn(saved.parent, (folder, occupied))
      np.testing.assert_array_equal(np.load(saved), image)
      self.assertEqual(metadata_path.read_text(), 'already used\n')
      self.assertEqual((occupied / 'keep.txt').read_text(), 'keep')

  def test_prepare_reuses_existing_folder_without_metadata(self) -> None:
    """An existing empty directory receives the new recording."""

    with TemporaryDirectory() as tmp:
      folder = Path(tmp) / 'images'
      folder.mkdir()
      recorder = self.make_recorder(save_folder=folder)
      self.add_image_input(recorder)
      with patch.object(record_module.VisionBlock, 'prepare'):
        recorder.prepare()
      image = np.ones((2, 3), dtype=np.uint8)
      self.feed_image(recorder, image, {'ImageUniqueID': 1, 't(s)': 0.25}, 0)
      recorder.loop()

      saved, = folder.glob('*.npy')
      np.testing.assert_array_equal(np.load(saved), image)

  def test_loop_skips_when_no_new_image(self) -> None:
    """Checks idle polling and handled-image frequency accounting."""

    recorder = self.make_recorder(display_freq=True)
    recorder.receive_imgs = Mock(return_value=[])
    recorder._print_freq = Mock()

    recorder.loop()

    recorder._print_freq.assert_called_once_with(img_handled=False)

  def test_loop_uses_transport_id_for_periodic_numpy_recording(self) -> None:
    """Checks period selection, NumPy files, CSV rows, and notifications."""

    with TemporaryDirectory() as tmp:
      folder = Path(tmp)
      recorder = self.make_recorder(save_folder=folder,
                                    save_period=3,
                                    display_freq=True)
      recorder.send = Mock()
      recorder._print_freq = Mock()
      image = np.arange(6, dtype=np.uint16).reshape(2, 3)
      first_metadata = {
        'ImageUniqueID': 100,
        't(s)': 1.234,
        'DateTimeOriginal': '2020:01:01 00:00:00',
      }
      self.feed_image(recorder, image, first_metadata, transport_id=0)

      recorder.loop()

      first_path = folder / '000100_1.234.npy'
      self.assertTrue(first_path.exists())
      np.testing.assert_array_equal(np.load(first_path), image)
      recorder._print_freq.assert_called_with(img_handled=True)

      skipped = image + 10
      skipped_metadata = {**first_metadata,
                          'ImageUniqueID': 101,
                          't(s)': 1.5}
      self.feed_image(recorder, skipped, skipped_metadata, transport_id=1)
      recorder.loop()
      self.assertFalse((folder / '000101_1.500.npy').exists())
      recorder._print_freq.assert_called_with(img_handled=False)

      third = image + 30
      third_metadata = {**first_metadata,
                        'ImageUniqueID': 103,
                        't(s)': 2.0}
      self.feed_image(recorder, third, third_metadata, transport_id=3)
      recorder.loop()
      third_path = folder / '000103_2.000.npy'
      self.assertTrue(third_path.exists())
      np.testing.assert_array_equal(np.load(third_path), third)

      with open(folder / 'metadata.csv', newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
      self.assertEqual([row['ImageUniqueID'] for row in rows], ['100', '103'])
      self.assertEqual([row['t(s)'] for row in rows], ['1.234', '2.0'])
      self.assertEqual(recorder.send.call_args_list, [
        unittest.mock.call({'t(s)': 1.234,
                            'img_index': 100,
                            'meta': first_metadata}),
        unittest.mock.call({'t(s)': 2.0,
                            'img_index': 103,
                            'meta': third_metadata}),
      ])

  def test_loop_rejects_missing_metadata_or_mandatory_keys(self) -> None:
    """Incomplete metadata is rejected without requiring valid prior state."""

    image = np.zeros((2, 2), dtype=np.uint8)
    for metadata in (None, {'t(s)': 0.1}, {'ImageUniqueID': 1}):
      with self.subTest(metadata=metadata), TemporaryDirectory() as tmp:
        recorder = self.make_recorder(save_folder=tmp)
        self.feed_image(recorder, image, metadata, transport_id=0)
        with self.assertRaises((RuntimeError, KeyError, TypeError, ValueError)):
          recorder.loop()

  def test_loop_writes_simpleitk_grayscale_and_color(self) -> None:
    """Checks SimpleITK conversion, including BGR-to-RGB color reversal."""

    with TemporaryDirectory() as tmp:
      fake_sitk = Mock()
      fake_sitk.GetImageFromArray.return_value = sentinel.sitk_image
      with patch.object(record_module, 'Sitk', fake_sitk):
        for name, image, expected, kwargs in (
            ('gray', np.arange(6, dtype=np.uint8).reshape(2, 3),
             np.arange(6, dtype=np.uint8).reshape(2, 3), {}),
            ('color', np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
             np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, :, ::-1],
             {'isVector': True})):
          with self.subTest(name=name):
            folder = Path(tmp) / name
            folder.mkdir()
            recorder = ImageRecorder(save_folder=folder,
                                     save_backend='sitk')
            self.track_block(recorder)
            metadata = {'ImageUniqueID': 1, 't(s)': 0.25}
            self.feed_image(recorder, image, metadata, transport_id=0)
            recorder.loop()

            conversion = fake_sitk.GetImageFromArray.call_args
            np.testing.assert_array_equal(conversion.args[0], expected)
            self.assertEqual(conversion.kwargs, kwargs)
            fake_sitk.WriteImage.assert_called_with(
                sentinel.sitk_image,
                str(folder / '000001_0.250.tiff'))
            fake_sitk.reset_mock()

  def test_loop_writes_pillow_grayscale_and_color_with_exif(self) -> None:
    """Checks Pillow conversion, channel reversal, and EXIF forwarding."""

    with TemporaryDirectory() as tmp:
      fake_pil = Mock()
      encoded = Mock()
      fake_pil.fromarray.return_value = encoded
      with (patch.object(record_module, 'Image', fake_pil),
            patch.object(record_module, 'ExifTags', Mock())):
        for name, image, expected in (
            ('gray', np.arange(6, dtype=np.uint8).reshape(2, 3),
             np.arange(6, dtype=np.uint8).reshape(2, 3)),
            ('color', np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
             np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, :, ::-1])):
          with self.subTest(name=name):
            folder = Path(tmp) / name
            folder.mkdir()
            recorder = ImageRecorder(save_folder=folder, save_backend='pil')
            self.track_block(recorder)
            recorder._pil_exif = Mock(return_value=sentinel.exif)
            metadata = {'ImageUniqueID': 2, 't(s)': 0.5}
            self.feed_image(recorder, image, metadata, transport_id=0)
            recorder.loop()

            np.testing.assert_array_equal(
                fake_pil.fromarray.call_args.args[0], expected)
            encoded.save.assert_called_with(
                str(folder / '000002_0.500.tiff'), exif=sentinel.exif)
            recorder._pil_exif.assert_called_once_with(metadata)
            fake_pil.reset_mock()
            encoded.reset_mock()

  def test_loop_writes_opencv_image_without_channel_reversal(self) -> None:
    """Checks OpenCV receives the source image and final filename directly."""

    with TemporaryDirectory() as tmp:
      fake_cv2 = Mock()
      with patch.object(record_module, 'cv2', fake_cv2):
        recorder = ImageRecorder(img_extension='png',
                                 save_folder=tmp,
                                 save_backend='cv2')
        self.track_block(recorder)
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        metadata = {'ImageUniqueID': 3, 't(s)': 0.75}
        self.feed_image(recorder, image, metadata, transport_id=0)

        recorder.loop()

      path, written = fake_cv2.imwrite.call_args.args
      self.assertEqual(path, str(Path(tmp) / '000003_0.750.png'))
      np.testing.assert_array_equal(written, image)

  def test_pil_exif_converts_numpy_scalars_and_skips_unknown_keys(self) -> None:
    """Checks EXIF construction for normal Camera metadata values."""

    if not (record_module.Image.is_available() and
            record_module.ExifTags.is_available()):
      self.skipTest('Pillow is not available')
    tags_inv = {val: key for key, val in record_module.ExifTags.TAGS.items()}
    tmp = TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    recorder = ImageRecorder(save_backend='pil', save_folder=tmp.name)
    self.track_block(recorder)
    self.add_image_input(recorder)
    with patch.object(record_module.VisionBlock, 'prepare'):
      recorder.prepare()
    metadata = {
      'DateTimeOriginal': '2020:01:01 00:00:00',
      'ImageUniqueID': np.uint16(7),
      'SubsecTimeOriginal': np.float64(0.234567),
      'UnknownField': object(),
      't(s)': 1.0,
    }

    exif = recorder._pil_exif(metadata)

    self.assertEqual(exif.get(tags_inv['DateTimeOriginal']),
                     '2020:01:01 00:00:00')
    self.assertEqual(exif.get(tags_inv['ImageUniqueID']), '7')
    self.assertEqual(exif.get(tags_inv['SubsecTimeOriginal']), '0.234567')


if __name__ == '__main__':
  unittest.main()
