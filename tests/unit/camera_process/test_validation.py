"""Public validation and protected state for this feature."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from crappy.blocks import Camera, DICVE
from tests.support.camera_wrappers import CameraWrapperTestBase


class TestCameraValidation(CameraWrapperTestBase):
  def test_dicve_rejects_invalid_processor_configuration(self):
    for options in ({'patch_size': 8, 'patch_stride': 8}, {'alpha': np.nan}):
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          DICVE('', image_generator=lambda x, y: None, **options)

  def test_camera_options_fail_before_creating_workers(self):
    with patch('crappy.blocks.camera.ImageSaver') as saver, \
         patch('crappy.blocks.camera.Displayer') as displayer, \
         patch('crappy.blocks.camera.Manager') as manager:
      cases = ({'save_backend': 'bad'}, {'save_folder': ' '},
               {'img_extension': ' '}, {'save_period': 0},
               {'save_period': -1}, {'save_period': 1.5},
               {'save_period': '2'}, {'save_period': True},
               {'displayer_framerate': np.nan},
               {'displayer_framerate': np.inf},
               {'displayer_framerate': 0},
               {'displayer_framerate': -1},
               {'displayer_backend': 'bad'})
      for options in cases:
        with self.subTest(options=options):
          with self.assertRaises((TypeError, ValueError)):
            Camera('', config=False, image_generator=lambda x, y: None,
                   **options)
      saver.assert_not_called()
      displayer.assert_not_called()
      manager.assert_not_called()

  def test_save_folders_reject_files_and_allow_directory_paths(self):
    constructors = (
      lambda path: Camera('', config=False, image_generator=lambda x, y: None,
                           save_folder=path),
    )
    with TemporaryDirectory() as folder:
      existing = Path(folder) / 'keep.txt'
      existing.write_text('keep')
      missing = Path(folder) / 'new' / 'recording'
      for constructor in constructors:
        for value in (existing, str(existing)):
          with self.subTest(constructor=constructor, value=value):
            with self.assertRaises(NotADirectoryError):
              constructor(value)
        for value in ('', ' ', Path(' ')):
          with self.subTest(constructor=constructor, value=value):
            with self.assertRaises((TypeError, ValueError)):
              constructor(value)
        for value in (None, Path(), '.', Path(Path.cwd().anchor),
                       folder, Path(folder), missing, str(missing)):
          with self.subTest(constructor=constructor, value=value):
            self.assertIsNotNone(constructor(value))
      self.assertEqual(existing.read_text(), 'keep')
      self.assertFalse(missing.exists())
