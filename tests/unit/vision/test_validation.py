"""Public validation and protected state for this feature."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from crappy.blocks.vision import DICVEProcessor, ImageRecorder, VideoExtensoProcessor
from crappy.tool.camera_config import SpotsBoxes
from tests.support.vision import VisionTestBase


class TestProcessorValidation(VisionTestBase):
  def test_patch_configuration_fails_before_building_child_collections(self):
    with patch.object(SpotsBoxes, 'set_spots') as set_spots:
      with self.assertRaises((TypeError, ValueError)):
        DICVEProcessor(patches=[(1, 1, 2, 2), (2, 2, 0, 1)])
      set_spots.assert_not_called()
    with self.assertRaises((TypeError, ValueError)):
      SpotsBoxes(spot_1='box')
    for options in ({'blur': 0}, {'blur': 2}, {'border': -1}, {'num_spots': 5}):
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          VideoExtensoProcessor(**options)

  def test_save_folders_reject_files_and_allow_directory_paths(self):
    constructors = (
      lambda path: ImageRecorder(save_folder=path, save_backend='npy'),
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
