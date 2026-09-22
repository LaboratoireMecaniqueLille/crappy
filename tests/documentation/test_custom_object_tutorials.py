# coding: utf-8

from pathlib import Path
from runpy import run_path
import unittest
from unittest.mock import patch

import crappy


class TestCustomObjectTutorialDownloads(unittest.TestCase):
  """Checks the downloadable custom-object examples."""

  _download_dir = (Path(__file__).resolve().parents[2] / 'docs' / 'source' /
                   'downloads' / 'custom_objects')
  _examples = {
    'custom_actuator_simulated.py': 'custom-actuator',
    'custom_all_in_one_camera.py': 'custom-all-in-one-camera',
    'custom_block_average.py': 'custom-block',
    'custom_camera_source.py': 'custom-camera',
    'custom_generator_path.py': 'custom-generator-path',
    'custom_inout_regular.py': 'custom-inout',
    'custom_inout_streamer.py': 'custom-streaming-inout',
    'custom_modifier_calibration.py': 'custom-modifier',
    'custom_vision_block_brightness.py': 'custom-vision-block',
  }

  def test_examples_are_import_safe(self) -> None:
    """Imports each example without starting a test or opening hardware."""

    for file_name in self._examples:
      with self.subTest(example=file_name):
        path = self._download_dir / file_name
        with patch.object(
            crappy,
            'start',
            side_effect=AssertionError('An imported example called start()')):
          namespace = run_path(str(path), run_name='docs_smoke')

        self.assertTrue(callable(namespace.get('main')))

  def test_literalinclude_markers_are_complete(self) -> None:
    """Checks each displayed code range has exactly one ordered marker pair."""

    for file_name, marker in self._examples.items():
      with self.subTest(example=file_name):
        path = self._download_dir / file_name
        source = path.read_text(encoding='utf-8')
        start = f'# [{marker}-start]'
        end = f'# [{marker}-end]'

        self.assertEqual(source.count(start), 1)
        self.assertEqual(source.count(end), 1)
        self.assertLess(source.index(start), source.index(end))
        compile(source, str(path), 'exec')
