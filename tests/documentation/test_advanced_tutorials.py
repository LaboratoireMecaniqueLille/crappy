# coding: utf-8

from pathlib import Path
import unittest


class TestAdvancedTutorialDownloads(unittest.TestCase):
  """Checks that the downloadable advanced examples remain safe to import."""

  _download_dir = (Path(__file__).resolve().parents[2] / 'docs' / 'source' /
                   'downloads' / 'more_complexity')
  _examples = {
    'feedback_loop.py': 'feedback-loop',
    'generator_conditions.py': 'generator-conditions',
    'modifier.py': 'modifier',
    'organized_script.py': 'organized-script',
    'streaming_acquisition.py': 'streaming-acquisition',
    'test_camera_object.py': 'test-camera-object',
  }

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
