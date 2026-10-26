# coding: utf-8

from pathlib import Path as FilePath
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.generator_path.custom as custom_module
from crappy.blocks.generator_path.custom import Custom
from crappy.blocks.generator_path.meta_path import Path


class TestCustom(TestCase):
  """Unit tests for the Custom Generator Path."""

  def setUp(self) -> None:
    Path.t0 = 10
    Path.last_cmd = None

  @staticmethod
  def _write_file(folder: str, name: str, content: str) -> FilePath:
    """Writes a temporary path file."""

    path = FilePath(folder) / name
    path.write_text(content)
    return path

  def test_custom_interpolates_values_from_file(self) -> None:
    """Checks interpolation and end-of-file transition."""

    with TemporaryDirectory() as folder:
      path = self._write_file(folder, 'path.csv',
                              '0,0\n1,10\n2,20\n')
      custom = Custom(path)

      with patch.object(custom_module, 'time', return_value=10.5):
        self.assertEqual(custom.get_cmd({}), 5)
      with patch.object(custom_module, 'time', return_value=12):
        self.assertEqual(custom.get_cmd({}), 20)
      with patch.object(custom_module, 'time', return_value=12.1):
        with self.assertRaises(StopIteration):
          custom.get_cmd({})

  def test_custom_accepts_custom_delimiters(self) -> None:
    """Checks delimiter forwarding to loadtxt."""

    with TemporaryDirectory() as folder:
      path = self._write_file(folder, 'path.txt',
                              '0;0\n1;10\n')

      custom = Custom(path, delimiter=';')

      with patch.object(custom_module, 'time', return_value=10.5):
        self.assertEqual(custom.get_cmd({}), 5)

  def test_custom_rejects_non_two_column_files(self) -> None:
    """Checks shape validation for path files."""

    cases = {
      'one_row.csv': '0,1\n',
      'one_column.csv': '0\n1\n',
      'three_columns.csv': '0,1,2\n1,2,3\n',
    }

    with TemporaryDirectory() as folder:
      for name, content in cases.items():
        with self.subTest(name=name):
          path = self._write_file(folder, name, content)
          with self.assertRaises(ValueError):
            Custom(path)

  def test_custom_rejects_unsorted_timestamps(self) -> None:
    """Checks timestamp monotonicity validation."""

    with TemporaryDirectory() as folder:
      path = self._write_file(folder, 'unsorted.csv',
                              '1,10\n0,0\n')

      with self.assertRaises(ValueError):
        Custom(path)
