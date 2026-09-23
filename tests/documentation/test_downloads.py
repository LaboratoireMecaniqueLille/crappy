# coding: utf-8

from pathlib import Path
from runpy import run_path
import unittest
from unittest.mock import patch

import crappy


class TestDownloadablePythonFiles(unittest.TestCase):
  """Checks every Python file offered as a documentation download."""

  _download_dir = (Path(__file__).resolve().parents[2] / 'docs' / 'source' /
                   'downloads')

  def test_all_downloads_compile(self) -> None:
    """Syntax-checks current downloads and files added in the future."""

    for path in sorted(self._download_dir.rglob('*.py')):
      with self.subTest(download=path.relative_to(self._download_dir)):
        source = path.read_text(encoding='utf-8')
        compile(source, str(path), 'exec')

  def test_all_downloads_are_import_safe(self) -> None:
    """Imports downloads without starting a test or accessing hardware."""

    for path in sorted(self._download_dir.rglob('*.py')):
      with self.subTest(download=path.relative_to(self._download_dir)):
        with patch.object(
            crappy,
            'start',
            side_effect=AssertionError('An imported example called start()')):
          namespace = run_path(str(path), run_name='docs_smoke')

        self.assertTrue(callable(namespace.get('main')))
