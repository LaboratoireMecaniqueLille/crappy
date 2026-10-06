# coding: utf-8

from ast import literal_eval
from pathlib import Path
from re import search, DOTALL, MULTILINE
import unittest


class TestInstallationMetadata(unittest.TestCase):
  """Keeps documented base requirements aligned with package metadata."""

  _repository = Path(__file__).resolve().parents[2]
  _pyproject = (_repository / 'pyproject.toml').read_text(encoding='utf-8')
  _installation = (_repository / 'docs' / 'source' /
                   'installation.rst').read_text(encoding='utf-8')

  def test_python_requirement_matches_pyproject(self) -> None:
    """Checks the documented Python constraint against project metadata."""

    match = search(r'^requires-python\s*=\s*"([^"]+)"', self._pyproject,
                   flags=MULTILINE)
    self.assertIsNotNone(match)
    self.assertIn(f'Python ``{match.group(1)}``', self._installation)

  def test_base_dependencies_match_pyproject(self) -> None:
    """Checks every mandatory dependency is named with its constraint."""

    match = search(r'^dependencies\s*=\s*(\[.*?\])', self._pyproject,
                   flags=MULTILINE | DOTALL)
    self.assertIsNotNone(match)
    requirements = literal_eval(match.group(1))

    for requirement in requirements:
      with self.subTest(requirement=requirement):
        self.assertIn(f'``{requirement}``', self._installation)
