"""Tkinter camera-configuration integration tests."""

from pathlib import Path
from unittest import TestLoader, TestSuite


def load_tests(loader: TestLoader,
               standard_tests: TestSuite,
               pattern: str | None) -> TestSuite:
  """Discover this layer without importing its tests at package import."""

  package_dir = Path(__file__).resolve().parent
  return loader.discover(start_dir=str(package_dir),
                         pattern=pattern or 'test_*.py',
                         top_level_dir=str(package_dir.parents[2]))
