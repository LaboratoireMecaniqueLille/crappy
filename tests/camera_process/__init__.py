"""Tests for :class:`crappy.blocks.camera_processes.CameraProcess`."""

from pathlib import Path
from unittest import TestLoader, TestSuite


def load_tests(loader: TestLoader,
               standard_tests: TestSuite,
               pattern: str | None) -> TestSuite:
  """Discover this package without importing every test at package import."""

  package_dir = Path(__file__).resolve().parent
  return loader.discover(start_dir=str(package_dir),
                         pattern=pattern or 'test_*.py',
                         top_level_dir=str(package_dir.parents[1]))
