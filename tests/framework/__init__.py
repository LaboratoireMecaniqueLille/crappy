"""Headless tests for shared framework failure and cleanup primitives."""

from pathlib import Path
from unittest import TestLoader, TestSuite


def load_tests(loader: TestLoader,
               standard_tests: TestSuite,
               pattern: str | None) -> TestSuite:
  """Discover this package through the existing lazy root test aggregator."""

  package_dir = Path(__file__).resolve().parent
  return loader.discover(start_dir=str(package_dir),
                         pattern=pattern or 'test_*.py',
                         top_level_dir=str(package_dir.parent.parent))
