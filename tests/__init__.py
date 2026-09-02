"""Crappy's test suite.

Test packages are deliberately not imported here.  Eager imports make an
otherwise unrelated test depend on every optional test dependency and are
especially expensive when multiprocessing uses the ``spawn`` or ``forkserver``
start method.
"""

from pathlib import Path
from unittest import TestLoader, TestSuite


def load_tests(loader: TestLoader,
               standard_tests: TestSuite,
               pattern: str | None) -> TestSuite:
  """Load each test package once without importing them eagerly.

  Calling :meth:`~unittest.TestLoader.discover` recursively here would collect
  test classes exported by legacy package initializers and then collect the
  same classes again from their modules.
  """

  tests_dir = Path(__file__).resolve().parent
  suite = TestSuite()

  for package_dir in sorted(tests_dir.iterdir()):
    if package_dir.is_dir() and (package_dir / '__init__.py').is_file():
      suite.addTests(loader.loadTestsFromName(
        f'{__name__}.{package_dir.name}'))

  return suite
