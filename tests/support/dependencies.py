"""Distinguish absent optional packages from broken installed dependencies."""

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


def optional_import(name: str) -> ModuleType | None:
  """Return None only when the top-level package is absent.

  Once a package exists, import errors (including a missing transitive
  dependency or native library) must fail the test instead of hiding coverage.
  """

  if find_spec(name.split('.')[0]) is None:
    return None
  return import_module(name)
