# coding: utf-8

"""Implementations specific to applications at the LaMcube laboratory.

They are kept separate because they are unlikely to be used as-is outside the
lab. Each implementation is imported only when requested.
"""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'Biaxe': ('.biaxe', 'Biaxe'),
            'Biotens': ('.biotens', 'Biotens')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .biaxe import Biaxe as Biaxe
  from .biotens import Biotens as Biotens


def __getattr__(name: str) -> _Any:
  """Loads and caches one public export on first access.

  .. versionadded:: 2.2.0
  """

  if name not in _EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module, attribute = _EXPORTS[name]
  value = getattr(_import_module(module, __name__), attribute)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
