# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'LostSpotError': ('.tracker', 'LostSpotError'),
            'VideoExtensoTool': ('.video_extenso', 'VideoExtensoTool')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .tracker import LostSpotError as LostSpotError
  from .video_extenso import VideoExtensoTool as VideoExtensoTool


def __getattr__(name: str) -> _Any:
  """Loads and caches one public export on first access.

  .. versionadded:: 2.2.0
  """

  if name not in _EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

  module, attribute = _EXPORTS[name]
  imported = _import_module(module, __name__)
  value = imported if attribute is None else getattr(imported, attribute)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
