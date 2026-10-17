# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'DICVETool': ('.dic_ve', 'DICVETool'),
            'LostPatchError': ('.dic_ve', 'LostPatchError'),
            'DISCorrelTool': ('.dis_correl', 'DISCorrelTool'),
            'GPUCorrelTool': ('.gpu_correl', 'GPUCorrelTool'),
            'VideoExtensoTool': ('.video_extenso', 'VideoExtensoTool'),
            'LostSpotError': ('.video_extenso', 'LostSpotError')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .dic_ve import DICVETool as DICVETool
  from .dic_ve import LostPatchError as LostPatchError
  from .dis_correl import DISCorrelTool as DISCorrelTool
  from .gpu_correl import GPUCorrelTool as GPUCorrelTool
  from .video_extenso import VideoExtensoTool as VideoExtensoTool
  from .video_extenso import LostSpotError as LostSpotError


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
