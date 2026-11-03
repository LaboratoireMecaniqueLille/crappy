# coding: utf-8

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'Box': ('.box', 'Box'),
            'HistogramProcess': ('.histogram_process', 'HistogramProcess'),
            'Overlay': ('.overlay_object', 'Overlay'),
            'SpotsBoxes': ('.spots_boxes', 'SpotsBoxes'),
            'SpotsDetector': ('.spots_detector', 'SpotsDetector'),
            'Zoom': ('.zoom', 'Zoom')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .box import Box
  from .histogram_process import HistogramProcess
  from .overlay_object import Overlay
  from .spots_boxes import SpotsBoxes
  from .spots_detector import SpotsDetector
  from .zoom import Zoom


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
  """Includes public exports without loading their implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
