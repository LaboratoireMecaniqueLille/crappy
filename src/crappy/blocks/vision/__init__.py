# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'VisionBlock': ('.block', 'VisionBlock'),
            'CameraSource': ('.camera', 'CameraSource'),
            'DICVEProcessor': ('.dic_ve', 'DICVEProcessor'),
            'DISCorrelProcessor': ('.dis_correl', 'DISCorrelProcessor'),
            'ImageDisplayer': ('.display', 'ImageDisplayer'),
            'ImageRecorder': ('.record', 'ImageRecorder'),
            'VideoExtensoProcessor': ('.video_extenso',
                                      'VideoExtensoProcessor')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .block import VisionBlock as VisionBlock
  from .camera import CameraSource as CameraSource
  from .dic_ve import DICVEProcessor as DICVEProcessor
  from .dis_correl import DISCorrelProcessor as DISCorrelProcessor
  from .display import ImageDisplayer as ImageDisplayer
  from .record import ImageRecorder as ImageRecorder
  from .video_extenso import VideoExtensoProcessor as VideoExtensoProcessor


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
