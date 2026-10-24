# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'CameraProcess': ('.camera_process', 'CameraProcess'),
            'DISCorrelProcess': ('.dis_correl', 'DISCorrelProcess'),
            'Displayer': ('.display', 'Displayer'),
            'DICVEProcess': ('.dic_ve', 'DICVEProcess'),
            'GPUCorrelProcess': ('.gpu_correl', 'GPUCorrelProcess'),
            'GPUVEProcess': ('.gpu_ve', 'GPUVEProcess'),
            'ImageSaver': ('.record', 'ImageSaver'),
            'VideoExtensoProcess': ('.video_extenso', 'VideoExtensoProcess')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .camera_process import CameraProcess as CameraProcess
  from .dis_correl import DISCorrelProcess as DISCorrelProcess
  from .display import Displayer as Displayer
  from .dic_ve import DICVEProcess as DICVEProcess
  from .gpu_correl import GPUCorrelProcess as GPUCorrelProcess
  from .gpu_ve import GPUVEProcess as GPUVEProcess
  from .record import ImageSaver as ImageSaver
  from .video_extenso import VideoExtensoProcess as VideoExtensoProcess


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
