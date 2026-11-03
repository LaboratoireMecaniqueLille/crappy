# coding: utf-8

"""Tkinter camera configuration windows."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'TkinterCameraConfig': ('.camera_config', 'TkinterCameraConfig'),
            'TkinterCameraConfigBoxes': ('.camera_config_boxes',
                                         'TkinterCameraConfigBoxes'),
            'TkinterDICVEConfig': ('.dic_ve_config', 'TkinterDICVEConfig'),
            'TkinterDISCorrelConfig': ('.dis_correl_config',
                                       'TkinterDISCorrelConfig'),
            'TkinterVideoExtensoConfig': ('.video_extenso_config',
                                          'TkinterVideoExtensoConfig')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .camera_config import TkinterCameraConfig
  from .camera_config_boxes import TkinterCameraConfigBoxes
  from .dic_ve_config import TkinterDICVEConfig
  from .dis_correl_config import TkinterDISCorrelConfig
  from .video_extenso_config import TkinterVideoExtensoConfig


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
