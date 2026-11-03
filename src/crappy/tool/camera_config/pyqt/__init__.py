# coding: utf-8

"""PyQt6 camera configuration windows."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'PyQtCameraConfig': ('.camera_config', 'PyQtCameraConfig'),
            'PyQtCameraConfigBoxes': ('.camera_config_boxes',
                                      'PyQtCameraConfigBoxes'),
            'PyQtDICVEConfig': ('.dic_ve_config', 'PyQtDICVEConfig'),
            'PyQtDISCorrelConfig': ('.dis_correl_config', 'PyQtDISCorrelConfig'),
            'PyQtVideoExtensoConfig': ('.video_extenso_config',
                                       'PyQtVideoExtensoConfig')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .camera_config import PyQtCameraConfig
  from .camera_config_boxes import PyQtCameraConfigBoxes
  from .dic_ve_config import PyQtDICVEConfig
  from .dis_correl_config import PyQtDISCorrelConfig
  from .video_extenso_config import PyQtVideoExtensoConfig


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
