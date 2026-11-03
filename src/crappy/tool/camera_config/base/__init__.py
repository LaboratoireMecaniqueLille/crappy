# coding: utf-8

"""Abstract camera configurations shared by all the GUI implementations."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'ConfigurationLifecycle': ('._configuration_lifecycle',
                                       'ConfigurationLifecycle'),
            'CameraConfig': ('.camera_config', 'CameraConfig'),
            'ConfigAction': ('.camera_config', 'ConfigAction'),
            'CameraConfigBoxes': ('.camera_config_boxes', 'CameraConfigBoxes'),
            'DICVEConfig': ('.dic_ve_config', 'DICVEConfig'),
            'DISCorrelConfig': ('.dis_correl_config', 'DISCorrelConfig'),
            'VideoExtensoConfig': ('.video_extenso_config',
                                   'VideoExtensoConfig')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from ._configuration_lifecycle import ConfigurationLifecycle
  from .camera_config import CameraConfig, ConfigAction
  from .camera_config_boxes import CameraConfigBoxes
  from .dic_ve_config import DICVEConfig
  from .dis_correl_config import DISCorrelConfig
  from .video_extenso_config import VideoExtensoConfig


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
