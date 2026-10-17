# coding: utf-8

""":class:`~crappy.camera.meta_camera.camera.Camera` configuration models, GUI
backends, and selection tools.

The base package contains abstract configuration classes shared by the
'tkinter' and 'pyqt' packages. The possible configuration windows are selected
by :class:`~crappy.blocks.vision.CameraSource` and the all-in-one
:class:`Camera Blocks <crappy.blocks.Camera>` during preparation. config_tools
contains the boxes, spot detector, zoom model, and histogram worker they use.
"""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'CameraConfig': ('.base', 'CameraConfig'),
            'Box': ('.config_tools', 'Box'),
            'Overlay': ('.config_tools', 'Overlay'),
            'SpotsBoxes': ('.config_tools', 'SpotsBoxes'),
            'SpotsDetector': ('.config_tools', 'SpotsDetector'),
            'Zoom': ('.config_tools', 'Zoom'),
            'TkinterCameraConfig': ('.tkinter', 'TkinterCameraConfig'),
            'TkinterCameraConfigBoxes': ('.tkinter',
                                         'TkinterCameraConfigBoxes'),
            'TkinterDICVEConfig': ('.tkinter', 'TkinterDICVEConfig'),
            'TkinterDISCorrelConfig': ('.tkinter', 'TkinterDISCorrelConfig'),
            'TkinterVideoExtensoConfig': ('.tkinter',
                                          'TkinterVideoExtensoConfig'),
            'create_configurator': ('.factory', 'create_configurator')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from .base import CameraConfig
  from .config_tools import Box, Overlay, SpotsBoxes, SpotsDetector, Zoom
  from .tkinter import (TkinterCameraConfig, TkinterCameraConfigBoxes,
                        TkinterDICVEConfig, TkinterDISCorrelConfig,
                        TkinterVideoExtensoConfig)
  from .factory import create_configurator


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
