# coding: utf-8

"""Per-driver exports and discovery without importing unused implementations.

The registry dictionary contains only loaded classes. Internal selection and
discovery use the same declared export table.
"""

from importlib import import_module as _import_module
from importlib.util import resolve_name as _resolve_name
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'FakeCamera': ('.fake_camera', 'FakeCamera'),
            'FileReader': ('.file_reader', 'FileReader'),
            'Webcam': ('.opencv_camera_webcam', 'Webcam'),
            'RaspberryPiCamera2': ('.raspberry_pi_camera_2',
                                   'RaspberryPiCamera2'),
            'XiAPI': ('.ximea_xiapi', 'XiAPI'),
            'CameraGstreamer': ('.gstreamer_camera_basic', 'CameraGstreamer'),
            'CameraOpencv': ('.opencv_camera_basic', 'CameraOpencv')}
_DRIVERS = _EXPORTS

from .meta_camera import Camera, camera_setting
from ._collection import moved_to_collection

camera_dict: dict[str, type[Camera]] = Camera.classes
__all__ = (*tuple(_EXPORTS),
           'Camera', 'camera_dict',
           'camera_setting',
           'moved_to_collection', )

if _TYPE_CHECKING:
  from .fake_camera import FakeCamera as FakeCamera
  from .file_reader import FileReader as FileReader
  from .opencv_camera_webcam import Webcam as Webcam
  from .raspberry_pi_camera_2 import RaspberryPiCamera2 as RaspberryPiCamera2
  from .ximea_xiapi import XiAPI as XiAPI
  from .gstreamer_camera_basic import CameraGstreamer as CameraGstreamer
  from .opencv_camera_basic import CameraOpencv as CameraOpencv

_v4l2_available: bool | None = None


def _driver_module(name: str) -> str:
  """Selects the appropriate base or v4l2 camera variant depending on driver
  availability.

  .. versionadded:: 2.2.0
  """

  global _v4l2_available
  module, _ = _DRIVERS[name]
  if name in ('CameraGstreamer', 'CameraOpencv'):
    if _v4l2_available is None:
      from platform import system
      from subprocess import run
      if system() == 'Linux':
        try:
          run(['v4l2-ctl'], capture_output=True)
        except FileNotFoundError:
          _v4l2_available = False
        else:
          _v4l2_available = True
      else:
        _v4l2_available = False
    if _v4l2_available:
      module = module.removesuffix('_basic') + '_v4l2'
  return module


def camera_names() -> set[str]:
  """Lists declared built-ins and loaded custom classes without imports.

  .. versionadded:: 2.2.0
  """

  return set(_DRIVERS) | set(camera_dict)


def resolve_camera(name: str) -> type[Camera]:
  """Returns a registered class or imports only the requested built-in.

  Unknown names raise KeyError, import failures propagate unchanged.

  .. versionadded:: 2.2.0
  """

  if name in camera_dict:
    return camera_dict[name]
  module, attribute = _DRIVERS[name]
  module = _driver_module(name)
  return getattr(_import_module(module, __name__), attribute)


def is_reserved(cls: type) -> bool:
  """Rejects custom subclasses taking an unloaded shipped driver's name.

  .. versionadded:: 2.2.0
  """

  reference = _DRIVERS.get(cls.__name__)
  if reference is None:
    return False

  module = _resolve_name(reference[0], __name__)
  # Both variants are reserved
  if cls.__name__ in ('CameraGstreamer', 'CameraOpencv'):
    return cls.__module__ not in (module,
                                  module.removesuffix('_basic') + '_v4l2')
  return cls.__module__ != module


def __getattr__(name: str) -> _Any:
  """Loads and caches one public driver export on first access.

  .. versionadded:: 2.2.0
  """

  if name not in _EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = resolve_camera(name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
