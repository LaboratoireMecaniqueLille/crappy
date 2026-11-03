# coding: utf-8

"""Public exports, resolved individually on first access."""

from importlib import import_module as _import_module
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'bindings': ('.bindings', None),
            'camera_config': ('.camera_config', None),
            'image_processing': ('.image_processing', None),
            'ApplyStrainToImage': ('.apply_strain_image',
                                   'ApplyStrainToImage')}
__all__ = tuple(_EXPORTS)

if _TYPE_CHECKING:
  from . import bindings as bindings
  from . import camera_config as camera_config
  from . import image_processing as image_processing
  from .apply_strain_image import ApplyStrainToImage as ApplyStrainToImage


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
