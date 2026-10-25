# coding: utf-8

"""Per-driver exports and discovery without importing unused implementations.

The registry dictionary contains only loaded classes. Internal selection and
discovery use the same declared export table.
"""

from importlib import import_module as _import_module
from importlib.util import resolve_name as _resolve_name
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'ADS1115': ('.ads1115', 'ADS1115'),
            'DAQmx': ('.daqmx', 'DAQmx'),
            'FakeInOut': ('.fake_inout', 'FakeInOut'),
            'GPIOPWM': ('.gpio_pwm', 'GPIOPWM'),
            'GPIOSwitch': ('.gpio_switch', 'GPIOSwitch'),
            'LabjackT7': ('.labjack_t7', 'LabjackT7'),
            'T7Streamer': ('.labjack_t7_streamer', 'T7Streamer'),
            'MPRLS': ('.mprls', 'MPRLS'),
            'NAU7802': ('.nau7802', 'NAU7802'),
            'NIDAQmx': ('.ni_daqmx', 'NIDAQmx'),
            'PhidgetWheatstoneBridge': ('.phidgets_wheatstone_bridge',
                                        'PhidgetWheatstoneBridge')}
_DRIVERS = _EXPORTS

from .. import _validation as check
from .meta_inout import InOut
from ._collection import moved_to_collection

inout_dict: dict[str, type[InOut]] = InOut.classes
__all__ = (*tuple(_EXPORTS),
           'InOut', 'inout_dict',
           'moved_to_collection')

if _TYPE_CHECKING:
  from .ads1115 import ADS1115 as ADS1115
  from .daqmx import DAQmx as DAQmx
  from .fake_inout import FakeInOut as FakeInOut
  from .gpio_pwm import GPIOPWM as GPIOPWM
  from .gpio_switch import GPIOSwitch as GPIOSwitch
  from .labjack_t7 import LabjackT7 as LabjackT7
  from .labjack_t7_streamer import T7Streamer as T7Streamer
  from .mprls import MPRLS as MPRLS
  from .nau7802 import NAU7802 as NAU7802
  from .ni_daqmx import NIDAQmx as NIDAQmx
  from .phidgets_wheatstone_bridge import (PhidgetWheatstoneBridge
                                           as PhidgetWheatstoneBridge)


def inout_names() -> set[str]:
  """Lists declared built-ins and loaded custom classes without imports.

  .. versionadded:: 2.2.0
  """

  return set(_DRIVERS) | set(inout_dict)


def resolve_inout(name: str) -> type[InOut]:
  """Returns a registered class or imports only the requested built-in.

  Unknown names raise KeyError; import failures propagate unchanged.

  .. versionadded:: 2.2.0
  """

  check.string(name, 'name')

  if name in inout_dict:
    return inout_dict[name]
  module, attribute = _DRIVERS[name]
  return getattr(_import_module(module, __name__), attribute)


def is_reserved(cls: type) -> bool:
  """Rejects custom subclasses taking an unloaded shipped driver's name.

  .. versionadded:: 2.2.0
  """

  check.instance(cls, 'cls', type)

  reference = _DRIVERS.get(cls.__name__)
  if reference is None:
    return False

  module = _resolve_name(reference[0], __name__)
  return cls.__module__ != module


def __getattr__(name: str) -> _Any:
  """Loads and caches one public driver export on first access.

  .. versionadded:: 2.2.0
  """

  if name not in _EXPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  value = resolve_inout(name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
