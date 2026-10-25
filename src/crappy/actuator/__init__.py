# coding: utf-8

"""Per-driver exports and discovery without importing unused implementations.

The registry dictionary contains only loaded classes. Internal selection and
discovery use the same declared export table.
"""

from importlib import import_module as _import_module
from importlib.util import resolve_name as _resolve_name
from typing import Any as _Any, TYPE_CHECKING as _TYPE_CHECKING

_EXPORTS = {'FakeDCMotor': ('.fake_dc_motor', 'FakeDCMotor'),
            'FakeStepperMotor': ('.fake_stepper_motor', 'FakeStepperMotor'),
            'JVLMac140': ('.jvl_mac_140', 'JVLMac140'),
            'ServoStar300': ('.kollmorgen_servostar_300', 'ServoStar300'),
            'Phidget4AStepper': ('.phidgets_stepper4a', 'Phidget4AStepper'),
            'PololuTic': ('.pololu_tic', 'PololuTic')}

# LaMcube classes are also selectable by actuator name
from .. import _validation as check
from ..lamcube import _EXPORTS as _LAB_EXPORTS
_DRIVERS = {**_EXPORTS,
            **{name: ('..lamcube' + module, attribute)
               for name, (module, attribute) in _LAB_EXPORTS.items()}}

from .meta_actuator import Actuator
from ._collection import moved_to_collection

actuator_dict: dict[str, type[Actuator]] = Actuator.classes
__all__ = (*tuple(_EXPORTS),
           'Actuator', 'actuator_dict',
           'moved_to_collection')

if _TYPE_CHECKING:
  from .fake_dc_motor import FakeDCMotor as FakeDCMotor
  from .fake_stepper_motor import FakeStepperMotor as FakeStepperMotor
  from .jvl_mac_140 import JVLMac140 as JVLMac140
  from .kollmorgen_servostar_300 import ServoStar300 as ServoStar300
  from .phidgets_stepper4a import Phidget4AStepper as Phidget4AStepper
  from .pololu_tic import PololuTic as PololuTic


def actuator_names() -> set[str]:
  """Lists declared built-ins and loaded custom classes without imports.

  .. versionadded:: 2.2.0
  """

  return set(_DRIVERS) | set(actuator_dict)


def resolve_actuator(name: str) -> type[Actuator]:
  """Returns a registered class or imports only the requested built-in.

  Unknown names raise KeyError, import failures propagate unchanged.

  .. versionadded:: 2.2.0
  """

  check.string(name, 'name')

  if name in actuator_dict:
    return actuator_dict[name]
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
  value = resolve_actuator(name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """Includes declared exports without importing implementations.

  .. versionadded:: 2.2.0
  """

  return sorted(set(globals()) | set(__all__))
