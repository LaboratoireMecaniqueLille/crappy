# coding: utf-8

from time import sleep
from typing import Optional

from .._global import DefinitionError


class MetaActuator(type):
  """Metaclass ensuring that two Actuators don't have the same name, and that
  all Actuators define the required methods. Also keeps track of all the
  Actuator classes, including the custom user-defined ones."""

  classes = {}
  needed_methods = ['open', 'stop', 'close']

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Gathering all the defined methods
    defined_methods = list(dct.keys())
    defined_methods += [base.__dict__.keys() for base in bases]

    # Checking for missing methods
    missing_methods = [meth for meth in cls.needed_methods
                       if meth not in defined_methods]

    # Raising if there are unexpected missing methods
    if missing_methods and name != "Actuator":
      raise DefinitionError(
        f'Class {name} is missing the required method(s): '
        f'{", ".join(missing_methods)}')

    # Otherwise, saving the class
    if name != 'Actuator':
      cls.classes[name] = cls


class Actuator(metaclass=MetaActuator):
  """The base class for all actuator classes, allowing to keep track of them
  and defining methods shared by all of them."""

  def set_speed(self, speed: float) -> None:
    """This method should drive the actuator so that it reaches the desired
    speed."""

    print(f"WARNING ! Trying to drive the Actuator {type(self).__name__} in "
          f"speed but it does not define a set_speed method !\nNo command "
          f"sent to the actuator.")
    sleep(1)
    return

  def set_position(self,
                   position: float,
                   speed: Optional[float] = None) -> None:
    """This method should drive the actuator so that it reaches the desired
    position. A speed value can optionally be provided for specifying the speed
    at which the actuator should move for getting to the desired position."""

    print(f"WARNING ! Trying to drive the Actuator {type(self).__name__} in "
          f"position but it does not define a set_position method !\n"
          f"No command sent to the actuator.")
    sleep(1)
    return

  def get_speed(self) -> Optional[float]:
    """This method should return the current speed of the actuator. It is also
    fine for this method to return :obj:`None`."""

    print(f"WARNING ! Trying to get the speed of the Actuator "
          f"{type(self).__name__}, but it does not define a get_speed "
          f"method !\nDefine such a method, don't set the speed_label, or "
          f"remove the output links of this block.")
    sleep(1)
    return

  def get_position(self) -> Optional[float]:
    """This method should return the current position of the actuator. It is
    also fine for this method to return :obj:`None`."""

    print(f"WARNING ! Trying to get the position of the Actuator "
          f"{type(self).__name__}, but it does not define a get_position "
          f"method !\nDefine such a method, don't set the pos_label, or remove"
          f" the output links of this block.")
    sleep(1)
    return
