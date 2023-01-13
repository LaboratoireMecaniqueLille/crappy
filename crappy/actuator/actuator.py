# coding: utf-8

from time import sleep
from typing import Optional
import logging
from multiprocessing import current_process

from .._global import DefinitionError


class MetaActuator(type):
  """Metaclass ensuring that two Actuators don't have the same name, and that
  all Actuators define the required methods. Also keeps track of all the
  Actuator classes, including the custom user-defined ones."""

  classes = {}

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Otherwise, saving the class
    if name != 'Actuator':
      cls.classes[name] = cls


class Actuator(metaclass=MetaActuator):
  """The base class for all actuator classes, allowing to keep track of them
  and defining methods shared by all of them."""

  ft232h: bool = False

  def __init__(self, *_, **__) -> None:
    """"""

    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self) -> None:
    """This method should initialize the connection to the actuator, and
    configure the actuator."""

    ...

  def set_speed(self, speed: float) -> None:
    """This method should drive the actuator so that it reaches the desired
    speed."""

    self.log(logging.WARNING, f"The set_speed method was called but is not "
                              f"defined ! No command sent to the actuator.")
    sleep(1)

  def set_position(self,
                   position: float,
                   speed: Optional[float] = None) -> None:
    """This method should drive the actuator so that it reaches the desired
    position. A speed value can optionally be provided for specifying the speed
    at which the actuator should move for getting to the desired position."""

    self.log(logging.WARNING, f"The set_position method was called but is not "
                              f"defined ! No command sent to the actuator.")
    sleep(1)

  def get_speed(self) -> Optional[float]:
    """This method should return the current speed of the actuator. It is also
    fine for this method to return :obj:`None`."""

    self.log(logging.WARNING, f"The get_speed method as called but is not "
                              f"defined ! Define such a method, don't set the "
                              f"speed_label, or remove the output links of "
                              f"this block.")
    sleep(1)
    return

  def get_position(self) -> Optional[float]:
    """This method should return the current position of the actuator. It is
    also fine for this method to return :obj:`None`."""

    self.log(logging.WARNING, f"The get_position method as called but is not "
                              f"defined ! Define such a method, don't set the "
                              f"position_label, or remove the output links of "
                              f"this block.")
    sleep(1)
    return

  def stop(self) -> None:
    """This method should stop all movements of the actuator, and if possible
    make sure that the actuator cannot move anymore.

    That includes for example de-energizing the actuator, or switching it to a
    locked state.

    This method will only be called once at the very end of the test. The
    default behavior is to call :meth:`set_speed` to set the speed to `0`. This
    method doesn't need to be overriden if the actuator doesn't have any
    feature for stopping other than speed control.
    """

    self.set_speed(0)

  def close(self) -> None:
    """This method should perform any action required for properly closing the
    connection to the actuator."""

    ...
