# coding: utf-8

from time import sleep
from typing import Optional
import logging
from multiprocessing import current_process

from .meta_actuator import MetaActuator


class Actuator(metaclass=MetaActuator):
  """The base class for all Actuator classes, allowing to keep track of them
  and defining methods shared by all of them.

  The Actuator objects are helper classes used by the
  :class:`~crappy.blocks.Machine` Block to communicate with motors or other
  actuators.

  .. versionadded:: 1.4.0
  """

  ft232h: bool = False

  def __init__(self, *_, **__) -> None:
    """Initializes the instance attributes."""

    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Actuator.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.

    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self) -> None:
    """This method should perform any action that's required for initializing
    the hardware and the communication with it.

    Communication with hardware should be avoided in the :meth:`__init__`
    method, and this method is where it should start. This method is called
    after Crappy's processes start, i.e. when the associated
    :class:`~crappy.blocks.Machine` already runs separately from all the other
    Blocks.

    It is fine for this method not to perform anything.

    .. versionadded:: 2.0.0
    """

    ...

  def set_speed(self, speed: float) -> None:
    """This method should drive the actuator so that it reaches the desired
    speed.

    It is used if the ``mode`` given in the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block is ``'speed'``.

    The value passed as the ``speed`` argument will be that received over
    the ``cmd_label`` given in the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block.

    Args:
      speed: The speed to reach, as a :obj:`float`.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, f"The set_speed method was called but is not "
                              f"defined ! No command sent to the actuator.")
    sleep(1)

  def set_position(self, position: float, speed: Optional[float]) -> None:
    """This method should drive the actuator so that it reaches the desired
    position.

    It is used if the ``mode`` given in the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block is ``'position'``.

    The value passed as the ``position`` argument will be that received over
    the ``cmd_label`` given in the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block.

    The value passed as the ``speed`` argument will be either :

    * :obj:`None` if no ``speed`` nor ``speed_cmd_label`` were specified in the
      ``actuators`` argument of the :class:`~crappy.blocks.Machine` Block.

    * The value given with the ``speed`` key of the ``actuators`` argument
      of the :class:`~crappy.blocks.Machine` Block, if no other speed command
      was received in the meantime.

    * The last value received over the ``speed_cmd_label`` if it was set in the
      ``actuators`` argument of the :class:`~crappy.blocks.Machine` Block.
      Before the first speed command is received, the value will either be
      :obj:`None` if no ``speed`` was specified, else the value given as
      ``speed`` in the ``actuators`` argument of the
      :class:`~crappy.blocks.Machine` Block.

    Important:
      The ``speed`` value might be :obj:`None`, but it is not optional ! When
      writing a custom Actuator, make sure to always handle it.

    Args:
      position: The position to reach, as a :obj:`float`.
      speed: The speed at which to move to the desired position, as a
        :obj:`float`, or :obj:`None` if no speed is specified.

        .. versionchanged:: 2.0.0
          *speed* is now a mandatory argument even if it is :obj:`None`

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, f"The set_position method was called but is not "
                              f"defined ! No command sent to the actuator.")
    sleep(1)

  def get_speed(self) -> Optional[float]:
    """This method should return the current speed of the actuator, as a
    :obj:`float`.

    This speed will be sent to downstream Blocks, over the label given with
    the ``speed_label`` key of the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block.

    It is also fine for this method to return :obj:`None` if the speed could
    not be acquired.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, f"The get_speed method as called but is not "
                              f"defined ! Define such a method, don't set the "
                              f"speed_label, or remove the output links of "
                              f"this block.")
    sleep(1)
    return

  def get_position(self) -> Optional[float]:
    """This method should return the current position of the actuator, as a
    :obj:`float`.

    This position will be sent to downstream Blocks, over the label given with
    the ``position_label`` key of the ``actuators`` argument of the
    :class:`~crappy.blocks.Machine` Block.

    It is also fine for this method to return :obj:`None` if the position could
    not be acquired.

    .. versionadded:: 1.5.10
    """

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

    .. versionadded:: 2.0.0
    """

    self.set_speed(0)

  def close(self) -> None:
    """This method should perform any action required for properly ending the
    test and closing the communication with hardware.

    It will be called when the associated :class:`~crappy.blocks.Machine`
    receives the order to stop (usually because the user hit `CTRL+C`, or
    because a :class:`~crappy.blocks.Generator` Block reached the end of its
    path, or because an exception was raised in any of the Blocks).

    It is fine for this method not to perform anything.

    .. versionadded:: 2.0.0
    """

    ...
