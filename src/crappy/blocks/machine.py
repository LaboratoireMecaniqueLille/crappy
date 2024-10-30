# coding: utf-8

from time import time
from typing import Any, Optional
from collections.abc import Iterable
from dataclasses import dataclass, fields
import logging

from .meta_block import Block
from ..actuator import actuator_dict, Actuator, deprecated_actuators
from ..tool.ft232h import USBServer


@dataclass
class ActuatorInstance:
  """This class holds all the information that can be associated to an
  Actuator."""

  actuator: Actuator
  speed: Optional[float] = None
  position_label: Optional[str] = None
  speed_label: Optional[str] = None
  mode: str = 'speed'
  cmd_label: str = 'cmd'
  speed_cmd_label: Optional[str] = None


class Machine(Block):
  """This Block is meant to drive one or several
  :class:`~crappy.actuator.Actuator`. It can set speed or position commands on
  hardware actuators.

  The possibility to drive several Actuators from a unique Block is given so
  that they can be driven in a synchronized way. If synchronization is not
  needed, it is preferable to drive the Actuators from separate Machine Blocks.

  This Block takes the speed or position commands for the Actuators  as inputs,
  and can optionally read and output the current speed and/or positions of the
  Actuators. The speed and position commands are set respectively by calling
  the :meth:`~crappy.actuator.Actuator.set_position` and
  :meth:`~crappy.actuator.Actuator.set_speed` methods of the Actuators, and the
  current speed and position values are acquired by calling the
  :meth:`~crappy.actuator.Actuator.get_position` and
  :meth:`~crappy.actuator.Actuator.get_speed` methods of the Actuators.

  It is possible to tune for each Actuator the label over which it receives its
  commands, and optionally the labels over which it sends its current speed
  and/or position. The driving mode (`'speed'` or `'position'`) can also be set
  independently for each Actuator.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               actuators: Iterable[dict[str, Any]],
               common: Optional[dict[str, Any]] = None,
               time_label: str = 't(s)',
               ft232h_ser_num: Optional[str] = None,
               spam: bool = False,
               freq: Optional[float] = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      actuators: An iterable (like a :obj:`list` or a :obj:`tuple`) of all the
        :class:`~crappy.actuator.Actuator` this Block needs to drive. It 
        contains one :obj:`dict` for every Actuator, with mandatory and 
        optional keys. The keys providing information on how to drive the 
        Actuator are listed below. Any other unrecognized key will be passed to 
        the Actuator as argument when instantiating it.
      common: The keys of this :obj:`dict` will be common to all the Actuators.
        If one key conflicts with an existing key for an Actuator, the common 
        one will prevail.
      time_label: If reading speed or position from one or more Actuators, the
        time information will be carried by this label.
      ft232h_ser_num: Serial number of the FT232H device to use for driving
        the controlled Actuator.
        
        .. versionadded:: 2.0.0
      spam: If :obj:`True`, a command is sent to the Actuators at each loop of
        the Block, else it is sent every time a new command is received.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0

    Note:
      - ``actuators`` keys:

        - ``type``: The name of the :class:`~crappy.actuator.Actuator` class to 
          instantiate. This key is mandatory.
        - ``cmd_label``: The label carrying the command for driving the
          Actuator. It defaults to `'cmd'`.
        - ``mode``: Can be either `'speed'` or `'position'`. Either
          :meth:`~crappy.actuator.Actuator.set_speed` or
          :meth:`~crappy.actuator.Actuator.set_position` is called to drive the
          Actuator, depending on the selected mode. When driven in `'position'` 
          mode, the speed of the actuator can also be adjusted, see the 
          ``speed`` and ``speed_cmd_label`` keys. The default mode is 
          `'speed'`.
        - ``speed``: If mode is `'position'`, the speed at which the Actuator
          should move. This speed is passed as second argument to the
          :meth:`~crappy.actuator.Actuator.set_position` method of the
          Actuator. If the ``speed_cmd_label`` key is not specified, this speed
          will remain the same for the entire test. This key is not mandatory.
        - ``position_label``: If given, the Block will return the value of
          :meth:`~crappy.actuator.Actuator.get_position` under this label. This
          key is not mandatory.
        - ``speed_label``: If given, the Block will return the value of
          :meth:`~crappy.actuator.Actuator.get_speed` under this label. This
          key is not mandatory.
        - ``speed_cmd_label``: The label carrying the speed to set when driving
          in `'position'` mode. Each time a value is received, the stored speed
          value is updated. It will also overwrite the ``speed`` key if given.
    """

    self._actuators: list[ActuatorInstance] = list()
    self._ft232h_args = None

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._time_label = time_label
    self._spam = spam

    # No extra information to add to the main dicts
    if common is None:
      common = dict()

    # Updating the settings with the common information
    for actuator in actuators:
      actuator |= common

    # Making sure all the dicts contain the 'type' key
    if not all('type' in dic for dic in actuators):
      raise ValueError("The 'type' key must be provided for all the "
                       "actuators !")

    # The names of the possible settings, to avoid typos and reduce verbosity
    actuator_settings = [field.name for field in fields(ActuatorInstance)
                         if field.type is not Actuator]

    # The list of all the Actuator types to instantiate
    self._types = [actuator['type'] for actuator in actuators]

    # Checking for deprecated names
    deprecated = [type_ for type_ in self._types
                  if type_ in deprecated_actuators]
    for type_ in deprecated:
      raise NotImplementedError(
          f"The {type_} Actuator was deprecated in version 2.0.0, and renamed "
          f"to {deprecated_actuators[type_]} ! Please update your code "
          f"accordingly and check the documentation for more information")

    # Checking that all the given actuators are valid
    if not all(type_ in actuator_dict for type_ in self._types):
      unknown = ', '.join(tuple(type_ for type_ in self._types if type_
                          not in actuator_dict))
      possible = ', '.join(sorted(actuator_dict.keys()))
      raise ValueError(f"Unknown actuator type(s) : {unknown} ! "
                       f"The possible types are : {possible}")

    # The settings that won't be passed to the Actuator objects
    self._settings = [{key: value for key, value in actuator.items()
                       if key in actuator_settings}
                      for actuator in actuators]

    # The settings that will be passed as kwargs to the Actuator objects
    self._actuators_kw = [{key: value for key, value in actuator.items()
                           if key not in ('type', *actuator_settings)}
                          for actuator in actuators]

    # Checking whether the Actuators communicate through an FT232H
    if any(actuator_dict[type_].ft232h for type_ in self._types):
      self._ft232h_args = USBServer.register(ft232h_ser_num)

  def prepare(self) -> None:
    """Checks the validity of the linking and initializes all the Actuator
    objects to drive.

    This method calls the :meth:`~crappy.actuator.Actuator.open` method of each
    Actuator.
    """

    # Instantiating the actuators and storing them
    self._actuators = [ActuatorInstance(
      actuator=actuator_dict[type_](**actuator_kw)
      if not actuator_dict[type_].ft232h else
      actuator_dict[type_](**actuator_kw, _ft232h_args=self._ft232h_args),
      **setting)
      for type_, setting, actuator_kw in zip(self._types,
                                             self._settings,
                                             self._actuators_kw)]

    # Checking the consistency of the linking
    if not self.inputs and not self.outputs:
      raise IOError("The Machine block isn't linked to any other block !")

    # Opening each actuator
    for actuator in self._actuators:
      self.log(logging.INFO, f"Opening the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.open()
      self.log(logging.INFO, f"Opened the {type(actuator.actuator).__name__}"
                             f"Actuator")

  def loop(self) -> None:
    """Sets the received position and speed commands, and reads the current 
    speed and position from the :class:`~crappy.actuator.Actuator`.
    
    For each Actuator, a command is set **only** if a new one was received or 
    if the ``spam`` argument is :obj:`True`. It is set using either 
    :meth:`~crappy.actuator.Actuator.set_position` or
    :meth:`~crappy.actuator.Actuator.set_speed` depending on the selected
    driving mode.
    
    For each Actuator, a speed and/or position value is read **only** if the 
    ``speed_label`` and/or the ``position_label`` was set. If so, these values
    are read at each loop and sent to downstream Blocks over the given labels.
    This is independent of the chosen driving mode. The
    :meth:`~crappy.actuator.Actuator.get_position` and
    :meth:`~crappy.actuator.Actuator.get_speed` are called for acquiring the
    position and speed values respectively.
    """

    # Iterating over the actuators for setting the commands
    if recv := self.recv_last_data(fill_missing=self._spam):
      for actuator in self._actuators:
        # Setting the speed attribute if it was received
        if (actuator.speed_cmd_label is not None
            and actuator.speed_cmd_label in recv):
          self.log(logging.DEBUG,
                   f"Updating the speed of the "
                   f"{type(actuator.actuator).__name__} Actuator from "
                   f"{actuator.speed} to {recv[actuator.speed_cmd_label]}")
          actuator.speed = recv[actuator.speed_cmd_label]

        # Setting only the commands that were received
        if actuator.cmd_label in recv:
          # Setting the speed command
          if actuator.mode == 'speed':
            self.log(logging.DEBUG,
                     f"Setting speed of the {type(actuator.actuator).__name__}"
                     f" Actuator to {recv[actuator.cmd_label]}")
            actuator.actuator.set_speed(recv[actuator.cmd_label])
          # Setting the position command
          else:
            actuator.actuator.set_position(recv[actuator.cmd_label],
                                           actuator.speed)
            self.log(
              logging.DEBUG,
              f"Setting position of the {type(actuator.actuator).__name__} "
              f"Actuator to {recv[actuator.cmd_label]} with speed "
              f"{actuator.speed}")

    to_send = {}

    # Iterating over the actuators to get the speeds and the positions
    for actuator in self._actuators:
      if actuator.position_label is not None:
        position = actuator.actuator.get_position()
        if position is not None:
          to_send[actuator.position_label] = position
      if actuator.speed_label is not None:
        speed = actuator.actuator.get_speed()
        if speed is not None:
          to_send[actuator.speed_label] = speed

    # Sending the speed and position values if any
    if to_send:
      to_send[self._time_label] = time() - self.t0
      self.send(to_send)

  def finish(self) -> None:
    """Stops and closes all the Actuators to drive.

    This method calls the :meth:`~crappy.actuator.Actuator.stop` and
    :meth:`~crappy.actuator.Actuator.close` method of each Actuator.
    """

    for actuator in self._actuators:
      self.log(logging.INFO, f"Stopping the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.stop()
    for actuator in self._actuators:
      self.log(logging.INFO, f"Closing the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.close()
      self.log(logging.INFO, f"Closed the {type(actuator.actuator).__name__}"
                             f"Actuator")
