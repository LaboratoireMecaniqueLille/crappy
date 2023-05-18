# coding: utf-8

from time import time
from typing import Dict, List, Any, Optional, Iterable
from dataclasses import dataclass, fields
import logging

from .meta_block import Block
from ..actuator import actuator_dict, Actuator
from ..tool.ft232h import USBServer


@dataclass
class ActuatorInstance:
  """"""

  actuator: Actuator
  speed: Optional[float] = None
  position_label: Optional[str] = None
  speed_label: Optional[str] = None
  mode: str = 'speed'
  cmd_label: str = 'cmd'
  speed_cmd_label: Optional[str] = None


class Machine(Block):
  """This block is meant to drive one or several :ref:`Actuators`.

  The possibility to drive several Actuators from a unique block is given so
  that they can be driven in a synchronized way. If synchronization is not
  needed, it is preferable to drive the Actuators from separate Machine blocks.
  """

  def __init__(self,
               actuators: Iterable[Dict[str, Any]],
               common: Optional[Dict[str, Any]] = None,
               time_label: str = 't(s)',
               ft232h_ser_num: Optional[str] = None,
               spam: bool = False,
               freq: Optional[float] = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      actuators: An iterable (like a :obj:`list` or a :obj:`tuple`) of all the
        :ref:`Actuators` this block needs to drive. It contains one :obj:`dict`
        for every Actuator, with mandatory and optional keys. The keys
        providing information on how to drive the Actuator are listed below.
        Any other key will be passed to the Actuator object as argument when
        instantiating it.
      common: The keys of this :obj:`dict` will be common to all the Actuators.
        If it conflicts with an existing key for an Actuator, the common one
        will prevail.
      time_label: If reading speed or position from one or more Actuators, the
        time information will be carried by this label.
      spam: If :obj:`True`, a command is sent to the Actuators on each loop of
        the block, else it is sent every time a new command is received.
      freq: The block will try to loop at this frequency.
      display_freq: If :obj:`True`, displays the looping frequency of the
        block.

    Note:
      - ``actuators`` keys:

        - ``type``: The name of the Actuator class to instantiate. This key is
          mandatory.
        - ``cmd_label``: The label carrying the command for driving the
          Actuator. It defaults to `'cmd'`.
        - ``mode``: Can be either `'speed'` or `'position'`. Will either call
          :meth:`set_speed` or :meth:`set_position` to drive the actuator. When
          driven in `'position'` mode, the speed of the actuator can also be
          adjusted, see the ``speed_cmd_label`` key. The default mode is
          `'speed'`.
        - ``speed``: If mode is `'position'`, the speed at which the Actuator
          should move. This speed is passed as second argument to the
          :meth:`set_position` method of the Actuator. If the
          ``speed_cmd_label`` key is not specified, this speed will remain the
          same for the entire test. This key is not mandatory.
        - ``position_label``: If given, the block will return the value of
          :meth:`get_position` under this label. This key is not mandatory.
        - ``speed_label``: If given, the block will return the value of
          :meth:`get_speed` under this label. This key is not mandatory.
        - ``speed_cmd_label``: The label carrying the speed to set when driving
          in position mode. Each time a value is received, the stored speed
          value is updated. It will also overwrite the ``speed`` key if given.

    """

    self._actuators: List[ActuatorInstance] = list()
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
      actuator.update(common)

    # Making sure all the dicts contain the 'type' key
    if not all('type' in dic for dic in actuators):
      raise ValueError("The 'type' key must be provided for all the "
                       "actuators !")

    # The names of the possible settings, to avoid typos and reduce verbosity
    actuator_settings = [field.name for field in fields(ActuatorInstance)
                         if field.type is not Actuator]

    # The list of all the Actuator types to instantiate
    self._types = [actuator['type'] for actuator in actuators]

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
    objects to drive."""

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
    """Receives the commands from upstream blocks, sets them on the actuators
    to drive, and sends the read positions and speed to the downstream
    blocks."""

    # Receiving the latest command
    recv = self.recv_last_data(fill_missing=self._spam)

    # Iterating over the actuators for setting the commands
    if recv:
      for actuator in self._actuators:
        # Setting the speed attribute if it was received
        if actuator.speed_cmd_label is not None and \
            actuator.speed_cmd_label in recv:
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
    """Stops and closes all the actuators to drive."""

    for actuator in self._actuators:
      self.log(logging.INFO, f"Stopping the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.stop()
    for actuator in self._actuators:
      self.log(logging.INFO, f"Closing the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.close()
      self.log(logging.INFO, f"closed the {type(actuator.actuator).__name__}"
                             f"Actuator")
