# coding: utf-8

from time import time
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from numbers import Real
from math import isfinite
import logging

from .meta_block import Block
from .._collection import (CollectionEntry, collection_registry,
                           load_collection_class)
from ..actuator import actuator_dict, Actuator, moved_to_collection


@dataclass
class ActuatorInstance:
  """This class holds all the information that can be associated to an
  Actuator."""

  actuator: Actuator
  speed: float | None = None
  position_label: str | None = None
  speed_label: str | None = None
  mode: str = 'speed'
  cmd_label: str = 'cmd'
  speed_cmd_label: str | None = None

  # Lifecycle states
  opened: bool = field(default=False, init=False)
  stopped: bool = field(default=False, init=False)
  closed: bool = field(default=False, init=False)


class Machine(Block):
  """This Block is meant to drive one or several
  :class:`~crappy.actuator.meta_actuator.actuator.Actuator`. It can set speed
  or position commands on hardware actuators.

  The possibility to drive several Actuators from a unique Block is given so
  that they can be driven in a synchronized way. If synchronization is not
  needed, it is preferable to drive the Actuators from separate Machine Blocks.

  This Block takes the speed or position commands for the Actuators  as inputs,
  and can optionally read and output the current speed and/or positions of the
  Actuators. The speed and position commands are set respectively by calling
  the :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_position` and
  :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_speed` methods of
  the Actuators, and the current speed and position values are acquired by
  calling the
  :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_position` and
  :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_speed` methods of
  the Actuators.

  It is possible to tune for each Actuator the label over which it receives its
  commands, and optionally the labels over which it sends its current speed
  and/or position. The driving mode (`'speed'` or `'position'`) can also be set
  independently for each Actuator.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.1.0 removed support for Actuators communicating through
     an FT232H
  """

  def __init__(self,
               actuators: Sequence[dict[str, Any]],
               common: dict[str, Any] | None = None,
               time_label: str = 't(s)',
               spam: bool = False,
               freq: float | None = 200,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      actuators: A non-empty sequence (like a :obj:`list` or a :obj:`tuple`) of
        all the :class:`~crappy.actuator.meta_actuator.actuator.Actuator` this
        Block needs to drive. It contains one :obj:`dict` for every Actuator,
        with mandatory and optional keys. The keys providing information on how
        to drive the Actuator are listed below. Any other unrecognized key will
        be passed to the Actuator as argument when instantiating it.
      common: The keys of this :obj:`dict` will be common to all the Actuators.
        If one key conflicts with an existing key for an Actuator, the common 
        one will prevail.
      time_label: If reading speed or position from one or more Actuators, the
        time information will be carried by this non-empty string label.
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

        - ``type``: The name of the
          :class:`~crappy.actuator.meta_actuator.actuator.Actuator` class to
          instantiate. This key is mandatory.
        - ``cmd_label``: The label carrying the command for driving the
          Actuator. It defaults to `'cmd'`.
        - ``mode``: Can be either `'speed'` or `'position'`. Either
          :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_speed` or
          :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_position`
          is called to drive the Actuator, depending on the selected mode. When
          driven in `'position'` mode, the speed of the actuator can also be
          adjusted, see the ``speed`` and ``speed_cmd_label`` keys. The default
          mode is `'speed'`.
        - ``speed``: If mode is `'position'`, the speed at which the Actuator
          should move. This speed is passed as second argument to the
          :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_position`
          method of the Actuator. If the ``speed_cmd_label`` key is not
          specified, this speed will remain the same for the entire test. This
          key is not mandatory. When given, it must be finite or :obj:`None`.
        - ``position_label``: If given, the Block will return the value of
          :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_position`
          under this label. This key is not mandatory.
        - ``speed_label``: If given, the Block will return the value of
          :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_speed`
          under this label. This key is not mandatory.
        - ``speed_cmd_label``: The label carrying the speed to set when driving
          in `'position'` mode. Each time a value is received, the stored speed
          value is updated. It will also overwrite the ``speed`` key if given.

    .. versionremoved:: 2.1.0 *ft232h_ser_num* argument
    """

    self._actuators: list[ActuatorInstance] = list()

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match actuators:
      case ():
        raise ValueError("No actuator to drive was specified")
      case (*actuators,) if all(isinstance(actuator, dict)
                                for actuator in actuators):
        pass
      case _:
        raise TypeError("actuators must be a non-empty sequence of "
                        "dictionaries")

    match common:
      case None:
        common = dict()
      case dict():
        pass
      case _:
        raise TypeError("common must be a dictionary or None")

    match time_label:
      case str() if time_label.strip():
        self._time_label: str = time_label
      case str():
        raise ValueError("time_label must be a non-empty string")
      case _:
        raise TypeError("time_label must be a non-empty string")

    match spam:
      case bool():
        self._spam: bool = spam
      case _:
        raise TypeError("spam must be provided as a boolean")

    # Merge common values into the Actuators dictionaries
    actuators = [actuator | common for actuator in actuators]
    if not all(isinstance(key, str) for actuator in actuators
               for key in actuator):
      raise TypeError("All actuator dictionary keys must be strings")

    # Making sure all the dicts contain the 'type' key
    if not all('type' in dic for dic in actuators):
      raise ValueError("The 'type' key must be provided for all the "
                       "actuators !")

    # Validate Actuator types and modes
    for actuator in actuators:
      match actuator['type']:
        case str() if actuator['type'].strip():
          pass
        case str():
          raise ValueError("The actuator type must be a non-empty string")
        case _:
          raise TypeError("The actuator type must be a non-empty string")

      match actuator.get('mode', 'speed'):
        case 'speed' | 'position':
          pass
        case str():
          raise ValueError("The 'mode' key must be either 'speed' or "
                           "'position'")
        case _:
          raise TypeError("The 'mode' key must be a string")

      match actuator.get('cmd_label', 'cmd'):
        case str(label) if label.strip():
          pass
        case str():
          raise ValueError("cmd_label must be a non-empty string")
        case _:
          raise TypeError("cmd_label must be a non-empty string")

      for key in ('position_label', 'speed_label', 'speed_cmd_label'):
        match actuator.get(key):
          case None:
            pass
          case str(label) if label.strip():
            pass
          case str():
            raise ValueError(f"{key} must be a non-empty string or None")
          case _:
            raise TypeError(f"{key} must be a non-empty string or None")

      match actuator.get('speed'):
        case None:
          pass
        case Real() as speed if isfinite(speed):
          actuator['speed'] = float(speed)
        case Real():
          raise ValueError("speed must be a finite number or None")
        case _:
          raise TypeError("speed must be a finite number or None")

    # The names of the possible settings, to avoid typos and reduce verbosity
    actuator_settings = [setting.name for setting in fields(ActuatorInstance)
                         if setting.init and setting.type is not Actuator]

    # The list of all the Actuator types to instantiate
    self._types: list[str] = [actuator['type'] for actuator in actuators]

    # None means that this is an ordinary core or user-defined InOut
    self._collection_entries: list[CollectionEntry] = list()
    unknown = list()

    # Checking that all the given Actuators names are valid
    for type_ in self._types:
      # Check if the requested Actuator is part of crappy.collection
      entry = collection_registry.get("Actuator", type_)
      # This loop only handles invalid Actuators
      if type_ in actuator_dict:
        # Store the Actuator if it was already loaded in a separate Block
        if (entry is not None and
            actuator_dict[type_].__module__ == entry.module):
          self._collection_entries.append(entry)
        continue
      # First option, the Actuator should be loaded from crappy.collection
      if entry is not None:
        # This call raises early if the module cannot be loaded
        load_collection_class(entry, actuator_dict)
        self._collection_entries.append(entry)
      # Second case, the Actuator was moved to crappy.collection but this
      # module was not imported
      # Not reporting all moved Actuators at once but temporary for migration
      elif type_ in moved_to_collection:
        raise NotImplementedError(f"The Actuator {type_} was moved to "
                                  f"crappy.collection. To use it, simply "
                                  f"add import crappy.collection at the "
                                  f"beginning of your script")
      # The name of the Actuator simply cannot be found anywhere
      else:
        unknown.append(type_)
    # Report all the missing Actuators at once instead of raising for only one
    if unknown:
      unknown = ', '.join(unknown)
      possible = ', '.join(sorted(actuator_dict.keys()))
      raise ValueError(f"Unknown actuator name(s) : {unknown}! "
                       f"The currently available ones are: {possible}")

    # The settings that won't be passed to the Actuator objects
    self._settings = [{key: value for key, value in actuator.items()
                       if key in actuator_settings}
                      for actuator in actuators]

    # The settings that will be passed as kwargs to the Actuator objects
    self._actuators_kw = [{key: value for key, value in actuator.items()
                           if key not in ('type', *actuator_settings)}
                          for actuator in actuators]

  def prepare(self) -> None:
    """Checks the validity of the linking and initializes all the Actuator
    objects to drive.

    This method calls the
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.open` method of
    each Actuator.
    """

    # Checking the consistency of the linking
    if not self.inputs and not self.outputs:
      raise IOError("The Machine block isn't linked to any other block !")

    # Under the spawn multiprocessing start method, it is necessary to re-load
    # the modules from crappy.collection
    for entry in self._collection_entries:
      load_collection_class(entry, actuator_dict)

    # Instantiate all the Actuators to drive
    for type_, setting, actuator_kw in zip(self._types, self._settings,
                                           self._actuators_kw):
      actuator = ActuatorInstance(actuator=actuator_dict[type_](**actuator_kw),
                                  **setting)
      self._actuators.append(actuator)
      self.log(logging.INFO, f"Opening the {type(actuator.actuator).__name__}"
                             f"Actuator")
      actuator.actuator.open()
      actuator.opened = True
      self.log(logging.INFO, f"Opened the {type(actuator.actuator).__name__}"
                             f"Actuator")

  def loop(self) -> None:
    """Sets the received position and speed commands, and reads the current 
    speed and position from the
    :class:`~crappy.actuator.meta_actuator.actuator.Actuator`.
    
    For each Actuator, a command is set **only** if a new one was received or 
    if the ``spam`` argument is :obj:`True`. It is set using either 
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_position` or
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.set_speed`
    depending on the selected driving mode.
    
    For each Actuator, a speed and/or position value is read **only** if the 
    ``speed_label`` and/or the ``position_label`` was set. If so, these values
    are read at each loop and sent to downstream Blocks over the given labels.
    This is independent of the chosen driving mode. The
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_position` and
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.get_speed` are
    called for acquiring the position and speed values respectively.
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
          elif actuator.mode == 'position':
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

    Calls :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.stop` only
    for successfully opened Actuators, then attempts
    :meth:`~crappy.actuator.meta_actuator.actuator.Actuator.close` for every
    constructed Actuator, including one whose open failed.
    """

    failures: list[Exception | KeyboardInterrupt] = list()

    # Stop every opened actuator
    for actuator in self._actuators:
      if not actuator.opened or actuator.stopped or actuator.closed:
        continue
      name = type(actuator.actuator).__name__
      self.log(logging.INFO, f"Stopping the {type(actuator.actuator).__name__}"
                             f"Actuator")
      try:
        actuator.actuator.stop()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note(f"Machine cleanup step: stop actuator "
                       f"({name}, cmd_label={actuator.cmd_label!r})")
        failures.append(error)
      else:
        actuator.stopped = True

    # Close every opened actuator
    for actuator in self._actuators:
      if actuator.closed:
        continue
      name = type(actuator.actuator).__name__
      self.log(logging.INFO, f"Closing the {type(actuator.actuator).__name__}"
                             f"Actuator")
      try:
        actuator.actuator.close()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note(f"Machine cleanup step: close actuator "
                       f"({name}, cmd_label={actuator.cmd_label!r})")
        failures.append(error)
      else:
        actuator.closed = True
        actuator.opened = False
        self.log(logging.INFO, f"Closed the {name} Actuator")

    # If there's only one Exception, raise it
    if len(failures) == 1:
      raise failures[0]
    # Handle the case when a KeyboardInterrupt is among the Exceptions
    elif any(isinstance(error, KeyboardInterrupt) for error in failures):
      for index, error in enumerate(failures):
        if isinstance(error, KeyboardInterrupt):
          others = failures[:index] + failures[index + 1:]
          if error.__cause__ is not None:
            others.insert(0, error.__cause__)
          raise error from BaseExceptionGroup("Other Machine cleanup failures",
                                              others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("Machine cleanup failures", failures)
