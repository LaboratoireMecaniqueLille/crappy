# coding: utf-8

from time import time
from typing import Any, Literal
from numbers import Real
from math import isfinite
import logging

from .meta_block import Block
from ..actuator import actuator_dict, Actuator, moved_to_collection
from .._collection import (CollectionEntry, collection_registry,
                           load_collection_class)


class AutoDriveVideoExtenso(Block):
  """This Block is meant to drive an
  :class:`~crappy.actuator.meta_actuator.actuator.Actuator` on which a
  :class:`~crappy.camera.meta_camera.camera.Camera` performing
  video-extensometry is mounted, so that the spots stay centered on the image.

  It takes the output of a :class:`~crappy.blocks.VideoExtenso` Block and uses 
  the coordinates of the spots to drive the Actuator. The Actuator can only be 
  driven in speed, not in position. The label carrying the coordinates of the
  tracked spots must be ``'Coord(px)'``.

  It also outputs the difference between the center of the image and the middle
  of the spots, along with a timestamp, over the ``'t(s)'`` and ``'diff(pix)'``
  labels. It can then be used by downstream Blocks.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *AutoDrive* to *AutoDriveVideoExtenso*
  .. versionchanged:: 2.1.0 removed support for Actuators communicating through
     an FT232H
  """

  def __init__(self,
               actuator: dict[str, Any],
               gain: float = 2000,
               direction: Literal['X-', 'X+', 'Y-', 'Y+'] = 'Y-',
               pixel_range: int = 2048,
               max_speed: float = 200000,
               freq: float | None = 200,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      actuator: A :obj:`dict` for initializing the 
        :class:`~crappy.actuator.meta_actuator.actuator.Actuator` to drive.
        Unlike for the :class:`~crappy.blocks.Machine` Block, only the
        ``'type'`` key is mandatory here. All the other keys will be considered
        as kwargs to pass to the Actuator.
      gain: The gain for driving the Actuator in speed. The speed command is
        simply the difference in pixels between the center of the image and the
        center of the spots, multiplied by this gain.

        .. versionchanged:: 1.5.10 renamed from *P* to *gain*
      direction: Indicates which axis to consider for driving the Actuator, and
        whether the action should be inverted. The first character is the axis
        (`X` or `Y`) and second character is the inversion (`+` or `-`). The
        inversion depends on whether a positive speed will bring the spots
        closer or farther.
      pixel_range: The size of the image (in pixels) along the chosen axis.
        Must be a strictly positive integer.

        .. versionchanged:: 1.5.10 renamed from *range* to *pixel_range*
      max_speed: The absolute maximum speed value that can be sent to the
        Actuator. Must be strictly positive and finite.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.

        .. versionadded:: 2.0.0
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.

        .. versionadded:: 2.0.0

    .. versionremoved:: 2.1.0 *ft232h_ser_num* argument
    """

    self._device: Actuator | None = None
    self._device_opened: bool = False
    self._device_stopped: bool = False

    super().__init__()
    self.labels = ['t(s)', 'diff(pix)']
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match actuator:
      case dict():
        pass
      case _:
        raise TypeError("actuator must be a dictionary")
    if not all(isinstance(key, str) for key in actuator):
      raise TypeError("All actuator dictionary keys must be strings")
    if 'type' not in actuator:
      raise ValueError("The 'type' key must be provided for instantiating the "
                       "Actuator !")
    match actuator['type']:
      case str(name) if name.strip():
        self._actuator_name: str = name
      case str():
        raise ValueError("The actuator type must be a non-empty string")
      case _:
        raise TypeError("The actuator type must be a non-empty string")

    self._collection_entry: CollectionEntry | None = None
    entry = collection_registry.get('Actuator', self._actuator_name)
    if self._actuator_name not in actuator_dict:
      if entry is not None:
        load_collection_class(entry, actuator_dict)
        self._collection_entry = entry
      elif self._actuator_name in moved_to_collection:
        raise NotImplementedError(f"The Actuator {self._actuator_name} was "
                                  f"moved to crappy.collection. To use it, "
                                  f"add import crappy.collection at the "
                                  f"beginning of your script")
      else:
        raise ValueError(f"Unknown actuator name: {self._actuator_name}!")
    elif (entry is not None and
          actuator_dict[self._actuator_name].__module__ == entry.module):
      self._collection_entry = entry

    self._actuator_kwargs: dict[str, Any] = {key: value for key, value
                                             in actuator.items()
                                             if key != 'type'}

    match direction:
      case 'X-' | 'X+' | 'Y-' | 'Y+' | 'x-' | 'x+' | 'y-' | 'y+':
        self._direction: str = direction
      case str():
        raise ValueError("Direction should be in "
                         "('X-', 'X+', 'Y-', 'Y+', 'x-', 'x+', 'y-', 'y+')")
      case _:
        raise TypeError("direction must be a string")

    match gain:
      case Real() if isfinite(gain):
        self._gain: float = float(-gain if '-' in direction else gain)
      case Real():
        raise ValueError("gain must be a finite number")
      case _:
        raise TypeError("gain must be a finite number")

    match pixel_range:
      case bool():
        raise TypeError("pixel_range must be a strictly positive integer")
      case int() if pixel_range > 0:
        self._pixel_range: int = pixel_range
      case int():
        raise ValueError("pixel_range must be a strictly positive integer")
      case _:
        raise TypeError("pixel_range must be a strictly positive integer")

    match max_speed:
      case Real() if max_speed > 0 and isfinite(max_speed):
        self._max_speed: float = float(max_speed)
      case Real():
        raise ValueError("max_speed must be a strictly positive, finite "
                         "number")
      case _:
        raise TypeError("max_speed must be a strictly positive, finite number")

  def prepare(self) -> None:
    """Checks the consistency of the linking and initializes the 
    :class:`~crappy.actuator.meta_actuator.actuator.Actuator` to drive."""

    # Checking that there's exactly one input link
    if not self.inputs:
      raise IOError("The AutoDriveVideoExtenso Block should have an input "
                    "Link !")
    elif len(self.inputs) > 1:
      raise IOError("The AutoDriveVideoExtenso Block can only have one input "
                    "Link !")

    # Under the spawn multiprocessing start method, it is necessary to re-load
    # the modules from crappy.collection
    if self._collection_entry is not None:
      load_collection_class(self._collection_entry, actuator_dict)

    # Instantiate the Actuator to drive
    self._device = actuator_dict[self._actuator_name](**self._actuator_kwargs)
    self._device_opened = False
    self._device_stopped = False

    assert self._device is not None
    self.log(logging.INFO, f"Opening the {type(self._device).__name__} "
                           f"actuator")
    self._device.open()
    self._device_opened = True
    self._device.set_speed(0)

  def loop(self) -> None:
    """Receives the latest data from the :class:`~crappy.blocks.VideoExtenso` 
    Block, calculates the center coordinate in the chosen direction, and sets 
    the :class:`~crappy.actuator.meta_actuator.actuator.Actuator` speed
    accordingly."""

    # Receiving the latest data
    if not (data := self.recv_last_data(fill_missing=False)):
      return

    # Extracting the coordinates of the spots
    coord = data['Coord(px)']
    t = time()

    # Getting the average coordinate in the chosen direction
    y, x = list(zip(*coord))
    if 'x' in self._direction.lower():
      center = (max(x) + min(x)) / 2
    else:
      center = (max(y) + min(y)) / 2

    # Calculating the new speed to set
    diff = center - self._pixel_range / 2
    speed = max(-self._max_speed, min(self._max_speed, self._gain * diff))

    # Setting the speed and sending to downstream blocks
    self.log(logging.DEBUG, f"Setting the speed: {speed} on the "
                            f"{type(self._device).__name__} actuator.")
    self._device.set_speed(speed)
    self.send([t - self.t0, diff])

  def finish(self) -> None:
    """Stops the :class:`~crappy.actuator.meta_actuator.actuator.Actuator` and
    closes it."""

    if self._device is None:
      return

    name = type(self._device).__name__
    failures: list[Exception | KeyboardInterrupt] = list()

    if self._device_opened and not self._device_stopped:
      self.log(logging.INFO, f"Stopping the {name} actuator")
      try:
        self._device.stop()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note(f"AutoDriveVideoExtenso cleanup step: stop {name}")
        failures.append(error)
      else:
        self._device_stopped = True

    self.log(logging.INFO, f"Closing the {name} actuator")
    try:
      self._device.close()
    except (Exception, KeyboardInterrupt) as error:
      error.add_note(f"AutoDriveVideoExtenso cleanup step: close {name}")
      failures.append(error)
    else:
      self._device = None
      self._device_opened = False

    # If there's only one Exception, raise it
    if len(failures) == 1:
      raise failures[0]
    # Handle the case when a KeyboardInterrupt is among the Exceptions
    elif any(isinstance(error, KeyboardInterrupt) for error in failures):
      for index, error in enumerate(failures):
        if isinstance(error, KeyboardInterrupt):
          others: list[BaseException] = failures[:index] + failures[index + 1:]
          if error.__cause__ is not None:
            others.insert(0, error.__cause__)
          raise error from BaseExceptionGroup("Other AutoDrive cleanup "
                                              "failures", others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("AutoDrive cleanup failures", failures)
