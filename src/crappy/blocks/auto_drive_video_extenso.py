# coding: utf-8

from time import time
from typing import Any, Optional, Literal
import logging

from .meta_block import Block
from ..actuator import actuator_dict, Actuator
from ..tool.ft232h import USBServer


class AutoDriveVideoExtenso(Block):
  """This Block is meant to drive an :class:`~crappy.actuator.Actuator` on 
  which a :class:`~crappy.camera.Camera` performing video-extensometry is 
  mounted, so that the spots stay centered on the image.

  It takes the output of a :class:`~crappy.blocks.VideoExtenso` Block and uses 
  the coordinates of the spots to drive the Actuator. The Actuator can only be 
  driven in speed, not in position. The label carrying the coordinates of the
  tracked spots must be ``'Coord(px)'``.

  It also outputs the difference between the center of the image and the middle
  of the spots, along with a timestamp, over the ``'t(s)'`` and ``'diff(pix)'``
  labels. It can then be used by downstream Blocks.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *AutoDrive* to *AutoDriveVideoExtenso*
  """

  def __init__(self,
               actuator: dict[str, Any],
               gain: float = 2000,
               direction: Literal['X-', 'X+', 'Y-', 'Y+'] = 'Y-',
               pixel_range: int = 2048,
               max_speed: float = 200000,
               ft232h_ser_num: Optional[str] = None,
               freq: Optional[float] = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      actuator: A :obj:`dict` for initializing the 
        :class:`~crappy.actuator.Actuator` to drive. Unlike for the
        :class:`~crappy.blocks.Machine` Block, only the ``'type'`` key is
        mandatory here. All the other keys will be considered as kwargs to
        pass to the Actuator.
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

        .. versionchanged:: 1.5.10 renamed from *range* to *pixel_range*
      max_speed: The absolute maximum speed value that can be sent to the
        Actuator.
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
    """

    self._device: Optional[Actuator] = None
    self._ft232h_args = None

    super().__init__()
    self.labels = ['t(s)', 'diff(pix)']
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Checking that the 'type' key is given
    if 'type' not in actuator:
      raise ValueError("The 'type' key must be provided for instantiating the "
                       "Actuator !")
    self._actuator = actuator

    self._gain = -gain if '-' in direction else gain
    self._direction = direction
    self._pixel_range = pixel_range
    self._max_speed = max_speed

    # Checking whether the Actuator communicates through an FT232H
    if actuator_dict[actuator['type']].ft232h:
      self._ft232h_args = USBServer.register(ft232h_ser_num)

  def prepare(self) -> None:
    """Checks the consistency of the linking and initializes the 
    :class:`~crappy.actuator.Actuator` to drive."""

    # Checking that there's exactly one input link
    if not self.inputs:
      raise IOError("The AutoDriveVideoExtenso Block should have an input "
                    "Link !")
    elif len(self.inputs) > 1:
      raise IOError("The AutoDriveVideoExtenso Block can only have one input "
                    "Link !")

    # Opening and initializing the actuator to drive
    actuator_name = self._actuator.pop('type')
    if self._ft232h_args is None:
      self._device = actuator_dict[actuator_name](**self._actuator)
    else:
      self._device = actuator_dict[actuator_name](
        **self._actuator, _ft232h_args=self._ft232h_args)
    self.log(logging.INFO, f"Opening the {type(self._device).__name__} "
                           f"actuator")
    self._device.open()
    self._device.set_speed(0)

  def loop(self) -> None:
    """Receives the latest data from the :class:`~crappy.blocks.VideoExtenso` 
    Block, calculates the center coordinate in the chosen direction, and sets 
    the :class:`~crappy.actuator.Actuator` speed accordingly."""

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
    """Stops the :class:`~crappy.actuator.Actuator` and closes it."""

    if self._device is not None:
      self.log(logging.INFO, f"Stopping the {type(self._device).__name__} "
                             f"actuator")
      self._device.stop()
      self.log(logging.INFO, f"Closing the {type(self._device).__name__} "
                             f"actuator")
      self._device.close()
