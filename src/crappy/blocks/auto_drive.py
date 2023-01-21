# coding: utf-8

from time import time
from typing import Dict, Any, Optional
import logging

from .meta_block import Block
from ..actuator import actuator_dict, Actuator
from ..tool.ft232h import USBServer


class AutoDrive(Block):
  """This block is meant to drive an actuator on which a camera performing
  videoextensometry is mounted so that the spots stay centered on the image.

  It takes the output of a :ref:`Video Extenso` block and uses the coordinates
  of the spots to drive the actuator. The actuator can only be driven in speed,
  not in position.

  It also outputs the difference between the center of the image and the middle
  of the spots, along with a timestamp, over the ``'t(s)'`` and ``'diff(pix)'``
  labels. It can then be used by downstream blocks.
  """

  def __init__(self,
               actuator: Optional[Dict[str, Any]] = None,
               gain: float = 2000,
               direction: str = 'Y-',
               pixel_range: int = 2048,
               max_speed: float = 200000,
               ft232h_ser_num: Optional[str] = None,
               freq: float = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      actuator: A :obj:`dict` for initializing the actuator to drive. It
        should contain the name of the actuator under the key ``'name'``, and
        all the arguments to pass to the actuator as key/value pairs. The
        default actuator if this argument is not set is the :ref:`CM Drive`
        with its default arguments.
      gain: The gain for driving the actuator in speed. The speed command is
        simply the difference in pixels between the center of the image and the
        center of the spots, multiplied by this gain.
      direction: Indicates which axis to consider for driving the actuator, and
        whether the action should be inverted. The first character is the axis
        (`X` or `Y`) and second character is the inversion (`+` or `-`). The
        inversion depends on whether a positive speed will make bring the spots
        closer or farther, you have to try !
      pixel_range: The size of the image (in pixels) along the chosen axis.
      max_speed: The absolute maximum speed value that can be sent to the
        actuator.
      freq: The block will try to loop at this frequency.
      display_freq: If :obj:`True`, displays the looping frequency of the
        block.
    """

    self._device: Optional[Actuator] = None
    self._ft232h_args = None

    super().__init__()
    self.labels = ['t(s)', 'diff(pix)']
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._actuator = {'name': 'CM_drive'} if actuator is None else actuator
    self._gain = -gain if '-' in direction else gain
    self._direction = direction
    self._pixel_range = pixel_range
    self._max_speed = max_speed

    # Checking whether the Actuator communicates through an FT232H
    if actuator_dict[actuator['name']].ft232h:
      self._ft232h_args = USBServer.register(ft232h_ser_num)

  def prepare(self) -> None:
    """Checks the consistency of the linking and initializes the actuator to
    drive."""

    # Checking that there's exactly one input link
    if not self.inputs:
      raise IOError("The AutoDrive block should have an input link !")
    elif len(self.inputs) > 1:
      raise IOError("The AUtoDrive block can only have one input link !")

    # Opening and initializing the actuator to drive
    actuator_name = self._actuator.pop('name')
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
    """Receives the latest data from the VideoExtenso block, calculates the
    center coordinate in the chosen direction, and sets the actuator speed
    accordingly."""

    # Receiving the latest data
    data = self.recv_last_data(fill_missing=False)
    if not data:
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
    """Simply sets the device speed to `0`."""

    if self._device is not None:
      self.log(logging.INFO, f"Stopping the {type(self._device).__name__} "
                             f"actuator")
      self._device.stop()
      self.log(logging.INFO, f"Closing the {type(self._device).__name__} "
                             f"actuator")
      self._device.close()
