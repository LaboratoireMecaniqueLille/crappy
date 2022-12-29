# coding: utf-8

from time import time
from typing import Dict, Any, Optional

from .block import Block
from ..actuator import actuator_list


class AutoDrive(Block):
  """This block is meant to drive an actuator on which a camera performing
  videoextensometry is mounted so that the spots stay centered on the image.

  It takes the output of a :ref:`VideoExtenso` block and uses the coordinates
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
               freq: float = 200,
               verbose: bool = False) -> None:
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
      verbose: If :obj:`True`, prints the looping frequency of the block.
    """

    super().__init__()
    self.labels = ['t(s)', 'diff(pix)']
    self.freq = freq
    self.verbose = verbose

    self._actuator = {'name': 'CM_drive'} if actuator is None else actuator
    self._gain = -gain if '-' in direction else gain
    self._direction = direction
    self._pixel_range = pixel_range
    self._max_speed = max_speed

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
    self._device = actuator_list[actuator_name](**self._actuator)
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
    self._device.set_speed(speed)
    self.send([t - self.t0, diff])

  def finish(self) -> None:
    """Simply sets the device speed to `0`."""

    self._device.set_speed(0)
