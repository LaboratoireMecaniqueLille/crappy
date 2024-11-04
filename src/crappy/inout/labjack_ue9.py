# coding: utf-8

from time import time
from typing import Optional
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from ue9 import UE9
except (ModuleNotFoundError, ImportError):
  UE9 = OptionalModule("LabJackPython")


@dataclass
class _Channel:
  """This class is a simple structure holding all the attributes a Labjack UE9
  channel can have."""

  num: int
  range_num: int = 12
  gain: float = 1
  offset: float = 0
  make_zero: bool = False


class LabjackUE9(InOut):
  """This class can read the analog input channels from a Labjack UE9 device.

  It cannot read nor drive any of the other inout or output channels on the
  UE9. The UE9 model has been discontinued, and replaced by the T7 model (see
  :class:`~crappy.inout.LabjackT7`). It is thus likely that this class won't be
  further improved in the future.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Labjack_ue9* to *LabjackUE9*
  """

  def __init__(self,
               channels: Iterable[int],
               gain: Optional[Iterable[float]] = None,
               offset: Optional[Iterable[float]] = None,
               make_zero: Optional[Iterable[bool]] = None,
               resolution: Optional[Iterable[int]] = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        all the channels to read, given as :obj:`int`. Only the `AIN` channels
        can be read by this class, so to read the channel `AIN2` the integer
        `2` should be added to the iterable.
      gain: An iterable (like a :obj:`list` or a :obj:`tuple`) containing for
        each channel the gain to apply to the measured voltage, as a
        :obj:`float`. The returned voltage is calculated as follows :
        :math:`returned\\_voltage = gain * measured\\_voltage + offset`. If not
        given, no gain is applied to the measured values.
      offset: An iterable (like a :obj:`list` or a :obj:`tuple`) containing for
        each channel the offset to apply to the measured voltage, as a
        :obj:`float`. The returned voltage is calculated as follows :
        :math:`returned\\_voltage = gain * measured\\_voltage + offset`. If not
        given, no offset is applied to the measured values.
      make_zero: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each channel a :obj:`bool` indicating whether the channel should be
        zeroed or not. If so, data will be acquired on this channel before the
        test starts, and a compensation value will be deduced so that the
        offset of this channel is `0`. **It will only take effect if the**
        ``make_zero_delay`` **argument of the** :class:`~crappy.blocks.IOBlock`
        **controlling the Labjack is set** ! If not given, the channels are by
        default not zeroed.
      resolution: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each channel the resolution of the acquisition as an integer. Refer
        to Labjack documentation for more details. The higher this value the
        better the resolution, but the lower the speed. The possible range is
        `1` to `12`, and the default is `12`.

    Note:
      All the :iterables given as arguments for the channels should have the
      same length. If that's not the case, all the given iterables are treated
      as if they had the same length as the shortest given one.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._handle = None

    super().__init__()

    # Setting the defaults for arguments that are not given
    if resolution is None:
      resolution = [12 for _ in channels]
    if gain is None:
      gain = [1 for _ in channels]
    if offset is None:
      offset = [0 for _ in channels]
    if make_zero is None:
      make_zero = [False for _ in channels]

    self._channels = [_Channel(num=chan, range_num=r_num, gain=g,
                               offset=off, make_zero=make_z)
                      for chan, r_num, g, off, make_z in
                      zip(channels, resolution, gain, offset, make_zero)]
    self.log(logging.DEBUG, f"Input channels: {self._channels}")

  def open(self) -> None:
    """Opens the connection to the Labjack."""

    self.log(logging.INFO, "Opening the connection to the Labjack")
    self._handle = UE9()

  def make_zero(self, delay: float) -> None:
    """Overriding the method of the parent class, because the user can choose
    which channels should be zeroed or not.

    It simply performs the regular zeroing, and resets the compensation to
    zero for the channels that shouldn't be zeroed.
    
    .. versionadded:: 1.5.10
    """

    # No need to acquire data if no channel should be zeroed
    if any(chan.make_zero for chan in self._channels):

      # Acquiring the data
      super().make_zero(delay)

      # Proceed only if the acquisition went fine
      if self._compensations:
        # Resetting the compensation for channels that shouldn't be zeroed
        self._compensations = [comp if chan.make_zero else 0 for comp, chan
                               in zip(self._compensations, self._channels)]

  def get_data(self) -> list[float]:
    """Reads sequentially the channels and returns the acquired values,
    corrected by the given gains and offsets."""

    return [time()] + [self._handle.getAIN(chan.num, Resolution=chan.range_num)
                       * chan.gain + chan.offset for chan in self._channels]

  def close(self) -> None:
    """Closes the connection to the Labjack, if it was already opened."""

    if self._handle is not None:
      self.log(logging.INFO, "Closing the connection to the Labjack")
      self._handle.close()
