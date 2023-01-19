# coding: utf-8

from time import time
from typing import Optional, List
from dataclasses import dataclass
import logging

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
  """"""

  def __init__(self,
               channels: List[int],
               gain: Optional[List[float]] = None,
               offset: Optional[List[float]] = None,
               make_zero: Optional[List[bool]] = None,
               resolution: Optional[List[int]] = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      channels: A :obj:`list` containing all the channels to read, given as
        :obj:`int`. Only the `AIN` channels can be read by this class, so to
        read the channel `AIN2` the integer `2` should be added to the list.
      gain: A :obj:`list` containing for each channel the gain to apply to the
        measured voltage, as a :obj:`float`. The returned voltage is
        calculated as follows :
        ::

          returned_voltage = gain * measured_voltage + offset

        If not given, no gain is applied to the measured values.
      offset: A :obj:`list` containing for each channel the offset to apply to
        the measured voltage, as a :obj:`float`. The returned voltage is
        calculated as follows :
        ::

          returned_voltage = gain * measured_voltage + offset

        If not given, no offset is applied to the measured values.
      make_zero: A :obj:`list` containing for each channel a :obj:`bool`
        indicating whether the channel should be zeroed or not. If so, data
        will be acquired on this channel before the test starts, and a
        compensation value will be deduced so that the offset of this channel
        is `0`. **It will only take effect if the ``make_zero_delay`` argument
        of the :ref:`IOBlock` controlling the Labjack is set** ! If not given,
        the channels are by default not zeroed.
      resolution: A :obj:`list` containing for each channel the resolution of
        the acquisition as an integer. Refer to Labjack documentation for more
        details. The higher this value the better the resolution, but the lower
        the speed. The possible range is `1` to `12`, and the default is `12`.

    Note:
      All the :obj:`list` given as arguments for the channels should have the
      same length. If that's not the case, all the given lists are treated as
      if they had the same length as the shortest given list.
    """

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
    """Simply opens the connection to the Labjack."""

    self.log(logging.INFO, "Opening the connection to the Labjack")
    self._handle = UE9()

  def make_zero(self, delay: float) -> None:
    """Overriding of the method of the parent class, because the user can
    choose which channels should be zeroed or not.

    It simply performs the regular zeroing, and resets the compensation to
    zero for the channels that shouldn't be zeroed.
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

  def get_data(self) -> List[float]:
    """Simply reads sequentially the channels and returns the acquired values,
    corrected by the given gains and offsets."""

    return [time()] + [self._handle.getAIN(chan.num, Resolution=chan.range_num)
                       * chan.gain + chan.offset for chan in self._channels]

  def close(self) -> None:
    """Closes the Labjack if it was already opened."""

    if self._handle is not None:
      self.log(logging.INFO, "Closing the connection to the Labjack")
      self._handle.close()
