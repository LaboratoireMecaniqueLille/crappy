# coding: utf-8

from time import time
from typing import Optional
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from  warnings import warn

from .meta_inout import InOut
from ..tool.bindings import comedi_bind as comedi


@dataclass
class _Channel:
  """This class is a simple structure holding all the attributes a Comedi
  channel can have."""

  num: int
  range_num: int = 0
  gain: float = 1
  offset: float = 0
  make_zero: bool = False
  max_data: int = 0
  range_ds: int = 0


class Comedi(InOut):
  """This class can control acquisition boards relying on the Comedi driver.

  It can read data from ADCs on input channels, and set voltages of DACs on
  output channels. Each channel can be tuned independently in terms of range,
  gan, offset, and for input channels it's possible to decide whether they
  should be offset to `0` at the beginning of the test.

  Communicates over serial. This class was implemented and tested on a USB-DUX
  sigma device. IT should work with other devices as well, but that was not
  tested.

  Note:
    This class requires the `libcomedi` library to be installed on the
    computer.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               device: str = '/dev/comedi0',
               sub_device: int = 0,
               channels: Optional[Iterable[int]] = None,
               range_num: Optional[Iterable[int]] = None,
               gain: Optional[Iterable[float]] = None,
               offset: Optional[Iterable[float]] = None,
               make_zero: Optional[Iterable[bool]] = None,
               out_sub_device: int = 1,
               out_channels: Optional[Iterable[int]] = None,
               out_range_num: Optional[Iterable[int]] = None,
               out_gain: Optional[Iterable[float]] = None,
               out_offset: Optional[Iterable[float]] = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      device: The address of the device, as a :obj:`str`.
      sub_device: The id of the subdevice to use for input channels, as an
        :obj:`int`.

        .. versionchanged:: 2.0.0 renamed from *subdevice* to *sub_device*
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        the indexes of the channels to use as inputs, given as :obj:`int`.
      range_num: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each input channel the index of the range to set for that channel,
        as an :obj:`int`. Refer to the documentation of the board to get the
        correspondence between range indexes and Volts. If not given, all input
        channels will be set to the range `0`.
      gain: An iterable (like a :obj:`list` or a :obj:`tuple`) containing for
        each input channel the gain to apply to the measured voltage, as a
        :obj:`float`. The returned voltage is calculated as follows :
        :math:`returned\\_voltage = gain * measured\\_voltage + offset`. If not
        given, no gain is applied to the measured values.
      offset: An iterable (like a :obj:`list` or a :obj:`tuple`) containing for
        each input channel the offset to apply to the measured voltage, as a
        :obj:`float`. The returned voltage is calculated as follows :
        :math:`returned\\_voltage = gain * measured\\_voltage + offset`. If not
        given, no offset is applied to the measured values.
      make_zero: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each input channel a :obj:`bool` indicating whether the channel
        should be zeroed or not. If so, data will be acquired on this channel
        before the test starts, and a compensation value will be deduced so
        that the offset of this channel is `0`. **It will only take effect if
        the** ``make_zero_delay`` **argument of the**
        :class:`~crappy.blocks.IOBlock` **controlling the Comedi is set** ! If
        not given, the channels are by default not zeroed.
      out_sub_device: The id of the subdevice to use for output channels, as an
        :obj:`int`.

        .. versionchanged:: 2.0.0
           renamed from *out_subdevice* to *out_sub_device*
      out_channels: An iterable (like a :obj:`list` or a :obj:`tuple`)
        containing the indexes of the channels to use as outputs, given as
        :obj:`int`.
      out_range_num: An iterable (like a :obj:`list` or a :obj:`tuple`)
        containing for each output channel the index of the range to set for
        that channel, as an :obj:`int`. Refer to the documentation of the board
        to get the correspondence between range indexes and Volts. If not
        given, all output channels will be set to the range `0`.
      out_gain: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each output channel the gain to apply to the command voltage, as a
        :obj:`float`. The set voltage is calculated as follows :
        .. math::

          set_voltage = out_gain * command_voltage + out_offset

        If not given, no gain is applied to the command values.
      out_offset: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        for each output channel the offset to apply to the command voltage, as
        a :obj:`float`. The set voltage is calculated as follows :
        .. math::

          set_voltage = out_gain * command_voltage + out_offset

        If not given, no offset is applied to the command values.

    Note:
      All the iterables given as arguments for the input channels should have
      the same length, and same for the output channels. If that's not the
      case, all the given iterables are treated as if they had the same length
      as the shortest given one.

    .. versionchanged:: 1.5.10
       now explicitly listing the *device*, *subdevice*, *channels*,
       *range_num*, *gain*, *offset*, *make_zero*, *out_subdevice*,
       *out_channels*, *out_range_num*, *out_gain* and *out_offset* arguments
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._device = None

    super().__init__()

    self._device_name = device.encode()
    self._sub_device = sub_device
    self._out_sub_device = out_sub_device

    # Setting the defaults for arguments that are not given
    if channels is None:
      channels = list()
    if out_channels is None:
      out_channels = list()

    if range_num is None:
      range_num = [0 for _ in channels]
    if gain is None:
      gain = [1 for _ in channels]
    if offset is None:
      offset = [0 for _ in channels]
    if make_zero is None:
      make_zero = [False for _ in channels]

    if out_range_num is None:
      out_range_num = [0 for _ in out_channels]
    if out_gain is None:
      out_gain = [1 for _ in out_channels]
    if out_offset is None:
      out_offset = [0 for _ in out_channels]

    # Creating the channel objects
    self._channels = [_Channel(num=chan, range_num=r_num, gain=g,
                               offset=off, make_zero=make_z)
                      for chan, r_num, g, off, make_z in
                      zip(channels, range_num, gain, offset, make_zero)]
    self.log(logging.DEBUG, f"Input channels: {self._channels}")

    self._out_channels = [_Channel(num=chan, range_num=r_num, gain=g,
                                   offset=off) for chan, r_num, g, off in
                          zip(out_channels, out_range_num,
                              out_gain, out_offset)]
    self.log(logging.DEBUG, f"Output channels: {self._out_channels}")

    self._device = None

  def open(self) -> None:
    """Opening the Comedi board and setting up the input and output
    channels."""

    # Opening the Comedi board
    self.log(logging.INFO, "Opening the connection to the Comedi device")
    self._device = comedi.comedi_open(self._device_name)

    # Setting up the input channels
    self.log(logging.INFO, "Setting up the input channels")
    for chan in self._channels:
      chan.max_data = comedi.comedi_get_maxdata(self._device, self._sub_device,
                                                chan.num)
      chan.range_ds = comedi.comedi_get_range(self._device, self._sub_device,
                                              chan.num, chan.range_num)

    # Setting up the output channels
    self.log(logging.INFO, "Setting up the output channels")
    for chan in self._out_channels:
      chan.max_data = comedi.comedi_get_maxdata(self._device,
                                                self._out_sub_device,
                                                chan.num)
      chan.range_ds = comedi.comedi_get_range(self._device,
                                              self._out_sub_device,
                                              chan.num, chan.range_num)

  def make_zero(self, delay: float) -> None:
    """Overriding the method of the parent class, because the user can choose
    which channels should be zeroed or not.

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

  def set_cmd(self, *cmd: float) -> None:
    """Sets the command value on the output channels.

    There should be as many commands as there are output channels. In case
    there would be fewer commands or channels, the extra commands/channels
    wouldn't be considered/set.
    """

    for val, chan in zip(cmd, self._out_channels):

      # Adjusting with the provided gain and offset
      val = val * chan.gain + chan.offset

      # Converting voltage to numeric
      out_a = comedi.comedi_from_phys(val, chan.range_ds, chan.max_data)

      # Sending the command
      self.log(logging.DEBUG, f"Writing value {out_a} to channel {chan.num}")
      comedi.comedi_data_write(self._device, self._out_sub_device, chan.num,
                               chan.range_num, comedi.AREF_GROUND, out_a)

  def get_data(self) -> list[float]:
    """Reads and returns the value of each channel, adjusted with the given
    gain and offset."""

    data = [time()]

    for chan in self._channels:

      # Reading the numeric values
      data_read = comedi.comedi_data_read(self._device, self._sub_device,
                                          chan.num, chan.range_num,
                                          comedi.AREF_GROUND)
      self.log(logging.DEBUG, f"Read value {data_read} to channel {chan.num}")

      # Converting numeric to a voltage
      val = comedi.comedi_to_phys(data_read, chan.range_ds, chan.max_data)

      # Adjusting with the provided gain and offset
      data.append(val * chan.gain + chan.offset)
    return data

  def close(self) -> None:
    """Closes the Comedi board and warns the user in case of failure."""

    if self._device is not None:
      self.log(logging.INFO, "Closing the connection to the Comedi device")
      if comedi.comedi_close(self._device):
        self.log(logging.WARNING, "Closing device failed !")
