# coding: utf-8

import numpy as np
from time import time
from typing import List, Optional
from .inout import InOut
from ..tool import pyspcm as spc


class Spectrum(InOut):
  """This class can read data from a Spectrum high speed ADC interfacing over
  PCIe.

  It can acquire data over multiple channels, and set for each channel a
  different voltage range. It is possible to tune the sample rate, the chunk
  size and the memory allocated to the data buffer.

  This class can only acquire data by streaming, it cannot acquire single data
  points.
  """

  def __init__(self,
               channels: List[int],
               device: str = '/dev/spcm0',
               ranges: Optional[List[int]] = None,
               sample_rate: int = 100000,
               buff_size: int = 2**26,
               notify_size: int = 2**16) -> None:
    """Sets the args and initializes the parent class.

    Args:
      channels: A :obj:`list` of all the channels to read data from, given as
        :obj:`int`. Refer to the documentation to know which combinations of
        channels are allowed.
      device: The address of the device to read data from, as a :obj:`str`.
      ranges: A :obj:`list` indicating for each channel the range of the
        acquisition in mV, as an :obj:`int`. There should be as many values in
        this list as there are channels. If not given, all ranges default to
        `10000` mV.
      sample_rate: The sample rate of the acquisition for all channels, in Hz.
        The default is 100KHz.
      buff_size: The size of the memory allocated as a rolling buffer to copy
        the data from the card, in bytes. The default is 67MB.
      notify_size: The size of each chunk of data to copy from the card, in
        bytes. The default is 65kB.
    """

    super().__init__()

    # Setting the args
    self._channels = channels
    self._device = device.encode()
    self._ranges = ranges if ranges is not None else [10000 for _ in channels]
    self._sample_rate = sample_rate
    self._buff_size = buff_size
    self._notify_size = notify_size
    self._chunk_size = notify_size // (2 * len(channels))

    print(f"[Spectrum] Will send "
          f"{2 * sample_rate * len(channels) / notify_size} chunks of "
          f"{notify_size / 1024} kB per second "
          f"({sample_rate * len(channels) / 512} kB/s)")

    # These attributes will be set later
    self._spectrum = None
    self._dt = None
    self._buff = None
    self._stream_t0 = 0
    self._n_points = 0
    self._stream_started = False

  def open(self) -> None:
    """Opens and configures the Spectrum, and sets the ranges and the sample
    rate as requested."""

    self._spectrum = spc.hOpen(self._device)

    # Configuring the Spectrum
    spc.dw_set_param(self._spectrum, spc.SPC_CHENABLE,
                     sum([2 ** chan for chan in self._channels]))
    spc.dw_set_param(self._spectrum, spc.SPC_CARDMODE, spc.SPC_REC_FIFO_SINGLE)
    spc.dw_set_param(self._spectrum, spc.SPC_TIMEOUT, 5000)
    spc.dw_set_param(self._spectrum, spc.SPC_TRIG_ORMASK,
                     spc.SPC_TMASK_SOFTWARE)
    spc.dw_set_param(self._spectrum, spc.SPC_TRIG_ANDMASK, 0)
    spc.dw_set_param(self._spectrum, spc.SPC_CLOCKMODE, spc.SPC_CM_INTPLL)

    # Setting for each channel its range
    for range_, chan in zip(self._ranges, self._channels):
      spc.dw_set_param(self._spectrum, spc.SPC_AMP0 + 100 * chan, range_)

    # Setting the target sample rate and reading the actual sample rate
    spc.dw_set_param(self._spectrum, spc.SPC_SAMPLERATE, self._sample_rate)
    spc.dw_set_param(self._spectrum, spc.SPC_CLOCKOUT, 0)
    real_samplerate = spc.dw_get_param(self._spectrum, spc.SPC_SAMPLERATE)

    self._dt = 1 / real_samplerate

    # Creating the buffer for data
    self._buff = spc.new_buffer(self._buff_size)
    spc.dw_def_transfer(self._spectrum, spc.SPCM_BUF_DATA,
                        spc.SPCM_DIR_CARDTOPC, self._notify_size, self._buff,
                        0, self._buff_size)

  def start_stream(self) -> None:
    """Starts the streams and saves the corresponding timestamp."""

    spc.dw_set_param(self._spectrum, spc.SPC_M2CMD, spc.M2CMD_CARD_START |
                     spc.M2CMD_CARD_ENABLETRIGGER | spc.M2CMD_DATA_STARTDMA)
    self._stream_t0 = time()
    self._stream_started = True

  def get_stream(self) -> List[np.ndarray]:
    """Waits for data to be available, and returns it along with an array of
    timestamps."""

    # Generating the array of timestamps
    start = self._stream_t0 + self._dt * self._n_points
    t = np.arange(start, start + (self._chunk_size - 1) * self._dt, self._dt)

    # Waiting until data is available
    spc.dw_set_param(self._spectrum, spc.SPC_M2CMD, spc.M2CMD_DATA_WAITDMA)
    pos = spc.dw_get_param(self._spectrum, spc.SPC_DATA_AVAIL_USER_POS)

    # Getting the data
    data = np.frombuffer(self._buff, dtype=np.int16,
                         count=self._notify_size // 2,
                         offset=pos).reshape(
      self._notify_size // (2 * len(self._channels)), len(self._channels))
    ret = data.copy()
    del data

    # Updating the number of received points
    spc.dw_set_param(self._spectrum, spc.SPC_DATA_AVAIL_CARD_LEN,
                     self._notify_size)
    self._n_points += self._chunk_size

    return [t, ret]

  def stop_stream(self) -> None:
    """Stops the stream, if it was started."""

    if self._stream_started:
      spc.dw_set_param(self._spectrum, spc.SPC_M2CMD, spc.M2CMD_CARD_STOP |
                       spc.M2CMD_DATA_STOPDMA)

  def close(self) -> None:
    """Closes the connection to the Spectrum if it was opened."""

    if self._spectrum is not None:
      spc.vClose(self._spectrum)
