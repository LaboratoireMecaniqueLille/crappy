# coding: utf-8

from time import time
import numpy as np
from typing import Any, Optional, Literal
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import current_process
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from labjack import ljm
except ImportError:
  ljm = OptionalModule("ljm",
                       "Please install Labjack LJM and the ljm Python module")


@dataclass
class _Channel:
  """This class is a simple structure holding all the attributes a Labjack
  channel can have for streaming."""

  name: str

  address: Optional[int] = None
  gain: float = 1
  offset: float = 0
  make_zero: bool = False
  range: float = 10
  write_at_open: list[tuple[str, Any]] = field(default_factory=list)

  def update(self, dic_in: dict[str, Any]) -> None:
    """Updates the channel keys based on the user input."""

    for key, val in dic_in.items():
      if hasattr(self, key):
        setattr(self, key, val)

      # Handling the case when the user enters a wrong key
      else:
        logger = logging.getLogger(
          f"{current_process().name}.LabjackT7.Channel_{self.name}")
        logger.log(logging.WARNING, f"Unknown channel key : {key}, ignoring")


class T7Streamer(InOut):
  """This InOut allows controlling a Labjack T7 device in stream mode.

  It can only acquire data on the `AIN` channels. For single point mode, and
  acquisition on all channels, use the :class:`~crappy.inout.LabjackT7` InOut.

  Compared with single point acquisition, the streaming mode can achieve much
  higher data rates and has a much greater regularity in the frequency of the
  acquisition. However, fewer options are available and not all types of
  channels can be read in the streamer mode.

  For each channel, the voltage range can be tuned, and a gain and offset can
  be defined. Also, the user can decide whether the channel should be zeroed
  before starting the test or not.

  Important:
    The ``streamer`` argument of the IOBlock controlling this InOut must be set
    to :obj:`True` to enable streaming in this class. Otherwise, only single
    point acquisition can be performed.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *T7_streamer* to *T7Streamer*
  """

  def __init__(self,
               channels: Iterable[dict[str, Any]],
               device: Literal['ANY', 'T7', 'T4', 'DIGIT']  = 'ANY',
               connection: Literal['ANY', 'TCP', 'USB',
                                   'ETHERNET', 'WIFI'] = 'ANY',
               identifier: str = 'ANY',
               scan_rate: int = 100000,
               scan_per_read: int = 10000,
               resolution: int = 1) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) of the
        channels to interface with on the Labjack. Each object in this iterable
        should be a :obj:`dict` representing a single channel, and whose keys
        provide information on the channel to use. Refer to the note below for
        more information on the possible keys.
      device: The type of Labjack to open. Possible values include :
        ::

          'ANY', 'T7', 'T4', 'DIGIT'

        Only tested with `'T7'` in Crappy.
      connection: The type of connection used for interfacing with the Labjack.
        Possible values include :
        ::

          'ANY', 'TCP', 'USB', 'ETHERNET', 'WIFI'

      identifier: Any extra information allowing to further identify the
        Labjack to open, like a serial number, an IP address, or a device name.
      scan_rate: The acquisition frequency in Hz for all channels. Note that
        the sample rate (`scan_rate * num of chan`) cannot exceed `100000`. If
        it is too high it will be lowered to the highest possible value.
      scan_per_read: The number of points to read at each loop.
      resolution: The resolution of the acquisition as an integer for all
        channels. Refer to Labjack documentation for more details. The higher
        this value the better the resolution, but the lower the speed. The
        possible range is either `1` to `8` or to `12` depending on the model.
        The default is `1`.

    Note:
      - ``channels`` keys:

        - name: The name of the channel to interface with, as written on the
          Labjack's case. Ex: `'AIN0'`. In streamer mode, only the `AIN`
          channels, i.e. the analog inputs, are available.

        - gain: The measured value will be modified in Crappy as follows :
          :math:`returned\\_value = gain * measured\\_value + offset`.

        - offset: The measured value will be modified in Crappy as follows :
          :math:`returned\\_value = gain * measured\\_value + offset`

        - make_zero: If :obj:`True`, data will be acquired on this channel
          before the test starts, and a compensation value will be deduced
          so that the offset of this channel is `0`. **It will only take effect
          if the** ``make_zero_delay`` **argument of the**
          :class:`~crappy.blocks.IOBlock` **controlling the Labjack is set** !

        - range: The range of the acquisition in Volts. A range of `x` means
          that values can be read  between `-x` and `x` Volts. The possible
          values are :
          ::

            0.01, 0.1, 1, 10

    """

    self._handle = None

    super().__init__()

    channels = list(channels)

    if len(channels) * scan_rate > 100000:
      scan_rate = 100000 / len(channels)
      self.log(logging.WARNING,
               f"scan_rate is too high! Sample rate cannot exceed 100kS/s, "
               f"lowering samplerate to {scan_rate} samples/s")

    self._device = device
    self._connection = connection
    self._identifier = identifier
    self._scan_rate = scan_rate
    self._scan_per_read = scan_per_read
    self._resolution = resolution

    self._channels = list()

    # Parsing the setting dict given for each channel
    for channel in channels:

      # Checking that the name was given as it's the most important attribute
      if 'name' not in channel:
        raise AttributeError("The given channels must contain the 'name' "
                             "key !")

      # Instantiating the channel and its attributes
      chan = _Channel(name=channel['name'])
      chan.update(channel)
      chan.write_at_open.append((f"{chan.name}_RANGE", chan.range))
      chan.address, _ = ljm.nameToAddress(chan.name)

      self._channels.append(chan)

    self.log(logging.DEBUG, f"Input channels: {self._channels}")

    # these attributes will be set later
    self._n_points = 0
    self._stream_t0 = 0
    self._stream_started = False

  def open(self) -> None:
    """Opens the Labjack, parses the commands to write at open, and sends them.

    Also checks whether the scan rate chosen by the Labjack is the same as
    requested by the user.
    """

    # Opening the Labjack
    self.log(logging.INFO, "Opening the connection to the Labjack")
    self._handle = ljm.openS(self._device, self._connection, self._identifier)

    # Setting the different channels to read from on the Labjack
    write_at_open = list(chain(*(chan.write_at_open
                                 for chan in self._channels)))
    write_at_open.extend([("STREAM_SCANRATE_HZ", self._scan_rate),
                          ("STREAM_RESOLUTION_INDEX", self._resolution)])
    names, values = tuple(zip(*write_at_open))
    self.log(logging.DEBUG, f"Writing values {values} to names {names}")
    ljm.eWriteNames(handle=self._handle,
                    numFrames=len(names),
                    aNames=names,
                    aValues=values)

    # Checking if the scan rate that will be used is the same as requested
    scan_rate = ljm.eReadName(handle=self._handle, name="STREAM_SCANRATE_HZ")
    if scan_rate != self._scan_rate:
      self.log(logging.WARNING, f"Actual scan_rate: {scan_rate} instead of "
                                f"requested {self._scan_rate}")
      self._scan_rate = scan_rate

  def make_zero(self, delay: float) -> None:
    """Overriding the method of the parent class, because the user can choose
    which channels should be zeroed or not.

    It simply performs the regular zeroing, and resets the compensation for the
    channels that shouldn't be zeroed.
    
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

  def start_stream(self) -> None:
    """Starts the stream, and saves the timestamp of the moment when the stream
    started."""

    ljm.eStreamStart(handle=self._handle,
                     scansPerRead=self._scan_per_read,
                     numAddresses=len(self._channels),
                     aScanList=[chan.address for chan in self._channels],
                     scanRate=self._scan_rate)
    self._stream_t0 = time()
    self._stream_started = True

  def get_data(self) -> list[float]:
    """Reads single data points, applies the given gains and offsets, and
    returns the data along with a timestamp."""

    data = ljm.eReadNames(handle=self._handle,
                          numFrames=len(self._channels),
                          aNames=[chan.name for chan in self._channels])

    return [time()] + [val * chan.gain + chan.offset for chan, val
                       in zip(self._channels, data)]

  def get_stream(self) -> Optional[list[np.ndarray]]:
    """Acquires the stream, reshapes the data, applies the gains and offsets,
    and returns the data along with a time array."""

    if not self._stream_started:
      return

    # Acquiring the data from the Labjack and reshaping it
    raw_data, *_ = ljm.eStreamRead(self._handle)
    data = np.array(raw_data)
    data = data.reshape(len(data) // len(self._channels), len(self._channels))

    # Applying the given gains and offsets
    for i, chan in enumerate(self._channels):
      data[:, i] = chan.gain * data[:, i] + chan.offset

    # Generating the array of time values
    t = self._stream_t0 + np.arange(self._n_points, self._n_points +
                                    data.shape[0]) / self._scan_rate
    self._n_points += data.shape[0]

    return [t[:, np.newaxis], data]

  def stop_stream(self) -> None:
    """Stops the stream, if it was started."""

    if self._stream_started:
      ljm.eStreamStop(self._handle)

  def close(self) -> None:
    """Closes the connection to the Labjack, if it was opened."""

    if self._handle is not None:
      self.log(logging.INFO, "Closing the connection to the Labjack")
      ljm.close(self._handle)
