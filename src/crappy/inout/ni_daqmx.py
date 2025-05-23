# coding: utf-8

from time import time
import numpy as np
from typing import Optional, Any
from collections.abc import Iterable
from dataclasses import dataclass, field
from re import fullmatch
from collections import defaultdict
from itertools import chain
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import nidaqmx
  from nidaqmx import stream_readers, stream_writers
except (ModuleNotFoundError, ImportError):
  nidaqmx = OptionalModule("nidaqmx")
  stream_readers = stream_writers = nidaqmx

thcp_map = {"B": 10047,
            "E": 10055,
            "J": 10072,
            "K": 10073,
            "N": 10077,
            "R": 10082,
            "S": 10085,
            "T": 10086}

unit_map = {"C": 10143,
            "F": 10144,
            "R": 10145,
            "K": 10325}


@dataclass
class _Channel:
  """This class is a simple structure holding all the attributes a NI DAQmx
  channel can have."""

  meas_type: str = 'voltage'
  name: Optional[str] = None
  kwargs: dict[str, Any] = field(default_factory=dict)

  def update(self, dic_in: dict[str, Any]) -> None:
    """Updates the channel keys based on the user input."""

    for key, val in dic_in.items():
      # The 'name' and 'type' keys are handled separately
      if key == 'name':
        self.name = val
      elif key == 'type':
        self.meas_type = val

      # All the other keys are put together in the kwargs attribute
      else:
        self.kwargs |= {key: val}


class NIDAQmx(InOut):
  """This class can drive data acquisition hardware from National Instruments.

  It is similar to :class:`~crappy.inout.DAQmx` InOut, except it relies on the
  :mod:`nidaqmx` module. It was written and tested on a USB 6008 DAQ board, but
  should work with other instruments as well.

  It can read single data points from digital and analog channels, read streams
  of data from analog channels, and set the voltage of analog and digital
  output channels. For analog input channels, several types of acquisition can
  be performed, like voltage, resistance, current, etc.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Nidaqmx* to *NIDAQmx*
  """

  def __init__(self,
               channels: Iterable[dict[str, Any]],
               sample_rate: float = 100,
               n_samples: Optional[int] = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      channels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        :obj:`dict` holding information on the channels to read data from or
        write data to. See below for the mandatory and optional keys for the
        dicts. Note that in streamer mode, the digital input channels are not
        available for reading. Also, only one type of analog input channel at a
        time can be read in streamer mode, with no restriction on the number of
        channels of this type.
      sample_rate: The target sample rate for data acquisition in streamer
        mode, given as a :obj:`float`. Default is `100` SPS.

        .. versionchanged:: 1.5.10 renamed from *samplerate* to *sample_rate*
      n_samples: The number of samples to acquire per chunk of data in streamer
        mode. Default is 20% of ``sample_rate``.

        .. versionchanged:: 1.5.10 renamed from *nsamples* to *n_samples*


    Note:
      - ``channels`` keys:

        - name: The name of the channel to drive, given with the following
          syntax :
          ::

            'DevX/[a/d][i/o]Y'

          With `X` the index of the device, and `Y` the line on which the
          channel is. `d` stands for digital, `a` for analog, `i` for input and
          `o` for output. For digital channels, `DevX/d[i/o]Y` is internally
          converted to `DevX/port<Y // 8>/line<Y % 8>`. Example of a valid
          name : `Dev1/ao3`.

        - type: The type of data to read, for analog input channels. This field
          can take many different values, refer to the documentation of the
          :mod:`nidaqmx` for more details. This field is internally used for
          calling the method : :meth:`nidaqmx.task.add_ai_[type]_chan`. The
          default for this field is `'voltage'`, possible values include
          `'thrmcpl'`, `'bridge'`, `'current'` and `'resistance'`.

        - All the other keys will be given as kwargs to the
          :meth:`nidaqmx.task.add_ai_[type]_chan` method for analog input
          channels, to the :meth:`nidaqmx.task.add_ao_voltage_chan` for analog
          output channels, to :meth:`nidaqmx.task.add_do_chan` for digital
          output channels, and to :meth:`nidaqmx.task.add_di_chan` for digital
          input channels. Refer to :mod:`nidaqmx` documentation for the
          possible arguments and values. Note that for the `'thrmcpl'` analog
          input channel type, the `'thermocouple_type'` argument must be given
          as a letter, same for the `'units'` argument. They will be parsed
          internally. Also note that for the analog output channels and the
          analog input channels of type `'voltage'`, the `'min_val'` and
          `'max_val'` arguments are internally set by default to `0` and `5`.
    """

    super().__init__()

    # Setting the number of samples per acquisition for streamer mode
    if n_samples is None:
      self._n_samples = max(1, int(sample_rate / 5))
    else:
      self._n_samples = n_samples

    self._sample_rate = sample_rate

    # These attributes will be set later
    self._task_ao = None
    self._stream_ao = None
    self._task_di = None
    self._stream_di = None
    self._task_do = None
    self._stream_do = None
    self._stream_started = False
    # For analog inputs a dict is needed as each type is handled separately
    self._tasks_ai = dict()
    self._stream_ai = dict()

    self._digital_in = list()
    self._digital_out = list()
    self._analog_out = list()
    # Here as well a dict is needed for handling each analog input type
    self._analog_in = defaultdict(list)

    for channel in channels:

      # Making sure each channel has a 'name' attribute
      if 'name' not in channel:
        raise AttributeError("The given channels must contain the 'name' "
                             "key !")

      # Parsing the channel name to retrieve info from it
      match = fullmatch(r'(.+)/(.+)(\d+)', channel['name'])
      if match is not None:
        dev, type_, num = match.groups()
        num = int(num)
      else:
        raise AttributeError(f"Invalid format for the channel name "
                             f": {channel['name']} !\nIt should be "
                             f"'Dev<dev num>/[a/d][i/o]<chan num>'")

      # Creating a _Channel object holding the information on the channel
      chan = _Channel()
      chan.update(channel)

      # Saving the channel to the right place and performing specific actions
      if type_ == 'ai':
        self._analog_in[chan.meas_type].append(chan)
      elif type_ == 'ao':
        self._analog_out.append(chan)
      elif type_ == 'di':
        chan.name = f"{dev}/port{num // 8}/line{num % 8}"
        self._digital_in.append(chan)
      elif type_ == 'do':
        chan.name = f"{dev}/port{num // 8}/line{num % 8}"
        self._digital_out.append(chan)
      else:
        raise ValueError(f"Wrong channel type : {type_} !\nIt should be "
                         f"either 'ai', 'ao', 'di', or 'do'.")

  def open(self) -> None:
    """Creates tasks and streams for analog output, digital input, digital
    output, and each type of analog input channels."""

    # Creating one task for each type of analog input channel
    self._tasks_ai = {type_: nidaqmx.Task() for type_ in self._analog_in}

    # Iterating over all the analog input channels
    for type_, channels in self._analog_in.items():
      for chan in channels:

        # Setting the min and max voltage for the voltage analog input channels
        if type_ == 'voltage':
          chan.kwargs['max_val'] = chan.kwargs.get('max_val', 5)
          chan.kwargs['min_val'] = chan.kwargs.get('min_val', 0)

        # Parsing the thermocouple related arguments
        if 'thermocouple_type' in chan.kwargs:
          chan.kwargs['thermocouple_type'] = thcp_map[
            chan.kwargs['thermocouple_type']]
          # Included in the if as 'units' is an arg for other types of channels
          if 'units' in chan.kwargs:
            chan.kwargs['units'] = thcp_map[chan.kwargs['units']]

        # Adding the channel to the task with the given kwargs
        try:
          func = getattr(self._tasks_ai[type_].ai_channels,
                         f'add_ai_{type_}_chan')
          func(chan.name, **chan.kwargs)
        except AttributeError:
          raise ValueError(f"Invalid channel type : {type_}")

    # Opening a stream for each analog input task
    self.log(logging.INFO, "Opening the streams for the analog input channels")
    self._stream_ai = {
      type_: stream_readers.AnalogMultiChannelReader(task.in_stream)
      for type_, task in self._tasks_ai.items()}

    # Creating a task and a stream for all the analog output channels
    if self._analog_out:
      self.log(logging.INFO,
               "Opening the streams for the analog output channels")
      self._task_ao = nidaqmx.Task()
      self._stream_ao = stream_writers.AnalogMultiChannelWriter(
        self._task_ao.out_stream, auto_start=True)

      # Setting the min and max voltage for all the analog output channels
      for chan in self._analog_out:
        chan.kwargs['max_val'] = chan.kwargs.get('max_val', 5)
        chan.kwargs['min_val'] = chan.kwargs.get('min_val', 0)
        self._task_ao.ao_channels.add_ao_voltage_chan(chan.name, **chan.kwargs)

    # Creating a task and a stream for all the digital input channels
    if self._digital_in:
      self.log(logging.INFO,
               "Opening the streams for the digital input channels")
      self._task_di = nidaqmx.Task()

      for chan in self._digital_in:
        self._task_di.di_channels.add_di_chan(chan.name, **chan.kwargs)

      self._stream_di = stream_readers.DigitalMultiChannelReader(
        self._task_di.in_stream)

    # Creating a task and a stream for all the digital output channels
    if self._digital_out:
      self.log(logging.INFO,
               "Opening the streams for the digital output channels")
      self._task_do = nidaqmx.Task()

      for chan in self._digital_out:
        self._task_do.do_channels.add_do_chan(chan.name, **chan.kwargs)

      self._stream_do = stream_writers.DigitalMultiChannelWriter(
        self._task_do.out_stream)

  def start_stream(self) -> None:
    """Starts the streaming task for analog input channels.

    Data can be acquired via streaming for multiple channels, but only for one
    type of channel.
    """

    # Making sure there's only one type of channel to read data from
    if len(self._tasks_ai) > 1:
      raise IOError("Stream mode can only open one type of channel !")
    elif len(self._tasks_ai) < 1:
      raise IOError("There's no analog in channel to read data from !")

    # Starting the streaming task, there should be only one
    for task in self._tasks_ai.values():
      task.timing.cfg_samp_clk_timing(
        self._sample_rate,
        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    self._stream_started = True

  def get_data(self) -> list[float]:
    """Reads data from the analog and digital input channels, and returns it
    along with a timestamp.

    Data from the analog channels is read first, and then data from the digital
    channels. Data is returned in the same order as it was acquired.
    """

    ret = [time()]

    # Reading the analog channels
    if self._analog_in:
      data = np.empty(len(list(chain(*self._analog_in.values()))))
      i = 0
      for type_, stream in self._stream_ai.items():
        stream.read_one_sample(data[i:i + len(self._analog_in[type_])])
        i += len(self._analog_in[type_])

      ret.extend(list(data))

    # Reading the digital channels
    if self._digital_in:
      data = np.empty((len(self._digital_in), 1), dtype=bool)
      self._stream_di.read_one_sample_multi_line(data)

      ret.extend(list(data[:, 0]))

    return ret

  def get_stream(self) -> Optional[list[np.ndarray]]:
    """Reads data from the device, and returns it in an array along with an
    array holding the timestamps.

    Only data from analog input channels can be read, this method cannot read
    stream data from digital input channels.
    """

    if not self._stream_started:
      return

    # Creating the container for the data
    data = np.empty((len(list(chain(*self._analog_in.values()))),
                     self._n_samples))
    # # Creating the array holding the timestamps
    t = time() + np.arange(0, self._n_samples) / self._sample_rate
    # Actually reading the data from the device
    for type_, stream in self._stream_ai.items():
      stream.read_many_sample(data, self._n_samples)

    return [t, data]

  def set_cmd(self, *cmd: float) -> None:
    """Sets the analog and digital output channels according to the given
    command values.

    The first command values correspond to the analog channels, the remaining
    ones correspond to the digital channels. It might be that not all channels
    are set if the number of commands doesn't match the number of channels.
    """

    # Setting the analog channels
    if self._analog_out:
      self._stream_ao.write_one_sample(np.array(cmd[:len(self._analog_out)],
                                       dtype=np.float64))

    # Setting the digital channels
    if self._digital_out:
      self._stream_do.write_one_sample_multi_line(
        np.array(cmd[len(self._analog_out):],
                 dtype=bool).reshape(len(self._digital_out), 1))

  def stop_stream(self) -> None:
    """Stops all the acquisition tasks."""

    if self._stream_started:
      for task in self._tasks_ai.values():
        task.stop()
        task.timing.cfg_samp_clk_timing(
          self._sample_rate,
          sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

    self._stream_started = False

  def close(self) -> None:
    """Stops all the acquisition tasks, and closes the connections to the
    device."""

    if self._analog_in:
      self.log(logging.INFO, "Closing the streams for the analog input "
                             "channels")
      for task in self._tasks_ai.values():
        task.stop()

    if self._analog_out:
      self.log(logging.INFO, "Closing the streams for the analog output "
                             "channels")
      self._task_ao.stop()

    if self._stream_started:
      self.stop_stream()

    if self._digital_in:
      self.log(logging.INFO, "Closing the streams for the digital input "
                             "channels")
      self._task_di.close()

    if self._digital_out:
      self.log(logging.INFO, "Closing the streams for the digital output "
                             "channels")
      self._task_do.close()
