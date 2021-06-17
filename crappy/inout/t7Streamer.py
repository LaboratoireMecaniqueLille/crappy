# coding: utf-8

from time import time
import numpy as np

from .inout import InOut
from .._global import OptionalModule
try:
  from labjack import ljm
except ImportError:
  ljm = OptionalModule("ljm",
      "Please install Labjack LJM and the ljm Python module")


class T7_streamer(InOut):
  """Class to use stream mode with Labjack T7 devices.

  Note:
    For single modes, see :ref:`Labjack T7`.

    You can use :ref:`IOBlock` with ``streamer=True`` to read data at high
    frequency from the Labjack. Streamer mode makes the Labjack unavailable for
    any other operation (single acquisition, `DAC` or `DIO`).

    You can specify each channel as a :obj:`dict`, allowing to set
    channel-specific settings such as gain, offset (computed on the host
    machine as this feature is not available on board with streamer mode),
    range and ability to zero the reading at startup.
  """

  def __init__(self,
               device='ANY',
               connection='ANY',
               identifier='ANY',
               channels=None,
               scan_rate=100000,
               scan_per_read=10000,
               resolution=1):
    """Sets the args and initializes the parent class.

    Args:
      device (:obj:`str`, optional): The type of the device to open. Possible
        values include:
        ::

          'ANY', 'T7', 'T4', 'DIGIT'

        Only tested with `'T7'` in Crappy.
      connection (:obj:`str`, optional): How is the Labjack connected ?
        Possible values include:
        ::

          'ANY', 'TCP', 'USB', 'ETHERNET', 'WIFI'

      identifier (:obj:`str`, optional): Something to identify the Labjack. It
        can be a serial number, an IP address, or a device name.
      channels (:obj:`list`, optional): Channels to use and their settings. It
        must be a :obj:`list` of :obj:`dict`.
      scan_rate (:obj:`int`, optional): The acquisition frequency in Hz for the
        channels. Note that the sample rate (`scan_rate * num of chan`) cannot
        exceed `100000`. If it is too high it will be lowered to the highest
        possible value.
      scan_per_read (:obj:`int`, optional): The number of points to read during
        each loop.
      resolution (:obj:`int`, optional): The resolution index for all channels.
        The bigger this value the better the resolution, but the lower the
        speed. The possible range is either `1` to `8` or to `12` according to
        the model.

    Note:
      - ``channels`` keys:

        - name (:obj:`str`): The name of the channel according to Labjack's
          naming convention. Ex: `'AIN0'`. This will be used to define the
          direction (in/out) and the available settings.

          It can be:
            - `AINx`: An analog input, if gain and/or offset is given, the
              integrated slope mechanism will be used with the extended
              features registers. It can also be used for thermocouples (see
              below). You can use any EF by using the ``write_at_open`` and
              ``to_read`` keys if necessary.
            - `(T)DACx`: An analog output, you can specify gain and/or offset.
            - `(E/F/C/M IOx)`: Digital in/outputs. You can specify the
              direction.

        - gain (:obj:`float`, default: `1`): A numeric value that will multiply
          the given value for inputs and outputs.
        - offset (:obj:`float`, default: `0`): Will be added to the value.

          For inputs:
          ::

            returned_value = gain * measured_value + offset

          For outputs:
          ::

            set_value = gain * given_value + offset.

          Where `measured_value` and `set_values` are in Volts.

        - make_zero (:obj:`bool`): AIN only, if :obj:`True` the input value
          will be evaluated at startup and the offset will be adjusted to
          return `0` (or the offset if any).
        - range (:obj:`float`, default: `10`): The range of the acquisition in
          Volts. A range of `x` means that values can be read  between `-x` and
          `x` Volts. The possible values are:
          ::

            0.01, 0.1, 1, 10

    """

    InOut.__init__(self)
    self.device = device
    self.connection = connection
    self.identifier = identifier
    self.channels = [{'name': 'AIN0'}] if channels is None else channels
    self.scan_rate = scan_rate
    self.scan_per_read = scan_per_read
    self.resolution = resolution

    default = {'gain': 1, 'offset': 0, 'make_zero': False, 'range': 10}
    if len(self.channels) * self.scan_rate > 100000:
      self.scan_rate = 100000 / len(self.channels)
      print("[Labjack] Warning! scan_rate is too high! Sample rate cannot "
          "exceed 100kS/s, lowering samplerate to",
            self.scan_rate, "samples/s")
    self.chan_list = []
    for d in self.channels:
      if isinstance(d, str):
        d = {'name': d}
      for k in ['gain', 'offset', 'make_zero', 'range']:
        if k not in d:
          d[k] = default[k]
      if 'to_write' not in d:
        d['to_write'] = []
      d['to_write'].append(("_RANGE", d['range']))
      d['to_read'] = ljm.nameToAddress(d['name'])[0]
      self.chan_list.append(d)

  def open(self):
    self.handle = ljm.openS(self.device, self.connection, self.identifier)
    names, values = [], []
    for c in self.chan_list:
      if "to_write" in c:
        for n, v in c['to_write']:
          names.append(c['name'] + n)
          values.append(v)
    # names.append("STREAM_NUM_ADDRESSES"); values.append(len(self.channels))
    names.append("STREAM_SCANRATE_HZ")
    values.append(self.scan_rate)
    names.append("STREAM_RESOLUTION_INDEX")
    values.append(self.resolution)
    ljm.eWriteNames(self.handle, len(names), names, values)
    scan_rate = ljm.eReadName(self.handle, "STREAM_SCANRATE_HZ")
    if scan_rate != self.scan_rate:
      print("[Labjack] Actual scan_rate:", scan_rate,
          "instead of", self.scan_rate)
      self.scan_rate = scan_rate
    if any([c.get("make_zero", False) for c in self.chan_list]):
      print("[Labjack] Please wait during offset evaluation...")
      off = self.eval_offset()
      for i, c in enumerate(self.chan_list):
        if 'make_zero' in c and c['make_zero']:
          c['offset'] += c['gain']*off[i]
    self.n = 0  # Number of data points (to rebuild time)

  def get_data(self):
    """Short version, only used for :meth:`InOut.eval_offset`."""

    return [time()] + ljm.eReadNames(self.handle, len(self.chan_list),
                              [c['name'] for c in self.chan_list])

  def start_stream(self):
    ljm.eStreamStart(self.handle, self.scan_per_read, len(self.chan_list),
        [c['to_read'] for c in self.chan_list], self.scan_rate)
    self.stream_t0 = time()

  def stop_stream(self):
    ljm.eStreamStop(self.handle)

  def get_stream(self):
    a = np.array(ljm.eStreamRead(self.handle)[0])
    r = a.reshape(len(a) // len(self.channels), len(self.channels))
    for i, c in enumerate(self.chan_list):
      r[:, i] = c['gain'] * r[:, i] + c['offset']
    t = self.stream_t0 + \
        np.arange(self.n, self.n + r.shape[0]) / self.scan_rate
    self.n += r.shape[0]
    return [t, r]

  def close(self):
    ljm.close(self.handle)
