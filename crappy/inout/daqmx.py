# coding: utf-8

import numpy as np
import time

from .inout import InOut
from .._global import OptionalModule
try:
  import PyDAQmx
except (ModuleNotFoundError,ImportError):
  PyDAQmx = OptionalModule("PyDAQmx")


def get_daqmx_devices_names():
  """
  Get all connected daqmx devices.

  Returns:
    A list of all connected daqmx devices.

  """
  buffer_size = 4096
  buffer = PyDAQmx.create_string_buffer(buffer_size)
  PyDAQmx.DAQmxGetSysDevNames(buffer, buffer_size)
  print(len(buffer.value.split(",")), " devices detected: ", buffer.value)
  return buffer.value.split(",")


def listify(stuff, l):
  r = stuff if isinstance(stuff, list) else [stuff] * l
  assert len(r) == l, "Invalid list length for " + str(r)
  return r


class Daqmx(InOut):
  """
  Class to use DAQmx devices.

  Kwargs:
    - device (str, default: 'Dev1'): Name of the device to open.
    - channels (list of str/int, default: ['ai0']): Names or ids of the channels
      to read.
    - gain (list of floats): Gains to apply to each reading.
    - offset (list of floats): Offset to apply to each reading.
    - range (list of floats, [.5, 1., 2.5, 5.], max: 5V, default: 5): Max value
      for the reading.

      Note:
        See niDAQ api for more details.

    - make_zero (list of bools, default: True): If True, the average value on
      the channel at opening will be evaluated and substracted to the actual
      reading.
    - nperscan (int): If using streamer mode, number of readings to acquire
      on each get_stream call.
    - samplerate (float): If using streamer mode, frequency of acquition
      when calling get_stream.
    - out_channels (list of str/int, default: []): Names or ids of the output
      channels.
    - out_gain (list of floats, default: 1): Gains to apply to the commands.
    - out_offset (list of floats, default: 0): Offset to apply to the commands.
    - out_range (list of floats, [.5, 1., 2.5, 5.], max: 5V, default: 5): Max
      value of the output.

      Note:
        See niDAQ api for more details.

        If an argument taken as a list is given as a single value, it
        will be applied to all channels.

  """

  def __init__(self, **kwargs):
    InOut.__init__(self)
    # For now, kwargs like in_gain are eqivalent to gain
    # (it is for consitency with out_gain, out_channels, etc...)
    for arg in kwargs:
      if arg in kwargs and arg.startswith('in_'):
        kwargs[arg[3:]] = kwargs[arg]
        del kwargs[arg]
    for arg, default in [('device', 'Dev1'),
                         ('channels', ['ai0']),
                         ('gain', 1),
                         ('offset', 0),
                         ('range', 5),  # Unipolar
                         ('make_zero', True),
                         ('nperscan', 1000),
                         ('sample_rate', 10000),
                         ('out_channels', []),
                         ('out_gain', 1),
                         ('out_offset', 0),
                         ('out_range', 5)  # Unipolar
                         ]:
      if arg in kwargs:
        setattr(self, arg, kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self, arg, default)
    assert len(kwargs) == 0, "Daqmx got unsupported arg(s)" + str(kwargs)
    self.check_vars()

  def check_vars(self):
    """
    Turns the settings into lists of the same length, each index standing for
    one channel.

    Note:
      If a list is given, simply check the length.

      Else make a list of the correct length containing only the given value.

    """
    # IN channels
    self.channels = self.channels if isinstance(self.channels, list) \
      else [self.channels]
    nin = len(self.channels)
    for i in range(nin):
      if isinstance(self.channels[i], int):
        self.channels[i] = 'ai' + str(self.channels[i])
    self.gain = listify(self.gain, nin)
    self.offset = listify(self.offset, nin)
    self.range = listify(self.range, nin)
    self.make_zero = listify(self.make_zero, nin)
    # OUT channels
    self.out_channels = self.out_channels if \
        isinstance(self.out_channels, list) else\
        [self.out_channels]
    nout = len(self.out_channels)
    for i in range(nout):
      if isinstance(self.out_channels[i], int):
        self.out_channels[i] = 'ao' + str(self.out_channels[i])
    self.out_gain = np.array(listify(self.out_gain, nout))
    self.out_offset = np.array(listify(self.out_offset, nout))
    self.out_range = listify(self.out_range, nout)
    assert nin + nout, "DAQmx has no in nor out channels!"

  def open(self):
    PyDAQmx.DAQmxResetDevice(self.device)
    self.handle,self.out_handle = None,None
    # IN channels
    if self.channels:
      self.handle = PyDAQmx.TaskHandle()
      self.nread = PyDAQmx.int32()
      PyDAQmx.DAQmxCreateTask("", PyDAQmx.byref(self.handle))
      for i, chan in enumerate(self.channels):
        PyDAQmx.DAQmxCreateAIVoltageChan(self.handle,
                                 self.device + "/" + chan, "",
                                 PyDAQmx.DAQmx_Val_Cfg_Default,
                                 0, self.range[i],
                                 PyDAQmx.DAQmx_Val_Volts, None)
      if any(self.make_zero):
        off = self.eval_offset()
        for i, make_zero in enumerate(self.make_zero):
          if make_zero:
            self.offset[i] += off[i]
    # OUT channels
    if self.out_channels:
      self.out_handle = PyDAQmx.TaskHandle()
      PyDAQmx.DAQmxCreateTask("", PyDAQmx.byref(self.out_handle))
      for i, chan in enumerate(self.out_channels):
        PyDAQmx.DAQmxCreateAOVoltageChan(self.out_handle,
                                 self.device + "/" + chan, "",
                                 0, self.out_range[i],
                                 PyDAQmx.DAQmx_Val_Volts, None)
      PyDAQmx.DAQmxStartTask(self.out_handle)

  def get_data(self):
    """
    Returns a tuple of length len(self.channels)+1.

    First element is the time, others are readings of each channel.
    """
    return [i[0] for i in self.get_single(1)]

  def get_single(self, npoints=None):
    """
    Read the analog voltage on specified channels.

    Args:
      - channels (List of ints, default: None): The INDEX of the channels to read.

        Example:
          if self.channels = ['ai1','ai2','ai4'],
          channels = [1,2] will read ai2 and ai4.

        Note:
          If None will read all opened channels.

      - npoints: Number of values to read.

        Note:
          If None, will use the value of self.nperscan.

    Returns:
      A tuple of len(self.channels)+1 lists of length npoints.

      First list is the time, the others are the read voltages.

    """
    if npoints is None:
      npoints = self.nperscan
    PyDAQmx.DAQmxCfgSampClkTiming(self.handle, "",
                          self.sample_rate, PyDAQmx.DAQmx_Val_Rising,
                          PyDAQmx.DAQmx_Val_FiniteSamps,
                          npoints + 1)
    PyDAQmx.DAQmxStartTask(self.handle)
    data = np.empty((len(self.channels), npoints), dtype=np.float64)
    t0 = time.time()
    # DAQmx Read Code
    PyDAQmx.DAQmxReadAnalogF64(self.handle, npoints, 10.0,
                       PyDAQmx.DAQmx_Val_GroupByChannel, data,
                       npoints * len(self.channels),
                       PyDAQmx.byref(self.nread), None)
    t = time.time()
    # DAQmx Stop Code
    PyDAQmx.DAQmxStopTask(self.handle)
    # Estimated starting of the acq
    t1 = ((
          t + t0) - npoints / self.sample_rate) / 2
    return [[t1 + i / self.sample_rate for i in range(npoints)]] \
           + [data[i, :] * self.gain[i] + self.offset[i] for i in
              range(len(self.channels))]

  def set_cmd(self, *args):
    """
    Set the output(s) to the specified value.

    Note:
      Takes n arguments, n being the number of channels open at init.

      ith argument is the value to set to the ith channel.

    """
    assert len(args) == len(self.out_channels)
    data = np.array(args, dtype=np.float64) * self.out_gain + self.out_offset
    PyDAQmx.DAQmxWriteAnalogF64(self.out_handle, 1, 1, 10.0,
        PyDAQmx.DAQmx_Val_GroupByChannel, data, None, None)

  def close(self):
    """Close the connection."""
    if self.handle:
      PyDAQmx.DAQmxStopTask(self.handle)
      PyDAQmx.DAQmxClearTask(self.handle)
    if self.out_handle:
      PyDAQmx.DAQmxStopTask(self.out_handle)
      PyDAQmx.DAQmxClearTask(self.out_handle)
