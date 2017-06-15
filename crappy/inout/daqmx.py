# coding: utf-8

import numpy as np
import time
from PyDAQmx import *

from .inout import InOut


def get_daqmx_devices_names():
  """
  Get all connected daqmx devices.
  Returns:
      a list of all connected daqmx devices.

  """
  buffer_size = 4096
  buffer = ctypes.create_string_buffer(buffer_size)
  DAQmxGetSysDevNames(buffer, buffer_size)
  print(len(buffer.value.split(",")), " devices detected: ", buffer.value)
  return buffer.value.split(",")


def listify(stuff, l):
  r = stuff if isinstance(stuff, list) else [stuff] * l
  assert len(r) == l, "Invalid list length for " + str(r)
  return r


class Daqmx(InOut):
  """
  Class to use DAQmx devices
  kwargs:
    device (str): Name of the device to open. Default: 'Dev1'
    channels (list of str/int): Names or ids of the channels to read
      default: ['ai0']
    gain (list of floats): Gains to apply to each reading
    offset (list of floats): Offset to apply to each reading
    range (list of floats): Max value for the reading (max: 5V)
      must be in [.5,1.,2.5,5.]
      default: 5
      See niDAQ api for more details
    make_zero (list of bools): If True, the average value on the channel
      at opening will be evaluated and substracted to the actual reading
      default: True
    nperscan (int): If using streamer mode, number of readings to acquire
      on each get_stream call
    samplerate (float): If using streamer mode, frequency of acquition
      when calling get_stream
    out_channels (list of str/int): names or ids of the output channels
      default: []
    out_gain (list of floats): gains to apply to the commands
      default: 1
    out_offset (list of floats): offset to apply to the commands
      default: 0
    out_range (list of floats): Max value of the output (max: 5V)
      must be in [.5,1.,2.5,5.]
      default: 5
      See niDAQ api for more details
  Note: If an argument taken as a list is given as a single value, it
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
    if a list is given, simply check the length
    else make a list of the correct length containing only the given value
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
    self.out_channels = self.out_channels if isinstance(self.out_channels, list) \
      else [self.out_channels]
    nout = len(self.out_channels)
    for i in range(nout):
      if isinstance(self.out_channels[i], int):
        self.out_channels[i] = 'ao' + str(self.out_channels[i])
    self.out_gain = np.array(listify(self.out_gain, nout))
    self.out_offset = np.array(listify(self.out_offset, nout))
    self.out_range = listify(self.out_range, nout)
    assert nin + nout, "DAQmx has no in nor out channels!"

  def open(self):
    DAQmxResetDevice(self.device)
    # IN channels
    if self.channels:
      self.handle = TaskHandle()
      self.nread = int32()
      DAQmxCreateTask("", byref(self.handle))
      for i, chan in enumerate(self.channels):
        DAQmxCreateAIVoltageChan(self.handle, self.device + "/" + chan, "",
                                 DAQmx_Val_Cfg_Default,
                                 0, self.range[i],
                                 DAQmx_Val_Volts, None)
      if any(self.make_zero):
        off = self.eval_offset()
        for i, make_zero in enumerate(self.make_zero):
          if make_zero:
            self.offset[i] += off[i]
    # OUT channels
    if self.out_channels:
      self.out_handle = TaskHandle()
      DAQmxCreateTask("", byref(self.out_handle))
      for i, chan in enumerate(self.out_channels):
        DAQmxCreateAOVoltageChan(self.out_handle, self.device + "/" + chan, "",
                                 0, self.out_range[i],
                                 DAQmx_Val_Volts, None)
      DAQmxStartTask(self.out_handle)

  def get_data(self):
    """
    Returns a tuple of length len(self.channels)+1
    first element is the time, others are readings of each channel
    """
    return [i[0] for i in self.get_stream(1)]

  def get_stream(self, npoints=None):
    """
    Read the analog voltage on specified channels
    Args:
        channels: List of ints, the INDEX of the channels to read
          Ex: if self.channels = ['ai1','ai2','ai4']
            channels = [1,2] will read ai2 and ai4
          if None (default) will read all opened channels
        npoints: number of values to read.
          if None, will use the value of self.nperscan
    Returns:
        a tuple of len(self.channels)+1 lists of length npoints
        first list is the time, the others are the read voltages
    """
    if npoints is None:
      npoints = self.nperscan
    DAQmxCfgSampClkTiming(self.handle, "",
                          self.sample_rate, DAQmx_Val_Rising,
                          DAQmx_Val_FiniteSamps,
                          npoints + 1)
    DAQmxStartTask(self.handle)
    data = np.empty((len(self.channels), npoints), dtype=np.float64)
    t0 = time.time()
    # DAQmx Read Code
    DAQmxReadAnalogF64(self.handle, npoints, 10.0,
                       DAQmx_Val_GroupByChannel, data,
                       npoints * len(self.channels), byref(self.nread), None)
    t = time.time()
    # DAQmx Stop Code
    DAQmxStopTask(self.handle)
    t1 = ((
          t + t0) - npoints / self.sample_rate) / 2  # Estimated starting of the acq
    return [[t1 + i / self.sample_rate for i in range(npoints)]] \
           + [data[i, :] * self.gain[i] + self.offset[i] for i in
              range(len(self.channels))]

  def set_cmd(self, *args):
    """
    Set the output(s) to the specified value
    Takes n arguments, n being the number of channels open at init
    ith argument is the value to set to the ith channel
    """
    assert len(args) == len(self.out_channels)
    data = np.array(args, dtype=np.float64) * self.out_gain + self.out_offset
    DAQmxWriteAnalogF64(self.out_handle, 1, 1, 10.0, DAQmx_Val_GroupByChannel,
                        data, None, None)

  def close(self):
    """Close the connection."""
    if self.handle:
      DAQmxStopTask(self.handle)
      DAQmxClearTask(self.handle)
    if self.out_handle:
      DAQmxStopTask(self.out_handle)
      DAQmxClearTask(self.out_handle)
