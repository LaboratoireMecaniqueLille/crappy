# coding: utf-8
from __future__ import print_function,division

import numpy
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

def listify(stuff,l):
  r = stuff if isinstance(stuff,list) else [stuff]*l
  assert len(r) == l,"Invalid list length for "+str(r)
  return r


class Daqmx(InOut):
  """
  Class to use DAQmx devices
  """
  def __init__(self, **kwargs):
    InOut.__init__(self)
    # For now, kwargs like in_gain are eqivalent to gain
    # (it is for consitency with out_gain, out_channels, etc...)
    for arg in kwargs:
      if arg in kwargs and arg.startswith('in_'):
        kwargs[arg[3:]] = kwargs[arg]
        del kwargs[arg]
    for arg,default in [('device','Dev2'),
                        ('channels',['ai0']),
                        ('gain',1),
                        ('offset',0),
                        ('range',5),
                        ('make_zero',True),
                        ('sample_rate',10000),
                        ('out_channels',[]),
                        ('out_gain',1),
                        ('out_offset',0),
                        ('out_range',5)
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    assert len(kwargs) == 0,"Daqmx got unsupported arg(s)"+str(kwargs)
    self.check_vars()
      
  def check_vars(self):
    #IN channels
    self.channels = self.channels if isinstance(self.channels,list)\
               else [self.channels]
    nin = len(self.channels)
    for i in range(nin):
      if isinstance(self.channels[i],int):
        self.channels[i] = 'ai'+str(self.channels[i])
    self.gain = listify(self.gain,nin)
    self.offset = listify(self.offset,nin)
    self.range = listify(self.range,nin)
    self.make_zero = listify(self.make_zero,nin)
    #OUT channels
    self.out_channels = self.out_channels if isinstance(self.out_channels,list)\
               else [self.out_channels]
    nout = len(self.out_channels)
    for i in range(nout):
      if isinstance(self.out_channels[i],int):
        self.out_channels[i] = 'ao'+str(self.out_channels[i])
    self.out_gain = listify(self.out_gain,nout)
    self.out_offset = listify(self.out_offset,nout)
    self.out_range = listify(self.out_range,nout)
    assert nin+nout,"DAQmx has no in nor out channels!"

  def open(self):
    DAQmxResetDevice(self.device)
    # IN channels
    self.handle = TaskHandle()
    self.nread = int32()
    DAQmxCreateTask("", byref(self.handle))
    for i,chan in enumerate(self.channels):
      DAQmxCreateAIVoltageChan(self.handle, self.device+"/"+chan, "",
                             DAQmx_Val_Cfg_Default,
                             0, self.range[i],
                             DAQmx_Val_Volts, None)
    if any(self.make_zero):
      off = self.eval_offset()
      for i,make_zero in enumerate(self.make_zero):
        if make_zero:
          self.offset[i] += off[i]
    # OUT channels
    # ...

  def get_data(self):
    return [i[0] for i in self.get_stream(1)]

  def get_stream(self, npoints=100):
    """
    Read the analog voltage on specified channels
    Args:
        channels: List of ints, the INDEX of the channels to read
          Ex: if self.channels = ['ai1','ai2','ai4']
            channels = [1,2] will read ai2 and ai4
          if None (default) will read all opened channels
        npoints: number of values to read.

    Returns:
        a tuple of len(self.channels)+1 lists of length npoints
        first list is the time, the others are the read voltages
    """
    DAQmxCfgSampClkTiming(self.handle, "",
                          self.sample_rate, DAQmx_Val_Rising,
                          DAQmx_Val_FiniteSamps,
                          npoints + 1)
    DAQmxStartTask(self.handle)
    data = numpy.empty((len(self.channels),npoints), dtype=numpy.float64)
    t0 = time.time()
    # DAQmx Read Code
    DAQmxReadAnalogF64(self.handle, npoints, 10.0,
                       DAQmx_Val_GroupByChannel, data,
                       npoints*len(self.channels), byref(self.nread), None)
    t = time.time()
    # DAQmx Stop Code
    DAQmxStopTask(self.handle)
    t1 = ((t+t0)-npoints/self.sample_rate)/2 # Estimated starting of the acq
    return [[t1+i/self.sample_rate for i in range(npoints)]]\
     +[data[i,:]*self.gain[i]+self.offset[i] for i in range(len(self.channels))]


  def close(self):
    """Close the connection."""
    if self.handle:
      # DAQmx Stop Code
      DAQmxStopTask(self.handle)
      DAQmxClearTask(self.handle)
