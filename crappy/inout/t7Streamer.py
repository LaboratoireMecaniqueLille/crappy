#coding: utf-8
from __future__ import print_function,division

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
  """
  Class to use stream mode with Labjack T7 devices.

  For single modes, see Labjack_t7. You can use IOBlock with streamer=True
  to read data at high frequency from the Labjack. Streamer mode makes
  the Labjack unavailable for any other operation (single acq, DAC or DIO).
  You can specify each channel as a dict, allowing to set channel-specific
  settings such as gain, offset (computed on the host machine as this feature
  is not available on board with streamer mode), range and ability to zero
  the reading at startup.
  Args:
    device: The type of the device to open (str). Ex: 'T7'.
       Can be 'ANY' (default).
    connection: How is the Labjack connected ? (str). Ex: 'USB', 'ETHERNET',..
      Can be 'ANY' (default).
    identifier: Something to identify the Labjack (str)
      It can be a name, serial number or functionality.
      Can be 'ANY' (default).
    channels: Channels to use and their settings. It must be a list of dicts.
    scan_rate: The acquisition frequency in Hz (int) for the channels
      Note that the sample rate (scan_rate*num of chan) cannot exceed 100000
      If too high, it will be lowered to the highest possible value.
      (default = 100000)
    scan_per_read: The number of points to read on each loop. Default=10000
    resolution: The resolution index for all channels. The higher it is, the
      slower the acquisition will be (but more precise). It cannot be set for
      each channel in streamer mode.

  Channel keys:
    name: The name of the channel according to Labjack's naming convention
      (str). Ex: 'AIN0'. This will be used to define the direction (in/out)
      and the available settings. Only inputs can be used in stream mode.

    gain: A numeric value that will multiply the given value for inputs
      and outputs. Default=1

    offset: Will be added to the value. Default=0
      returned_value = gain*measured_value+offset
      Where measured_value is in Volts.

    make_zero: If True the input value will be evaluated at startup
      and the offset will be adjusted to return 0 (or the offset if any).

    range: 10/1/.1/.01. The range of the acquisition (V).
      10 means -10V>+10V default=10
  """
  def __init__(self, **kwargs):
    InOut.__init__(self)
    for arg, default in [
       ('device', 'ANY'), # Model (T7, DIGIT,...)
       ('connection', 'ANY'), # Connection (USB,ETHERNET,...)
       ('identifier', 'ANY'), # Identifier (serial n°, ip,..)
       ('channels', [{'name':'AIN0'}]),
       ('scan_rate',100000),
       ('scan_per_read',10000),
       ('resolution',1)
     ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert len(kwargs) == 0, "T7_streamer got unsupported arg(s)" + str(kwargs)
    default = {'gain':1,'offset':0,'make_zero':False, 'range':10}
    if len(self.channels)*self.scan_rate > 100000:
      self.scan_rate = 100000/len(self.channels)
      print("[Labjack] Warning! scan_rate is too high! Sample rate cannot "
          "exceed 100kS/s, lowering samplerate to",self.scan_rate,"samples/s")
    self.chan_list = []
    for d in self.channels:
      if isinstance(d,str):
        d = {'name':d}
      for k in ['gain','offset','make_zero','range']:
        if k not in d:
          d[k] = default[k]
      if 'to_write' not in d:
        d['to_write'] = []
      d['to_write'].append(("_RANGE",d['range']))
      d['to_read'] = ljm.nameToAddress(d['name'])[0]
      self.chan_list.append(d)

  def open(self):
    self.handle = ljm.openS(self.device,self.connection,self.identifier)
    names,values = [],[]
    for c in self.chan_list:
      if "to_write" in c:
        for n,v in c['to_write']:
          names.append(c['name']+n)
          values.append(v)
    #names.append("STREAM_NUM_ADDRESSES");values.append(len(self.channels))
    names.append("STREAM_SCANRATE_HZ")
    values.append(self.scan_rate)
    names.append("STREAM_RESOLUTION_INDEX")
    values.append(self.resolution)
    ljm.eWriteNames(self.handle,len(names),names,values)
    scan_rate = ljm.eReadName(self.handle,"STREAM_SCANRATE_HZ")
    if scan_rate != self.scan_rate:
      print("[Labjack] Actual scan_rate:",scan_rate,
          "instead of",self.scan_rate)
      self.scan_rate = scan_rate
    if any([c.get("make_zero",False) for c in self.chan_list]):
      print("[Labjack] Please wait during offset evaluation...")
      off = self.eval_offset()
      names,values = [],[]
      for i,c in enumerate(self.chan_list):
        if 'make_zero' in c and c['make_zero']:
          c['offset'] += c['gain']*off[i]
    self.n = 0 # Number of data points (to rebuild time)

  def get_data(self):
    """
    Short version, only used for eval_offset
    """
    return [time()]+ljm.eReadNames(self.handle, len(self.chan_list),
                              [c['name'] for c in self.chan_list])

  def start_stream(self):
    ljm.eStreamStart(self.handle,self.scan_per_read,len(self.chan_list),
        [c['to_read'] for c in self.chan_list],self.scan_rate)
    self.stream_t0 = time()

  def stop_stream(self):
    ljm.eStreamStop(self.handle)

  def get_stream(self):
    a = np.array(ljm.eStreamRead(self.handle)[0])
    r = a.reshape(len(a)//len(self.channels),len(self.channels))
    for i,c in enumerate(self.chan_list):
      r[:,i] = c['gain']*r[:,i]+c['offset']
    t = self.stream_t0+np.arange(self.n,self.n+r.shape[0])/self.scan_rate
    self.n += r.shape[0]
    return [t,r]

  def close(self):
    ljm.close(self.handle)
