#coding: utf-8
from __future__ import print_function, absolute_import, division

from time import time
import numpy as np
import nidaqmx
from nidaqmx import stream_readers,stream_writers

from .inout import InOut

thermocouple_type = {
"B":nidaqmx.constants.ThermocoupleType.B,
"E":nidaqmx.constants.ThermocoupleType.E,
"J":nidaqmx.constants.ThermocoupleType.J,
"K":nidaqmx.constants.ThermocoupleType.K,
"N":nidaqmx.constants.ThermocoupleType.N,
"R":nidaqmx.constants.ThermocoupleType.R,
"S":nidaqmx.constants.ThermocoupleType.S,
"T":nidaqmx.constants.ThermocoupleType.T,
}

units = {
"C":nidaqmx.constants.TemperatureUnits.DEG_C,
"F":nidaqmx.constants.TemperatureUnits.DEG_F,
"R":nidaqmx.constants.TemperatureUnits.DEG_R,
"K":nidaqmx.constants.TemperatureUnits.K,
# To complete...
}

class Nidaqmx(InOut):
  """
  TODO: doc!
  """
  def __init__(self,device="Dev1", **kwargs):
    InOut.__init__(self)
    self.device = device
    for arg, default in [
                         ('channels', [{'name':'ai0'}]),
                         ('samplerate',100),
                         ('nsamples',None)
                         ]:
      setattr(self, arg, kwargs.pop(arg,default))
    assert len(kwargs) == 0, "Nidaqmx got unsupported arg(s)" + str(kwargs)
    if self.nsamples is None: self.nsamples = max(1,int(self.samplerate/5))
    self.streaming = False
    self.ao_channels = []
    self.ai_channels = {}
    for c in self.channels:
      c['type'] = c.get('type','voltage').lower()
      if c['name'].startswith('ai'):
        if c['type'] in self.ai_channels:
          self.ai_channels[c['type']].append(c)
        else:
          self.ai_channels[c['type']] = [c]
      elif c['name'].startswith('ao'):
        self.ao_channels.append(c)
      else:
        raise AttributeError("Unknown channel in nidaqmx"+str(c))
    self.ai_chan_list = sum(self.ai_channels.values(),[])
    

  def open(self):
    self.t_in = {}
    if self.ao_channels:
      self.t_out = nidaqmx.Task()
    
    for c in self.ai_chan_list:
      kwargs = dict(c)
      kwargs.pop("name",None)
      kwargs.pop("type",None)
      for k in kwargs:
        if isinstance(kwargs[k],str):
          if k == "thermocouple_type":
            kwargs[k] = thermocouple_type[kwargs[k]]
          elif k == "units":
            kwargs[k] = units[kwargs[k]]
      if not c['type'] in self.t_in:
        self.t_in[c['type']] = nidaqmx.Task()
      try:
        getattr(self.t_in[c['type']].ai_channels,"add_ai_%s_chan"%c['type'])(
        "/".join([self.device,c['name']]),
        **kwargs)
      except Exception as e:
        print("Invalid channel settings in nidaqmx:"+str(c))
        raise

    for c in self.ao_channels:
      kwargs = dict(c)
      kwargs.pop("name",None)
      kwargs.pop("type",None)
      self.t_out.ao_channels.add_ao_voltage_chan(
          "/".join([self.device,c['name']]),min_val=0,max_val=5, **kwargs)

    self.stream_in = {}
    for chan_type,t in self.t_in.items():
      self.stream_in[chan_type] = \
          stream_readers.AnalogMultiChannelReader(t.in_stream)
    if self.ao_channels:
      self.stream_out = stream_writers.AnalogMultiChannelWriter(
          self.t_out.out_stream,auto_start=True)
    print("DEBUG",self.ai_channels)

  def start_stream(self):
    for t in self.t_in.values():
      t.timing.cfg_samp_clk_timing(self.samplerate,
          sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    self.streaming = True

  def stop_stream(self):
    self.streaming = False
    for t in self.t_in.values():
      t.stop()
      t.timing.cfg_samp_clk_timing(self.samplerate,
        sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

  def get_stream(self):
    if not self.streaming:
      self.start_stream()
    a = np.empty((len(self.ai_chan_list),self.nsamples))
    i = 0
    for chan_type,s in self.stream_in.items():
      s.read_many_sample(a[i:i+len(self.ai_channels[chan_type]),:],self.nsamples)
      i += len(self.ai_channels[chan_type])
    return [time(),a]

  def close(self):
    for t in self.t_in.values():
      t.stop()
    if self.ao_channels:
      self.t_out.stop()
    if self.streaming:
      self.stop_stream()
    for t in self.t_in.values():
      t.close()
    if self.ao_channels:
      self.t_out.close()
    
  def get_data(self):
    a = np.empty((len(self.ai_channels)))
    i = 0
    t = time()
    for chan_type,s in self.stream_in.items():
      s.read_one_sample(a[i:i+len(self.ai_channels[chan_type])])
      i += len(self.ai_channels[chan_type])
    return [t]+list(a)

  def set_cmd(self,v):
    self.stream_out.write_one_sample(np.array([v],dtype=np.float64))