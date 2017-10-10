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
    self.di_channels = []
    self.do_channels = []
    for c in self.channels:
      c['type'] = c.get('type','voltage').lower()
      if c['name'].startswith('ai'):
        if c['type'] in self.ai_channels:
          self.ai_channels[c['type']].append(c)
        else:
          self.ai_channels[c['type']] = [c]
      elif c['name'].startswith('ao'):
        self.ao_channels.append(c)
      elif c['name'].startswith('di'):
        c['name'] = 'line'+c['name'][2:]
        self.di_channels.append(c)
      elif c['name'].startswith('do'):
        c['name'] = 'line'+c['name'][2:]
        self.do_channels.append(c)
      else:
        raise AttributeError("Unknown channel in nidaqmx"+str(c))
    self.ai_chan_list = sum(self.ai_channels.values(),[])


  def open(self):
    # AI
    self.t_in = {}

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
    self.stream_in = {}
    for chan_type,t in self.t_in.items():
      self.stream_in[chan_type] = \
          stream_readers.AnalogMultiChannelReader(t.in_stream)
    # AO
    if self.ao_channels:
      self.t_out = nidaqmx.Task()
      self.stream_out = stream_writers.AnalogMultiChannelWriter(
          self.t_out.out_stream,auto_start=True)

    for c in self.ao_channels:
      kwargs = dict(c)
      kwargs.pop("name",None)
      kwargs.pop("type",None)
      self.t_out.ao_channels.add_ao_voltage_chan(
          "/".join([self.device,c['name']]),min_val=0,max_val=5, **kwargs)
    # DI
    if self.di_channels:
      self.t_di = nidaqmx.Task()
    for c in self.di_channels:
      kwargs = dict(c)
      kwargs.pop("name",None)
      kwargs.pop("type",None)
      self.t_di.di_channels.add_di_chan(
          "/".join([self.device,c['name']]), **kwargs)
    if self.di_channels:
      self.di_stream = stream_readers.DigitalMultiChannelReader(
          self.t_di.in_stream)

    # DO
    if self.do_channels:
      self.t_do = nidaqmx.Task()
    for c in self.do_channels:
      kwargs = dict(c)
      kwargs.pop("name",None)
      kwargs.pop("type",None)
      self.t_do.do_channels.add_do_chan(
          "/".join([self.device,c['name']]), **kwargs)
    if self.do_channels:
      self.do_stream = stream_writers.DigitalMultiChannelWriter(
          self.t_do.out_stream)

  def start_stream(self):
    if len(self.t_in) != 1:
      raise IOError("Stream mode can only open one type of chanchan type in stream mode!")
    for t in self.t_in.values(): # Only one loop
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
    for chan_type,s in self.stream_in.items(): #Will only loop once
      s.read_many_sample(a,self.nsamples)
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
    a = np.empty((len(self.ai_channels),))
    i = 0
    t = time()
    for chan_type,s in self.stream_in.items():
      s.read_one_sample(a[i:i+len(self.ai_channels[chan_type])])
      i += len(self.ai_channels[chan_type])
    if self.di_channels:
      b = np.empty((len(self.di_channels),1),dtype=np.bool)
      self.di_stream.read_one_sample_multi_line(b)
      return [t]+list(a)+list(b[:,0])
    return [t]+list(a)

  def set_cmd(self,*v):
    if self.ao_channels:
      self.stream_out.write_one_sample(np.array(v[:len(self.ao_channels)],
        dtype=np.float64))
    if self.do_channels:
      self.do_stream.write_one_sample_multi_line(
        np.array(v[len(self.ao_channels):],dtype=np.bool).reshape(len(self.do_channels),1))