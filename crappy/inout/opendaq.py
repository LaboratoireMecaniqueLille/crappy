# coding: utf-8
from __future__ import print_function,absolute_import,division

from time import time,sleep

import opendaq

from .inout import InOut

def listify(stuff,l):
  r = stuff if isinstance(stuff,list) else [stuff]*l
  assert len(r) == l,"Invalid list length for "+str(r)
  return r

class Opendaq(InOut):
  """
  Can read data from an OpenDAQ card
  """
  def __init__(self, **kwargs):
    InOut.__init__(self)
    for arg,default in [('channels',1),
                        ('port','/dev/ttyUSB0'),
                        ('gain',1),
                        ('offset',0),
                        ('cmd_label','cmd'),
                        ('out_gain',1),
                        ('out_offset',0),
                        ('make_zero',True),
                        ('mode','single'),
                        ('nsamples',20)
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    assert len(kwargs) == 0,"Open_daq got unsupported kwarg(s)"+str(kwargs)
    self.channels = self.channels if isinstance(self.channels,list) else\
          [self.channels]
    n = len(self.channels)
    self.gain = listify(self.gain,n)
    self.offset = listify(self.offset,n)
    self.make_zero = listify(self.make_zero,n)


  def open(self):
    self.handle = opendaq.DAQ(self.port)
    if any(self.make_zero):
      off = self.eval_offset()
      for i,make_zero in enumerate(self.make_zero):
        if make_zero:
          self.offset[i] += off[i]
    if len(self.channels) == 1:
      self.handle.conf_adc(pinput=self.channels[0],ninput=0,gain=1)
    if self.mode == 'streamer':
      self.init_stream()

  def init_stream(self):
    self.stream_exp = self.handle.create_stream(mode=opendaq.ExpMode.ANALOG_IN,
                                                period=1,
                                                npoints=1,
                                                continuous=True,
                                                buffersize=1000)

    self.stream_exp.analog_setup(pinput=self.channels[0],
                                 ninput=0,
                                 gain=1,
                                 nsamples=self.nsamples)
    self.generator = self.stream_grabber()
    self.stream_started = False

  def start_stream(self):
    self.handle.start()
    self.stream_started = True

  def stream_grabber(self):
    filling = []
    while True:
      try:
        while len(filling) < self.sample_rate:
          filling.extend(self.stream_exp.read())
          sleep(0.001)
        yield filling[:self.sample_rate]
        del filling[:self.sample_rate]
        # print 'filling taille out:', len(filling)
      except:
        self.handle.close()
        break

  def get_data(self):
    t = [time()]
    if len(self.channels) == 1:
      return t+[self.handle.read_analog()]
    else:
      l = self.handle.read_all()
      return t+[l[i-1] for i in self.channels]

  def get_stream(self):
    if not self.stream_started:
      self.start_stream()
    return self.generator.next()

  def set_cmd(self,v):
    self.handle.set_analog(v*self.out_gain+self.out_offset)

  def close(self):
    if hasattr(self,'handle') and self.handle is not None:
      self.handle.stop()
      self.handle.close()
