# coding: utf-8

from time import time, sleep

from .inout import InOut
from .._global import OptionalModule
try:
  import opendaq
except (ModuleNotFoundError, ImportError):
  opendaq = OptionalModule("opendaq",
      "Please install the OpenDAQ Python Module")


def listify(stuff, length):
  r = stuff if isinstance(stuff, list) else [stuff] * length
  assert len(r) == length, "Invalid list length for " + str(r)
  return r


class Opendaq(InOut):
  """Can read data from an OpenDAQ card."""

  def __init__(self,
               channels=1,
               port='/dev/ttyUSB0',
               gain=1,
               offset=0,
               cmd_label='cmd',
               out_gain=1,
               out_offset=0,
               make_zero=True,
               mode='single',
               nsamples=20):

    InOut.__init__(self)
    self.channels = channels
    self.port = port
    self.gain = gain
    self.offset = offset
    self.cmd_label = cmd_label
    self.out_gain = out_gain
    self.out_offset = out_offset
    self.make_zero = make_zero
    self.mode = mode
    self.nsamples = nsamples

    self.channels = self.channels if isinstance(self.channels, list) else \
      [self.channels]
    n = len(self.channels)
    self.gain = listify(self.gain, n)
    self.offset = listify(self.offset, n)
    self.make_zero = listify(self.make_zero, n)

  def open(self):
    self.handle = opendaq.DAQ(self.port)
    if any(self.make_zero):
      off = self.eval_offset()
      for i, make_zero in enumerate(self.make_zero):
        if make_zero:
          self.offset[i] += off[i]
    if len(self.channels) == 1:
      self.handle.conf_adc(pinput=self.channels[0], ninput=0, gain=1)
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
      except Exception:
        self.handle.close()
        break

  def get_data(self):
    t = [time()]
    if len(self.channels) == 1:
      return t + [self.handle.read_analog()]
    else:
      lst = self.handle.read_all()
      return t + [lst[i - 1] for i in self.channels]

  def get_stream(self):
    if not self.stream_started:
      self.start_stream()
    return next(self.generator)

  def set_cmd(self, v):
    self.handle.set_analog(v * self.out_gain + self.out_offset)

  def close(self):
    if hasattr(self, 'handle') and self.handle is not None:
      self.handle.stop()
      self.handle.close()
