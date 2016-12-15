from opendaq import DAQ
from time import time
from ._meta import io
from operator import itemgetter

class OpenDAQ(io.Control_Command):
  """
  Class for openDAQ Devices.
  """

  def __init__(self, *args, **kwargs):

    self.input_channels = kwargs.get('channels', 1)  # Possible values: 1..8
    self.nchannels = 1 if not isinstance(self.input_channels, list) else len(self.input_channels)
    self.input_gain = kwargs.get('gain', 0)  # Possible values: 0..4 (x1/3, x1, x2, x10, x100)
    self.input_offset = kwargs.get('offset', 0)  # not a parameter. apply after reading it.
    self.input_nsamples_per_read = kwargs.get('nsamples', 20)  # possible values : 0..254
    self.mode = kwargs.get('mode', 'single')
    self.i = 0
    self.new()
    if self.nchannels > 1:
      self.getter = itemgetter(*self.input_channels)
  def new(self):
    self.handle = DAQ("/dev/ttyUSB0")
    if self.nchannels == 1:
      self.handle.conf_adc(pinput=self.input_channels, ninput=0, gain=self.input_gain,
                           nsamples=self.input_nsamples_per_read)

  def get_data(self, mock=None):
    if self.nchannels == 1:
      data = [self.handle.read_analog()]
    else:
      data = list(self.getter(self.handle.read_all(self.input_nsamples_per_read)))
    self.i += 1
    return time(), data

  def set_cmd(self, command):
    self.handle.set_dac(command)

  def close(self):
    self.handle.close()
    pass
