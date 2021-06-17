# coding: utf-8

from time import time

from .inout import InOut
from .._global import OptionalModule

try:
  from ue9 import UE9
except (ModuleNotFoundError, ImportError):
  UE9 = OptionalModule("ue9")


def get_channel_number(channels):
  """Register needs to be called with the channel name as :obj:`int`."""

  for i, channel in enumerate(channels):
    if isinstance(str, channel):
      channels[i] = int(channel[-1])


def format_lists(list_to_format, length):
  """In case the user only specifies one parameter, and wants it applied to all
  inputs."""

  if not isinstance(list_to_format, list):
    list_to_format = [list_to_format]
  if length != 0:
    if len(list_to_format) == 1:
      return list_to_format * length
    elif len(list_to_format) == length:
      return list_to_format
    else:
      raise TypeError('Wrong Labjack Parameter definition.')
  else:
    return list_to_format


class Labjack_ue9(InOut):
  """Can read data from a LabJack UE9.

  Important:
    Streamer mode and DAC are not supported yet.
  """

  def __init__(self,
               channels=0,
               gain=1,
               offset=0,
               make_zero=True,
               resolution=12):
    InOut.__init__(self)
    self.channels = channels
    self.gain = gain
    self.offset = offset
    self.make_zero = make_zero
    self.resolution = resolution

    self.channels = format_lists(self.channels, 0)
    self.nb_channels = len(self.channels)
    get_channel_number(self.channels)
    self.gain = format_lists(self.gain, self.nb_channels)
    self.offset = format_lists(self.offset, self.nb_channels)
    self.resolution = format_lists(self.resolution, self.nb_channels)
    self.make_zero = format_lists(self.make_zero, self.nb_channels)

  def open(self):
    self.handle = UE9()
    if any(self.make_zero):
      off = self.eval_offset()
      for i, make_zero in enumerate(self.make_zero):
        if make_zero:
          self.offset[i] += off[i]

  def get_data(self):
    results = []
    t0 = time()
    for index, channel in enumerate(self.channels):
      results.append(
        self.handle.getAIN(channel, Resolution=self.resolution[index]) *
        self.gain[index] + self.offset[index])
    t1 = time()
    return (t0 + t1) / 2, results

  def close(self):
    if hasattr(self, 'handle') and self.handle is not None:
      self.handle.close()
