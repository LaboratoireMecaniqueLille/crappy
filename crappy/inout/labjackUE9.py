# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjack LabJack
# @{

## @file labJack.py
# @brief  General class for LabJack devices.
# @author Francois Bari
# @version 0.9
# @date 18/08/2016
from __future__ import print_function,absolute_import,division
from time import time
from ue9 import UE9

from .inout import InOut

def get_channel_number(channels):
  """
  register needs to be called with the channel name as int.
  """
  for i,channel in enumerate(channels):
    if isinstance(str,channel):
      channels[i] = int(channel[-1])

def format_lists(list_to_format, length):
  """
  In case the user only specifies one parameter, and wants
  it applied to all inputs.
  """
  if not isinstance(list_to_format, list):
    list_to_format = [list_to_format]
  if length is not 0:
    if len(list_to_format) == 1:
      return list_to_format * length
    elif len(list_to_format) == length:
      return list_to_format
    else:
      raise TypeError('Wrong Labjack Parameter definition.')
  else:
    return list_to_format

class Labjack_UE9(InOut):
  """Can read data from a LabJack UE9
  streamer mode and DAC are not supported yet
  """
  def __init__(self, **kwargs):
    InOut.__init__(self)
    for arg,default in [('channels',0),
                        ('gain',1),
                        ('offset',0),
                        ('resolution',12),
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    assert len(kwargs) == 0,"Labjack_UE9 got unsupported arg(s)"+str(kwargs)
    self.channels = format_lists(self.channels, 0)
    self.nb_channels = len(self.channels)
    get_channel_number(self.channels)
    self.gain = format_lists(self.gain, self.nb_channels)
    self.offset = format_lists(self.offset, self.nb_channels)
    self.resolution = format_lists(self.resolution, self.nb_channels)

  def open(self):
    self.handle = UE9()

  def get_data(self):
    results = []
    t0 = time()
    for index, channel in enumerate(self.channels):
      results.append(
        self.handle.getAIN(channel, Resolution=self.resolution[index]) *
        self.gain[index] + self.offset[index])
    t1 = time()
    return (t0+t1)/2, results

  def close(self):
    if hasattr(self,'handle') and self.handle is not None:
      self.handle.close()
