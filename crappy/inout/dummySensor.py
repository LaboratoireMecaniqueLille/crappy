# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup dummysensor DummySensor
# @{

## @file _dummySensor.py
# @brief  Mock a sensor and return the time. Use it for testing.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016
import random
import time

import math

from crappy.sensor._meta import motion
from inout import InOut


class DummySensor(InOut):
  """Mock a sensor and return the time. Use it for testing."""

  def __init__(self, *args, **kwargs):
    super(DummySensor, self).__init__()
    self.position = random.random() * random.random() * 10
    self.channels = kwargs.get("channels", [0])
    if not isinstance(self.channels, list):
      self.channels = [self.channels]
    self.offset = kwargs.get("offset", [0] * len(self.channels))
    if not isinstance(self.offset, list):
      self.offset = [self.offset] * len(self.channels)
    self.gain = kwargs.get("gain", [1] * len(self.channels))
    if not isinstance(self.gain, list):
      self.gain = [self.gain] * len(self.channels)
    self.ret_type = kwargs.get("ret_type", "random")
    self.infinite_iterator = self.frange_infinite(0, 1)

    if self.ret_type == "sin":
      self.iterator = self.frange(-math.pi / 2, math.pi / 2, 0.1)
    elif self.ret_type == "cos":
      self.iterator = self.frange(-math.pi, math.pi, 0.1)
    else:
      self.iterator = self.frange(-10, 10, 0.1)
      print('gain and offset:', self.gain, self.offset)
  @staticmethod
  def frange(x, y, jump):
    a = x
    b = y
    while 1:
      if x < y:
        yield round(x, 1)
        x += jump
      else:
        if round(y, 1) == round(float(a), 1):
          # yield y
          x = a
          y = b
        else:
          yield round(y, 1)
          y -= jump

  @staticmethod
  def frange_infinite(start, jump):
    while 1:
      yield start
      start += jump

  def get_data(self, channel_number=0):
    """
    Read the signal for desired channel
    """
    if self.ret_type == "sin":
      ret_value = math.sin(self.iterator.next())
    elif self.ret_type == "cos":
      ret_value = math.cos(self.iterator.next())
    elif self.ret_type == "def":
      ret_value = abs(self.infinite_iterator.next())
    elif self.ret_type == "triangle":
      ret_value = self.iterator.next()
    else:
      ret_value = round(random.random() * random.random() * 10, 3)
    if channel_number == "all":
      result = []
      for i in xrange(len(self.channels)):
        result.append(ret_value * self.gain[i] + self.offset[i])
      t = time.time()
      return t, result
    else:
      t = time.time()
      return t, ret_value

  def get_position(self):
    return self.position

  def new(self, *args):
    """Do nothing."""
    pass

  def close(self, *args):
    """Do nothing."""
    pass
