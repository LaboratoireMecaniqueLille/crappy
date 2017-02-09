#coding: utf-8
from __future__ import print_function,division

from time import time,sleep
from collections import OrderedDict

from ._compacterblock import CompacterBlock

class DataReader(CompacterBlock):
  def __init__(self,sensor='DaqmxSensor', freq=100, labels=['t','V'], **kwargs):
    CompacterBlock.__init__(self, compacter=100, labels=labels)
    self.sensor_name = sensor
    self.sensor_kwargs = kwargs
    self.period = 1/freq
    
  def prepare(self):
    s = __import__('crappy.sensor',fromlist=[self.sensor_name])
    self.sensor_class = getattr(s,self.sensor_name)
    self.sensor = self.sensor_class(**self.sensor_kwargs)
    self.sensor.new()
    self.t = time()
    
  def main(self):
    while True:
      left = 1
      while left > 0:
        #sleep(left/2)
        left = self.t-time()+self.period

      self.t = time()
      d = self.sensor.get_data()
      """
      data = OrderedDict()
      data['t'] = d[0]
      data['V'] = d[1]
      print('sent',data)"""
      self.send((d[0]-self.t0,d[1]))