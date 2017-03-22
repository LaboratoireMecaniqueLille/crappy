#coding: utf-8
from __future__ import print_function,division

from time import time,sleep

from .masterblock import MasterBlock

class DataReader(MasterBlock):
  def __init__(self,sensor='DaqmxSensor', freq=100, labels=['t(s)','V(V)'],
                                                                  **kwargs):
    MasterBlock.__init__(self)
    self.labels=labels
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
        left = self.t-time()+self.period
        sleep(max(0,left/2))

      self.t = time()
      d = self.sensor.get_data()
      self.send((d[0]-self.t0,d[1]))
