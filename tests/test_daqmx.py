#coding: utf-8

import crappy

s = crappy.sensor.DaqmxSensor(device='Dev2')
s.new()
while True:
  print(s.get_data())