#coding: utf-8
from __future__ import division,print_function

import crappy
from time import ctime
from numpy import pi,mean

"""
This example can run videoextenso with autodrive, fully controlled with a labjack device, and saving data in csv file.
"""

timestamp = ctime()[:-5].replace(" ","_")
save_path = "/home/extenso/Bureau/essais/"+timestamp+"/"

gains = [20,50] # N/V, mm/V
gain_cmd = 1/20 # V/mm

def eval_offsets(sensor,count=200):
  print("Evaluating offset...")
  l = len(sensor.get_data()[1])
  offsets = []
  for i in range(l):
    offsets.append(list())
  for i in range(count):
    data = sensor.get_data()[1]
    for j in range(l):
      offsets[j].append(data[j])
  return map(lambda l: -mean(l),offsets)


lj = crappy.sensor.LabJackSensor(channels=[0,1],gain=gains)
offsets = eval_offsets(lj)

lj = crappy.technical.LabJack(sensor={'channels':[0,1],'gain':gains,'offset':offsets},actuator={'gain':gain_cmd})
lj.set_cmd(0) # To make sure we start from 0


videoextenso = crappy.blocks.VideoExtenso(camera='Ximea')
autodrive = crappy.blocks.AutoDrive(technical='CmDrive',dev_args={'port': '/dev/ttyUSB0'},direction='Y-')
crappy.link(videoextenso,autodrive)

graph_ve = crappy.blocks.Grapher(('t(s)','Exx(%)'),('t(s)','Eyy(%)'),length=200)
crappy.link(videoextenso,graph_ve)

saver_extenso = crappy.blocks.Saver(save_path+"extenso.csv")
crappy.link(videoextenso,saver_extenso)

controlcommand = crappy.blocks.ControlCommand(lj,labels=['t(s)','F(N)','x(mm)'],freq=50)

saver_sensors = crappy.blocks.Saver(save_path+"sensors.csv")
crappy.link(controlcommand,saver_sensors)

signalgenerator = crappy.blocks.SignalGenerator(send_freq=50,repeat=False,labels=['t(s)','signal'],path=[
  #{'waveform': 'sinus', 'time':60, 'phase': -pi/2. , 'amplitude':2, 'offset':2,'freq':.1},
  {'waveform':'hold','time':1},
  {"waveform": "ramp", "gain": 2, "cycles": 1, "phase": 0, "lower_limit": [1, 'F(N)'],
       "upper_limit": [20, 'Eyy(%)'], 'origin':0},#gain: mm/s
  #{"waveform": "ramp", "gain": 2, "cycles": 2, "phase": 0, "lower_limit": [1, 'F(N)'],
  #     "upper_limit": [20, 'F(N)']},
  {"waveform": "ramp", "gain": 2, "cycles": 1, "phase": 0, "lower_limit": [0.2, 'x(mm)'],
       "upper_limit": [20, 'x(mm)']}
  ])
crappy.link(videoextenso,signalgenerator)
crappy.link(controlcommand,signalgenerator)
crappy.link(signalgenerator,controlcommand)

graph = crappy.blocks.Grapher(('t(s)','F(N)'),('t(s)','x(mm)'),('t(s)','signal'),length=200)
crappy.link(controlcommand,graph)


crappy.start()