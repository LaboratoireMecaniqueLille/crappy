#coding: utf-8
import crappy

lj = crappy.technical.LabJack(sensor={'channels':[0,1,2,3]},actuator={1:1})

cc = crappy.blocks.ControlCommand(lj,labels=['t','A','B','C','D'],compacter=10,
                                     freq=50)

sg = crappy.blocks.SignalGenerator(send_freq=50,repeat=True,
  labels=['t','signal'],path=[
  {'waveform': 'sinus', 'time':10, 'phase': 0,
     'amplitude':2, 'offset':2,'freq':1},
  ])
crappy.link(sg,cc,name='SG')

g = crappy.blocks.Grapher(('t','A'),('t','B'),length=20)

crappy.link(cc,g,name='G1')


crappy.start()
