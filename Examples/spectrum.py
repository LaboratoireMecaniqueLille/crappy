#coding: utf-8
from __future__ import print_function

import crappy
#import numpy as np

def my_mean(data):
  #print("D",data)
  for k,val in data.items():
    #data[k] = np.mean(val)
    data[k] = val[0]
  return data

spectrum = crappy.blocks.IOBlock('spectrum',ranges=[1000]*16,
    channels=list(range(16)),
    streamer=True,
    labels=['t(s)']+['ch'+str(i) for i in range(16)])

#graph = crappy.blocks.Grapher(('t(s)','ch0'),('t(s)','ch8'))
graph = crappy.blocks.Grapher(*[('t(s)','ch'+str(i)) for i in range(16)])

crappy.link(spectrum,graph,condition=my_mean)

crappy.start()
