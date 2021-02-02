#coding: utf-8

import crappy
import tables
import numpy as np

def my_mean(data):
  #print("D",data)
  #print("t",type(data))
  for k,val in data.items():
    data[k] = np.mean(val)
    #data[k] = val[0]
  return data

s = crappy.blocks.IOBlock("T7_streamer",
    #labels=['t','AIN0','AIN1'],
    #labels=['t','stream'],
    channels = [{'name':'AIN0','gain':2,'offset':-13},
      {'name':'AIN1','gain':2,"make_zero":True}],
      #channels=['AIN0','AIN1'],
    streamer=True)

#g = crappy.blocks.Grapher(('t','AIN0'),('t','AIN1'))
#crappy.link(s,g,modifier=my_mean)

save = crappy.blocks.Hdf_saver("/home/vic/out.h5",atom=tables.Float64Atom())
crappy.link(s,save)
crappy.start()
