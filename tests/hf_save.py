#coding: utf-8

import numpy as np
from tables import Int8Atom

import crappy

class Randomhf(crappy.blocks.MasterBlock):
  def __init__(self):
    crappy.blocks.MasterBlock.__init__(self)
    self.labels = ['stream']

  def loop(self):
    self.send([np.random.randint(-128,127,size=(500,16),dtype=np.int8) for i in range(10)])


a = Randomhf()
b = crappy.blocks.Hdf_saver("/home/vic/crappy/out.h5",
    metadata={'channels':list(range(16)),'ranges':[50]*16,'lala':1},
    node='array',
    atom=Int8Atom())
crappy.link(a,b)
crappy.start()
