#coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup FakeCamera FakeCamera
# @{

## @file _fakeCamera.py
# @brief Streams a fake picture.
# @author Victor Couty
# @version 0.1
# @date 05/10/2016

from _meta import MasterBlock
import numpy as np
import time

class FakeCamera(MasterBlock):
  """
  Streams a static picture to fake a camera
  """
  def __init__(self,width=1024,height=1024,timer=True,animated=True):
    MasterBlock.__init__(self)
    self.w = width
    self.h = height
    self.timer = timer
    # Create a simple gradient image
    self.img = np.arange(width)*255./width
    self.img = np.repeat(self.img.reshape(width,1),height,axis=1).astype(np.uint8)
    self.loops = 0
    self.animated = animated
    
  def main(self):
    #If timer is true, will print debug info every "skip" frame
    skip = 100
    if self.timer:
      t0 = time.time()
    while True:
      self.loops += 1
      if self.loops%skip == 0 and self.timer:
        t1 = t0
        t0 = time.time()
        print "[FakeCamera] running at",skip/(t0-t1),"fps (avg:",self.loops/(t0-self.t0),")"
      i = self.loops%self.h
      for o in self.outputs:
        if self.animated:
          o.send(np.concatenate((self.img[i:,:],self.img[:i,:]),axis=0))
        else:
          o.send(self.img)
