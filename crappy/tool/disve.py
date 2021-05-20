# coding: utf-8

import numpy as np
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DISVE():
  def __init__(self,img0,patches,**kwargs):
    self.img0 = img0
    self.patches = patches
    self.h,self.w = img0.shape
    for arg,default in [("alpha",3),
                        ("delta",1),
                        ("gamma",0),
                        # alpha, delta, gamma: settings for disflow
                        ("finest_scale",1),
                        # finest_scale: last scale for disflow (0=fullscale)
                        ("iterations",1),
                        # Gradient descent iterations
                        ("gditerations",10),
                        # DIS patch size
                        ("patch_size",8),
                        # DIS patch stride
                        ("patch_stride",3),
                        # Remove borders 10% of the size of the patch (0 to .5)
                        ("border",.1)
                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg in ve:"+str(kwargs)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis.setGradientDescentIterations(self.gditerations)
    self.dis.setPatchSize(self.patch_size)
    self.dis.setPatchStride(self.patch_stride)
    self.last = [None for p in self.patches]

  def get_patch(self,img,patch):
    ymin,xmin,h,w = patch
    return np.array(img[ymin:ymin+h,xmin:xmin+w])

  def get_center(self,f):
    h,w,*_ = f.shape
    return f[int(h*self.border):int(h*(1-self.border)),
        int(w*self.border):int(w*(1-self.border))]

  def calc(self,img):
    r = []
    for p in self.patches:
      f = self.dis.calc(
        self.get_patch(self.img0,p),
        self.get_patch(img,p),None)
      r.append(np.average(self.get_center(f),axis=(0,1)).tolist())
    self.last = r
    return sum(r,[])
