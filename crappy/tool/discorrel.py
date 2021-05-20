# coding: utf-8

from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")
import numpy as np

from .fields import get_fields,Projector,get_res


class DISCorrel():
  def __init__(self,img0,**kwargs):
    self.img0 = img0
    self.h,self.w = img0.shape
    for arg,default in [("bbox",None),
                        # fields: Base of fields to use for the projection
                        ("fields",["x","y","exx","eyy"]),
                        # alpha, delta, gamma: settings for disflow
                        ("alpha",3),
                        ("delta",1),
                        ("gamma",0),
                        # finest_scale: last scale for disflow (0=fullscale)
                        ("finest_scale",1),
                        # init: Should we use the last field to init ?
                        ("init",True),
                        # Variational refinement iterations
                        ("iterations",1),
                        # Gradient descent iterations
                        ("gditerations",10),
                        # DIS patch size
                        ("patch_size",8),
                        # DIS patch stride
                        ("patch_stride",3),

                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg in ve:"+str(kwargs)
    if self.bbox is None:
      self.bbox = (0,0,self.h,self.w)
    self.bh,self.bw = self.bbox[2]-self.bbox[0],self.bbox[3]-self.bbox[1]
    self.fields = get_fields(self.fields,self.bh,self.bw)
    self.p = Projector(self.fields)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setGradientDescentIterations(self.gditerations)
    self.dis.setPatchSize(self.patch_size)
    self.dis.setPatchStride(self.patch_stride)
    self.dis_flow = np.zeros((self.h,self.w,2))

  def crop(self,img):
    ymin,xmin,ymax,xmax = self.bbox
    return img[ymin:ymax,xmin:xmax]

  def calc(self,img):
    self.img = img
    if self.init:
      self.dis_flow = self.dis.calc(self.img0,img,self.dis_flow)
    else:
      self.dis_flow = self.dis.calc(self.img0,img,None)
    self.proj = self.p.get_scal(self.crop(self.dis_flow))
    return self.proj

  def dis_res(self):
    return get_res(self.img0,self.img,self.dis_flow)

  def dis_res_scal(self):
    return np.average(np.abs(get_res(self.img0,self.img,self.dis_flow)))

  def proj_flow(self):
    return self.p.get_full(self.dis_flow)
