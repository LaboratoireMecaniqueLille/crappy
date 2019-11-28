#coding: utf-8
import cv2
import numpy as np

from .fields import get_fields,Projector,get_res


class DISCorrel():
  def __init__(self,img0,**kwargs):
    self.img0 = img0
    self.h,self.w = img0.shape
    for arg,default in [("bbox",None),
                        ("fields",["x","y","exx","eyy"]),
                        # fields: Base of fields to use for the projection
                        ("alpha",3),
                        ("delta",1),
                        ("gamma",0),
                        # alpha, delta, gamma: settings for disflow
                        ("finest_scale",1),
                        # finest_scale: last scale for disflow (0=fullscale)
                        ("init",True),
                        # init: Should we use the last field to init ?
                        ("iterations",10)]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg in ve:"+str(kwargs)
    if self.bbox is None:
      self.bbox = (0,0,self.h,self.w)
    self.bh,self.bw = self.bbox[2]-self.bbox[0],self.bbox[3]-self.bbox[1]
    self.fields = get_fields(self.fields,self.bh,self.bw)
    self.p = Projector(self.fields)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.dis_flow = np.zeros((self.h,self.w,2))

  def crop(self,img):
    ymin,xmin,ymax,xmax = self.bbox
    return img[ymin:ymax,xmin:xmax]

  def calc(self,img):
    self.img = img
    self.f = self.dis.calc(self.img0,img,self.dis_flow)
    self.proj = self.p.get_scal(self.crop(self.f))
    return self.proj

  def dis_res(self):
    return get_res(self.img0,self.img,self.dis_flow)

  def dis_res_scal(self):
    return np.average(np.abs(get_res(self.img0,self.img,self.dis_flow)))

  def proj_flow(self):
    return self.p.get_full(self.dis_flow)
