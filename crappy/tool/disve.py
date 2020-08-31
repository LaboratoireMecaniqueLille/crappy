#coding: utf-8
import cv2
import numpy as np


class DISVE():
  def __init__(self,img0,patches,**kwargs):
    self.img0 = img0
    self.patches = patches
    self.h,self.w = img0.shape
    for arg,default in [("alpha",3),
                        ("delta",1),
                        ("gamma",0),
                        # alpha, delta, gamma: settings for disflow
                        ("finest_scale",0),
                        # finest_scale: last scale for disflow (0=fullscale)
                        ("init",True),
                        # init: Should we use the last field to init ?
                        ("iterations",10)]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"Invalid kwarg in ve:"+str(kwargs)
    self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    self.dis.setVariationalRefinementIterations(self.iterations)
    self.dis.setVariationalRefinementAlpha(self.alpha)
    self.dis.setVariationalRefinementDelta(self.delta)
    self.dis.setVariationalRefinementGamma(self.gamma)
    self.dis.setFinestScale(self.finest_scale)
    self.last = [None for p in self.patches]

  def get_patch(self,img,patch):
    ymin,xmin,h,w = patch
    return np.array(img[ymin:ymin+h,xmin:xmin+w])

  def calc(self,img):
    r = []
    for p in self.patches:
      r.append(self.dis.calc(
        self.get_patch(self.img0,p),
        self.get_patch(img,p),None))
    self.last = r
    return sum([np.average(i,axis=(0,1)).tolist() for i in r],[])

if __name__ == '__main__':
  #TEST
  import matplotlib.pyplot as plt

  def tobw(img):
    return np.average(img,axis=2).astype(np.uint8)
  cam = cv2.VideoCapture(0)
  #cam.open()
  r,f = cam.read()
  assert r, "Error opening the camera"
  for i in range(5):
    r,f = cam.read() # For auto brigthness to adjust
  plt.imshow(f[:,:,::-1])
  plt.show()
  ve = DISVE(tobw(f),[(100,100,100,100),(200,200,100,100)])
  while True:
    r,f = cam.read()
    print(ve.calc(tobw(f)))
