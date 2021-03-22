import numpy as np
import cv2

from .modifier import Modifier


class Apply_strain_img(Modifier):
  """
  This modifier reads the strain values along X and Y (in %) and creates an
  image deformed to match these values

  img: The image to use (must be a numpy array)

  exx(eyy)_label: the labels containing the strain to apply

  img_label: the label of the generated image
  """
  def __init__(self,img,exx_label='Exx(%)',eyy_label='Eyy(%)',
      img_label='frame'):
    self.img = img
    self.lexx = exx_label
    self.leyy = eyy_label
    self.img_label = img_label
    h,w = img.shape
    self.exx = np.concatenate((np.linspace(-w/2, w/2, w,
          dtype=np.float32)[np.newaxis, :],)*h, axis=0)
    self.eyy = np.concatenate((np.linspace(-h/2, h/2, h,
      dtype=np.float32)[:, np.newaxis],)*w, axis=1)
    xx,yy = np.meshgrid(range(w),range(h))
    self.xx = xx.astype(np.float32)
    self.yy = yy.astype(np.float32)

  def evaluate(self,d):
    exx,eyy = d[self.lexx]/100,d[self.leyy]/100
    tx,ty = (self.xx-(exx/(1+exx))*self.exx),(self.yy-(eyy/(1+eyy))*self.eyy)
    d[self.img_label] = cv2.remap(self.img,tx,ty,1)
    return d
