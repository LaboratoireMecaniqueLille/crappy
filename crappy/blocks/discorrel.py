# coding: utf-8

import cv2
import sys
import os
import numpy as np

from ..tool import DISCorrel as DIS
from ..tool import DISConfig
from .masterblock import MasterBlock
from ..camera import Camera


def draw_box(box,img):
  for s in [
      (box[0],slice(box[1],box[3])),
      (box[2],slice(box[1],box[3])),
      (slice(box[0],box[2]),box[1]),
      (slice(box[0],box[2]),box[3])
   ]:
    # Turn these pixels white or black for highest possible contrast
    img[s] = 255*int(np.mean(img[s])<128)


class DISCorrel(MasterBlock):
  def __init__(self,**kwargs):
    MasterBlock.__init__(self)
    self.niceness = -5
    default_labels = ['t(s)','x(pix)','y(pix)','Exx(%)','Eyy(%)']
    for arg,default in [("camera","XimeaCV"),
                        ("save_folder",None),
                        ("save_period",1),
                        ("labels",default_labels),
                        ("show_fps",True),
                        ("show_image",False),
                        ("residual",False),
                        ("residual_full",False),
                        ]:
      try:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      except KeyError:
        setattr(self,arg,default)
    self.cam_kwargs = kwargs
    if self.residual:
      self.labels.append('res')
    if self.residual_full:
      self.labels.append('res_full')

  def prepare(self):
    if self.save_folder and not os.path.exists(self.save_folder):
      try:
        os.makedirs(self.save_folder)
      except OSError:
        assert os.path.exists(self.save_folder),\
            "Error creating "+self.save_folder
    self.cam = Camera.classes[self.camera]()
    self.cam.open(**self.cam_kwargs)
    config = DISConfig(self.cam)
    config.main()
    self.bbox = config.box
    t,img0 = self.cam.get_image()
    self.correl = DIS(img0,bbox=self.bbox)
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("DISCorrel",flags)
    self.loops = 0
    self.last_fps_print = 0
    self.last_fps_loops = 0

  def begin(self):
    t,self.img0 = self.cam.get_image()
    self.correl.img0 = self.img0

  def loop(self):
    self.loops += 1
    if self.inputs and self.inputs[0].poll():
      self.inputs[0].clear()
      self.correl.img0 = img
      self.img0 = img
      print("[CORREL block] : Resetting L0")

    t,img = self.cam.read_image()
    d = self.correl.calc(img)
    if self.save_folder and not self.loops%self.save_period:
      self.save_img(t,img)
    if self.show_image:
      draw_box(self.bbox,img)
      cv2.imshow("DISCorrel",img)
      cv2.waitKey(5)
    if self.show_fps:
      if t - self.last_fps_print > 2:
        sys.stdout.write("\rFPS: %.2f"%(
          (self.loops - self.last_fps_loops)/(t - self.last_fps_print)))
        sys.stdout.flush()
        self.last_fps_print = t
        self.last_fps_loops = self.loops
    if self.residual:
      d.append(self.correl.dis_res_scal())
    self.send([t-self.t0]+d)

  def save_img(self,t,img):
    image = sitk.GetImageFromArray(img)
    sitk.WriteImage(image,
             self.save_folder + "img_%.6d_%.5f.tiff" % (
             self.loops, t-self.t0))

  def finish(self):
    self.cam.close()
    if self.show_image:
      cv2.destroyAllWindows()
