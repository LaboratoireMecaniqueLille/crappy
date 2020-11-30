# coding: utf-8

import cv2
import sys
import os
try:
  import SimpleITK as sitk
except ImportError:
  print("[Warning] SimpleITK is not installed, cannot save images!")

from ..tool import DISVE as VE
from ..tool import Camera_config
from .masterblock import MasterBlock
from ..camera import Camera


class DISVE(MasterBlock):
  def __init__(self,camera,patches,**kwargs):
    MasterBlock.__init__(self)
    self.niceness = -5
    for arg,default in [("save_folder",None),
                        ("save_period",1),
                        ("labels",None),
                        ("show_fps",True),
                        ("show_image",False),
                        ("alpha",3),
                        ("delta",1),
                        ("gamma",0),
                        ("finest_scale",2),
                        ("init",True),
                        ("iterations",0),
                        ("cam_kwargs",{})
                        ]:
      setattr(self,arg,kwargs.pop(arg,default))
    self.camera = camera
    self.patches = patches
    if self.labels is None:
      self.labels = ['t(s)']+sum(
          [[f'p{i}x', f'p{i}y'] for i in range(len(self.patches))],[])

  def prepare(self):
    if self.save_folder and not os.path.exists(self.save_folder):
      try:
        os.makedirs(self.save_folder)
      except OSError:
        assert os.path.exists(self.save_folder),\
            "Error creating "+self.save_folder
    self.cam = Camera.classes[self.camera]()
    self.cam.open(**self.cam_kwargs)
    config = Camera_config(self.cam)
    config.main()
    t,img0 = self.cam.get_image()
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("DISVE",flags)
    self.loops = 0
    self.last_fps_print = 0
    self.last_fps_loops = 0

  def begin(self):
    t,self.img0 = self.cam.get_image()
    if self.save_folder:
      self.save_img(t,self.img0)
    self.ve = VE(self.img0,self.patches,
        alpha=self.alpha,
        delta=self.delta,
        gamma=self.gamma,
        finest_scale=self.finest_scale,
        init=self.init,
        iterations=self.iterations)

  def loop(self):
    self.loops += 1
    t,img = self.cam.read_image()
    if self.inputs and self.inputs[0].poll():
      self.inputs[0].clear()
      self.ve.img0 = img
      self.img0 = img
      print("[DISVE block] : Resetting L0")
    d = self.ve.calc(img)
    if self.save_folder and not self.loops%self.save_period:
      self.save_img(t,img)
    if self.show_image:
      cv2.imshow("DISVE",img)
      cv2.waitKey(5)
    if self.show_fps:
      if t - self.last_fps_print > 2:
        sys.stdout.write("\rFPS: %.2f"%((self.loops - self.last_fps_loops)/
                              (t - self.last_fps_print)))
        sys.stdout.flush()
        self.last_fps_print = t
        self.last_fps_loops = self.loops
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
