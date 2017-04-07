# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup VideoExtenso VideoExtenso
# @{

## @file videoExtenso.py
# @brief Detects spots (2,3 or 4) on images, and evaluate the deformations
# @authors Victor Couty
# @version 0.2
# @date 04/04/2017
from __future__ import print_function

import cv2
import sys

from ..tool.videoextenso import LostSpotError,Video_extenso as VE
from ..tool.videoextensoConfig import VE_config
from .masterblock import MasterBlock
from ..camera import Camera

class Video_extenso(MasterBlock):
  def __init__(self,**kwargs):
    MasterBlock.__init__(self)
    default_labels = ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    for arg,default in [("camera","Ximea"),
                        ("max_fps",None),
                        ("save_folder",None),
                        ("save_period",1),
                        ("labels",default_labels),
                        ("show_fps",True),
                        ("show_image",False)
                        ]:
      try:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      except KeyError:
        setattr(self,arg,default)
    self.ve_kwargs = {}
    for arg in ['white_spots','update_thresh','num_spots','safe_mode','border']:
      if arg in kwargs:
        self.ve_kwargs[arg] = kwargs[arg]
        del kwargs[arg]
    self.cam_kwargs = kwargs

  def prepare(self):
    self.cam = Camera.classes[self.camera](**self.cam_kwargs)
    self.cam.open()
    self.ve = VE(**self.ve_kwargs)
    config = VE_config(self.cam,self.ve)
    config.main()
    self.ve.start_tracking()
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("Videoextenso",flags)
    self.loops = 0
    self.last_fps_print = 0
    self.last_fps_loops = 0

  def loop(self):
    self.loops += 1
    t,img = self.cam.read_image()
    try:
      d = self.ve.get_def(img)
    except LostSpotError:
      print("[VE block] Lost spots, terminating")
      self.ve.stop_tracking()
      raise
    if self.show_image:
      boxes = map(lambda r: r['bbox'],self.ve.spot_list)
      for miny,minx,maxy,maxx in boxes:
        img[miny,minx:maxx] = 255
        img[maxy,minx:maxx] = 255
        img[miny:maxy,minx] = 255
        img[miny:maxy,maxx] = 255
      cv2.imshow("Videoextenso",img)
      cv2.waitKey(5)
    if self.show_fps:
      if t - self.last_fps_print > 2:
        sys.stdout.write("\rFPS: %.2f"%((self.loops - self.last_fps_loops)
                              /(t - self.last_fps_print)))
        sys.stdout.flush()
        self.last_fps_print = t
        self.last_fps_loops = self.loops


    centers = map(lambda r: (r['y'],r['x']),self.ve.spot_list)
    self.send([t-self.t0,centers]+d)



