# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup StreamerCamera StreamerCamera
# @{

## @file _streamerCamera.py
# @brief Streams pictures.
# @author V1ctor Couty, Robin Siemiatkowski
# @version 0.2
# @date 10/01/2017
from __future__ import print_function

import os
import time
import sys
import SimpleITK as sitk

from .masterblock import MasterBlock
from ..camera import Camera_wrapper as Cam
from ..tool import Camera_config


class Camera(MasterBlock):
  """
  Streams pictures.
  """
  def __init__(self,camera="Fake_camera",**kwargs):
    """
    This block fetch images from a camera object, can save and/or
    transmit them to another block.

    It can be triggered by an other block, internally,
    or run as fast as possible

    kwargs:
      camera : {"Ximea","Jai","Webcam",...}
        See crappy.camera.MasterCam.classes for a full list
      save_folder : directory, default = None
        directory to save the images. If inexistant, will be created.
        If None, will not save the images
      show_fps:
        Will print fps every 2 seconds in the console
      labels : string, default=['t(s)','frame']
        The labels for respectively time and the frame
      See below for default values
    """
    MasterBlock.__init__(self)
    self.camera = camera
    for arg,default in [("save_folder",None),
                        ("label","cycle"),
                        ("show_fps",False),
                        ("labels",['t(s)','frame']),
                        ("config",True)]:
      setattr(self,arg,kwargs.get(arg,default)) #Assign these attributes
      try:
        del kwargs[arg] # And remove them (if any) to
                        # keep the parameters for the technical
      except KeyError:
        pass
    self.camera_name = self.camera
    self.cam_kw = kwargs

  def prepare(self):
    if self.save_folder and not os.path.exists(self.save_folder):
      os.makedirs(self.save_folder)
    self.camera = Cam(self.camera_name, **self.cam_kw)
    self.camera.open()
    self.trigger = "internal" if len(self.inputs) == 0 else "external"
    if self.config:
      Camera_config(self.camera)

  def begin(self):
    self.timer = time.time()
    self.fps_timer = self.timer
    self.data = {}
    self.last_index = 0
    self.loops = 0

  def loop(self):
    if self.show_fps and self.timer - self.fps_timer > 2:
      sys.stdout.write("\r[StreamerCamera] FPS: %2.2f" % (
                (self.loops - self.last_index) / (self.timer - self.fps_timer)))
      sys.stdout.flush()
      self.fps_timer = self.timer
      self.last_index = self.loops
    if self.trigger == "internal":
      t,img = self.camera.read_image()
    elif self.trigger == "external":
      data = self.inputs[0].recv()  # wait for a signal
      if data is None:
        return
      t,img = self.camera.get_image()
    self.timer = time.time()
    if self.save_folder:
      image = sitk.GetImageFromArray(img)
      try:
        cycle = data[self.label] # If we received a data to add in the name
        sitk.WriteImage(image,
               self.save_folder + "img_%.6d_cycle%09.1f_%.5f.tiff" % (
               self.loops, cycle, t-self.t0))
      except (KeyError,UnboundLocalError): # If we did not
        sitk.WriteImage(image,
               self.save_folder + "img_%.6d_%.5f.tiff" % (
               self.loops, t-self.t0))
    self.loops += 1
    self.send([t-self.t0,img])

  def finish(self):
    self.camera.close()

    def __repr__(self):
      return "Streamer Camera (%s)"%self.camera_name
