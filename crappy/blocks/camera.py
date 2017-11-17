# coding: utf-8

from __future__ import print_function

import os
import time
try:
  import SimpleITK as sitk
except ImportError:
  sitk = None

from .masterblock import MasterBlock
from ..camera import camera_list
from ..tool import Camera_config


class Camera(MasterBlock):
  """
  Streams pictures.
  """
  def __init__(self,camera="Fake_camera",**kwargs):
    """
    Read images from a camera object, save and/or send them to another block.

    It can be triggered by an other block, internally,
    or try to run at a given framerate

    kwargs:
      camera : {"Ximea","Jai","Webcam",...}
        See crappy.camera.MasterCam.classes for a full list
        (str, default="Fake_camera")
      save_folder : directory to save the images. It will be created
        if necessary. If None, it will not save the images
        (str or None, default: None)
      verbose : If True, the block will print the number of fps
          (bool, default=False)
      labels : string, default=['t(s)','frame']
        The labels for respectively time and the frame
      config : Show the popup for config ? (bool, default = True)
    """
    MasterBlock.__init__(self)
    self.niceness = -10
    for arg,default in [("save_folder",None),
                        ("verbose",False),
                        ("labels",['t(s)','frame']),
                        ("config",True)]:
      setattr(self,arg,kwargs.get(arg,default)) # Assign these attributes
      try:
        del kwargs[arg] # And remove them (if any) to
                        # keep the parameters for the camera
      except KeyError:
        pass
    self.camera_name = camera
    self.cam_kw = kwargs
    assert self.camera_name in camera_list,"{} camera does not exist!".format(
                                        self.camera_name)

  def prepare(self):
    if self.save_folder and not os.path.exists(self.save_folder):
      os.makedirs(self.save_folder)
    self.camera = camera_list[self.camera_name]()
    self.camera.open(**self.cam_kw)
    self.trigger = "internal" if len(self.inputs) == 0 else "external"
    if self.config:
      conf = Camera_config(self.camera)
      conf.main()

  def begin(self):
    self.timer = time.time()
    self.fps_timer = self.timer
    self.data = {}
    self.last_index = 0
    self.loops = 0

  def loop(self):
    if self.trigger == "internal":
      t,img = self.camera.read_image()
    elif self.trigger == "external":
      data = self.inputs[0].recv()  # wait for a signal
      if data is None:
        return
      t,img = self.camera.get_image()
    self.timer = time.time()
    if self.save_folder:
      if not sitk:
        raise IOError("[Camera] Cannot save image, sitk is not installed !")
      image = sitk.GetImageFromArray(img)
      sitk.WriteImage(image,
               self.save_folder + "img_%.6d_%.5f.tiff" % (
               self.loops, t-self.t0))
    self.loops += 1
    self.send([t-self.t0,img])

  def finish(self):
    self.camera.close()
