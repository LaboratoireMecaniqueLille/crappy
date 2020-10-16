# coding: utf-8

from sys import platform
import os
import time
try:
  import SimpleITK as sitk
except ImportError:
  sitk = None
import cv2

from .masterblock import MasterBlock
from ..camera import camera_list
from ..tool import Camera_config

kw = dict(
  ("save_folder",None),
  ("verbose",False),
  ("labels",['t(s)','frame']),
  ("fps_label",False),
  ("img_name","{self.loops:06d}_{t-self.t0:.6f}"),
  ("ext","tiff"),
  ("save_period",1),
  ("save_backend",None),
  ("config",True)
)


class Camera(MasterBlock):
  """
  Streams pictures.
  """
  def __init__(self,camera,**kwargs):
    """
    Read images from a camera object, save and/or send them to another block.

    It can be triggered by an other block, internally,
    or try to run at a given framerate

    kwargs:
      camera : {"Ximea","Jai","Webcam",...}
        See crappy.camera.MasterCam.classes for a full list
        (str, mandatory)
      save_folder : directory to save the images. It will be created
        if necessary. If None, it will not save the images
        (str or None, default: None)
      verbose : If True, the block will print the number of fps
          (bool, default: False)
      labels : The labels for respectively time and the frame
          (list of strings, default: ['t(s)','frame'])
      fps_label : If set, self.max_fps will be set to the value received by
        the block with this label
      img_name : Template for the name of the image to save
        it will be evaluated as a f-string
        (string, default: "{self.loops:06d}_{t-self.t0:.6f}"
      ext : Extension of the image. Make sure it is supported by the
        image saving backend (str,default: "tiff")
      save_period : Will save only one in x images
        (int, default : 1)
      save_backend : module to use to save the images
        supported backends : sitk, cv2
        if None will try sitk, else cv2
        (str, default: None)
      config : Show the popup for config ? (bool, default: True)
    """
    MasterBlock.__init__(self)
    self.niceness = -10
    for arg,default in kw.items():
      setattr(self,arg,kwargs.get(arg,default)) # Assign these attributes
      try:
        del kwargs[arg]
        # And remove them (if any) to
        # keep the parameters for the camera
      except KeyError:
        pass
    self.camera_name = camera
    self.cam_kw = kwargs
    assert self.camera_name in camera_list,"{} camera does not exist!".format(
        self.camera_name)
    if self.save_backend is None:
      if sitk is None:
        self.save_backend = "cv2"
      else:
        self.save_backend = "sitk"
    assert self.save_backend in ["cv2","sitk"],\
        "Unknown saving backend: "+self.save_backend
    self.save = getattr(self,"save_"+self.save_backend)

  def prepare(self):
    sep = '\\' if 'win' in platform else '/'
    if self.save_folder and not self.save_folder.endswith(sep):
      self.save_folder += sep
    if self.save_folder and not os.path.exists(self.save_folder):
      try:
        os.makedirs(self.save_folder)
      except OSError:
        assert os.path.exists(self.save_folder),\
            "Error creating "+self.save_folder
    self.camera = camera_list[self.camera_name]()
    self.camera.open(**self.cam_kw)
    self.ext_trigger = True if self.inputs and not self.fps_label else False
    if self.config:
      conf = Camera_config(self.camera)
      conf.main()

  def begin(self):
    self.timer = time.time()
    self.fps_timer = self.timer
    self.data = {}
    self.last_index = 0
    self.loops = 0

  def save_sitk(self,img,fname):
    image = sitk.GetImageFromArray(img)
    sitk.WriteImage(image,fname)

  def save_cv2(self,img,fname):
    cv2.imwrite(fname,img)

  def loop(self):
    if not self.ext_trigger:
      if self.fps_label:
        while self.inputs[0].poll():
          self.camera.max_fps = self.inputs[0].recv()[self.fps_label]
      t,img = self.camera.read_image()
    else:
      data = self.inputs[0].recv()  # wait for a signal
      if data is None:
        return
      t,img = self.camera.get_image()
    self.timer = time.time()
    if self.save_folder and self.loops % self.save_period == 0:
      self.save(img, self.save_folder +
          eval('f"{}"'.format(self.img_name)) + f".{self.ext}")
    self.loops += 1
    self.send([t-self.t0,img])

  def finish(self):
    self.camera.close()
