#coding: utf-8

from time import time
import numpy as np
import SimpleITK as sitk
import os

from .masterblock import MasterBlock
from ..tool import Camera_config,GPUCorrel as GPUCorrel_tool
from ..camera import camera_list


class GPUCorrel(MasterBlock):
  """
    This block uses the Correl class (in crappy/tool/correl.py)

    See the docstring of Correl to have more informations about the
        arguments specific to Correl.
    It will try to identify the deformation parameters for each fields.
    If you use custom fields, you can use labels=(...) to name the data
        sent through the link.
    If no labels are specified, custom fields will be named by their position.
    Note that the reference image is only taken once, when the
        .start() method is called (after dropping the first image).
  """

  def __init__(self, camera, **kwargs):
    MasterBlock.__init__(self)
    self.ready = False
    self.camera_name = camera
    self.Nfields = kwargs.get("Nfields")
    self.verbose = kwargs.get("verbose", 0)
    self.config = kwargs.get("config", True)
    # A function to apply to the image
    self.transform = kwargs.pop("transform",None)
    self.discard_lim = kwargs.pop("discard_lim",3)
    self.discard_ref = kwargs.pop("discard_ref",5)
    # If the residual of the image exceeds <discard_lim> times the
    # average of the residual of the last <discard_ref> images,
    # do not send the result (requires res=True)
    if self.Nfields is None:
      try:
        self.Nfields = len(kwargs.get("fields"))
      except TypeError:
        print("Error: Correl needs to know the number of fields at init \
with fields=(.,.) or Nfields=k")
        raise NameError('Missing fields')

    # Creating the tuple of labels (to name the outputs)
    self.labels = ('t(s)',)
    for i in range(self.Nfields):
      # If explicitly named with labels=(...)
      if kwargs.get("labels") is not None:
        self.labels += (kwargs.get("labels")[i],)
      # Else if we got a default field as a string,
      # use this string (ex: fields=('x','y','r','exx','eyy'))
      elif kwargs.get("fields") is not None and \
          isinstance(kwargs.get("fields")[i], str):
        self.labels += (kwargs.get("fields")[i],)
      # Custom field and no label given: name it by its position...
      else:
        self.labels += (str(i),)

    # We don't need to pass these arg to the Correl class
    if kwargs.get("labels") is not None:
      del kwargs["labels"]
    # Handle res parameters: if true, also return the residual
    self.res = kwargs.get("res",True)
    if self.res:
      self.labels += ("res",)
    if "cam_kwargs" in kwargs:
      self.cam_kwargs = kwargs["cam_kwargs"]
      del kwargs["cam_kwargs"]
    else:
      self.cam_kwargs = {}
    if "save_folder" in kwargs:
      self.save_folder = kwargs["save_folder"]
      del kwargs["save_folder"]
    else:
      self.save_folder = None
    if "save_period" in kwargs:
      self.save_period = kwargs["save_period"]
      del kwargs["save_period"]
    else:
      self.save_period = 1
    self.kwargs = kwargs

  def prepare(self):
    if self.save_folder and not os.path.exists(self.save_folder):
      try:
        os.makedirs(self.save_folder)
      except OSError: # May happen if another blocks created the folder
        assert os.path.exists(self.save_folder),\
            "Error creating "+self.save_folder
    self.camera = camera_list[self.camera_name]()
    self.camera.open(**self.cam_kwargs)
    if self.config:
      Camera_config(self.camera).main()
    t,img = self.camera.read_image()
    if self.transform is not None:
      img = self.transform(img)
    self.correl = GPUCorrel_tool(img.shape, **self.kwargs)
    self.loops = 0
    self.nloops = 50
    self.res_hist = [np.inf]

  def begin(self):
    t,img = self.camera.read_image()
    if self.transform is not None:
      self.correl.setOrig(self.transform(img).astype(np.float32))
    else:
      self.correl.setOrig(img.astype(np.float32))
    self.correl.prepare()
    self.last_t = time() - 1
    if self.save_folder:
      image = sitk.GetImageFromArray(img)
      sitk.WriteImage(image,
               self.save_folder + "img_ref_%.5f.tiff" % (t-self.t0))

  def loop(self):
    if self.verbose and self.loops%self.nloops == 0:
      t = time()
      print("[Correl block] processed", self.nloops / (t-self.last_t), "ips")
      self.last_t = t
    self.loops += 1
    if self.inputs: # If we have an input: external trigger
      data = self.inputs[0].recv()
      if data is None:
        return
      else:
        t,img = self.camera.get_image() # No fps control
    else:
      t,img = self.camera.read_image() # Limits to max_fps
    if self.save_folder and self.loops % self.save_period == 0:
      image = sitk.GetImageFromArray(img)
      sitk.WriteImage(image,
               self.save_folder + "img_%.6d_%.5f.tiff" % (
               self.loops, t-self.t0))
    if self.transform is not None:
      out = [t-self.t0] + self.correl.getDisp(
          self.transform(img).astype(np.float32)).tolist()
    else:
      out = [t-self.t0] + self.correl.getDisp(img.astype(np.float32)).tolist()
    if self.res:
      out += [self.correl.getRes()]
      if self.discard_lim:
        self.res_hist = self.res_hist+[out[-1]]
        self.res_hist = self.res_hist[-self.discard_ref-1:]
        if self.res_hist[-1] > self.discard_lim*np.average(self.res_hist[:-1]):
          print("[Correl block] Residual too high, not sending values")
          return
    self.send(out)