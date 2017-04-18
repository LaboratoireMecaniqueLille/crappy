#coding; utf-8

from time import time
import numpy as np
import SimpleITK as sitk

from .masterblock import MasterBlock
from ..tool import Camera_config,Correl as Correl_class
from ..camera import camera_list

class Correl(MasterBlock):
  """
    This block uses the Correl class (in crappy/technicals/_correl.py)

    See the docstring of Correl to have more informations about the
        arguments specific to Correl.
    It will try to identify the deformation parameters for each fields.
    If you use custom fields, you can use labels=(...) to name the data
        sent through the link.
    If no labels are specified, custom fields will be named by their position.
    Note that the reference image is only taken once, when the
        .start() method is called (after dropping the first image).
  """

  def __init__(self, camera="Ximea", **kwargs):
    MasterBlock.__init__(self)
    self.ready = False
    self.camera_name = camera
    self.Nfields = kwargs.get("Nfields")
    self.verbose = kwargs.get("verbose", 0)
    if self.Nfields is None:
      try:
        self.Nfields = len(kwargs.get("fields"))
      except TypeError:
        print "Error: Correl needs to know the number of fields at init \
with fields=(.,.) or Nfields=k"
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
    if kwargs.get("res") is not None:
      self.res = kwargs["res"]
      del kwargs["res"]
      self.labels += ("res",)
    else:
      self.res = False
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
    self.kwargs = kwargs

  def prepare(self):
    self.camera = camera_list[self.camera_name](**self.cam_kwargs)
    self.camera.open()
    Camera_config(self.camera).main()
    t,img = self.camera.read_image()
    self.correl = Correl_class(img.shape, **self.kwargs)
    self.loops = 0

  def main(self):
    nLoops = 100  # Info will be printed every nLoops (if verbose)
    t2 = time() - 1
    # This is the only time the original picture is set, so the residual may
    # increase if lightning vary or large displacements are reached
    t,img = self.camera.read_image()
    self.correl.setOrig(img.astype(np.float32))
    self.correl.prepare()
    while True:
      if self.verbose:
        t1 = time()
        print "[Correl block] processed", nLoops / (t1 - t2), "ips"
        t2 = t1
      for i in range(nLoops):
        self.loops += 1
        t,img = self.camera.read_image()
        if self.save_folder:
          sitk.WriteImage(sitk.getImageFromArray(img),self.save_folder
          +"img_%.6d.png"%self.loops)


        self.correl.setImage(img.astype(np.float32))
        out = [t] + self.correl.getDisp().tolist()
        if self.res:
          out += [self.correl.getRes()]
        self.send(out)
