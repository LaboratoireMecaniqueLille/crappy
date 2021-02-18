#coding: utf-8

from time import time
import numpy as np
import SimpleITK as sitk
import os
#from pycuda.tools import make_default_context
#from pycuda.driver import init as cuda_init

from .masterblock import MasterBlock
from ..tool import Camera_config,GPUCorrel as GPUCorrel_tool
from ..camera import camera_list


class GPUVE(MasterBlock):
  """
  An optical Videoextensometry measuring the displacement of small
  areas using GPU accelerated DIC.

  Warning!
    Patches must be a list of tuples of length 4.

    Each tuple contains the origin and the size of
    each patch along Y and X respectively (ie Oy,Ox,Ly,Lx).

  Note:
    This block simply returns the displacement of each region along x and y
    in pixel.

    This block will not return the strain as it does not know
    how the patches are arranged.

    It should be done by another block or a condition if necessary.

  """

  def __init__(self, camera, patches, **kwargs):
    MasterBlock.__init__(self)
    self.ready = False
    self.camera_name = camera
    self.patches = patches
    self.verbose = kwargs.get("verbose", 0)
    self.config = kwargs.get("config", True)
    # A function to apply to the image
    self.transform = kwargs.pop("transform",lambda i: i)

    # Creating the tuple of labels (to name the outputs)
    labels = kwargs.pop('labels',None)
    if labels is not None:
      assert len(labels) == len(patches)*2,\
          "The number of labels must be twice the number of patches (x and y)"
      self.labels = ['t(s)']+list(labels)
    # Else if we got a default field as a string,
    # use this string (ex: fields=('x','y','r','exx','eyy'))
    else:
      self.labels = ['t(s)']+sum([
        [f'p{i}x',f'p{i}y'] for i in range(len(patches))],[])
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
    cuda_init()
    self.context = make_default_context()
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

    self.correl = []
    for oy,ox,h,w in self.patches:
      self.correl.append(GPUCorrel_tool((h,w),
        fields=['x','y'], context=self.context, levels=1, **self.kwargs))
    self.loops = 0
    self.nloops = 50

  def begin(self):
    t,img = self.camera.read_image()
    for c,(oy,ox,h,w) in zip(self.correl,self.patches):
      c.setOrig(
          self.transform(img[oy:oy+h,ox:ox+w]).astype(np.float32))
      c.prepare()
    self.last_t = time() - 1
    if self.save_folder:
      image = sitk.GetImageFromArray(img)
      sitk.WriteImage(image,
               self.save_folder + "img_ref_%.5f.tiff" % (t-self.t0))

  def loop(self):
    if self.verbose and self.loops%self.nloops == 0:
      t = time()
      print("[VE block] processed", self.nloops / (t-self.last_t), "ips")
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
    out = [t-self.t0]
    #+ self.correl.getDisp(
    #      self.transform(img).astype(np.float32)).tolist()
    for c,(oy,ox,h,w) in zip(self.correl,self.patches):
      out.extend(c.getDisp(self.transform(
        img[oy:oy+h,ox:ox+w]).astype(np.float32)).tolist())
      #out.extend([np.sum(self.transform(
      #  img[oy:oy+h,ox:ox+w]).astype(np.float32)),0])
    if self.res:
      pass # TODO
    self.send(out)
