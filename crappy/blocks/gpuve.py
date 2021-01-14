#coding: utf-8

from time import time
import numpy as np
try:
  from pycuda.tools import make_default_context
  from pycuda.driver import init as cuda_init
except (ModuleNotFoundError,ImportError):
  def cuda_init():
    print("PyCUDA is could not be imported, cannot use GPUVE block")
    raise ModuleNotFoundError("pycuda")

from ..tool import GPUCorrel as GPUCorrel_tool
from .camera import Camera,kw as default_cam_block_kw


class GPUVE(Camera):
  """
  An optical Videoextensometry measuring the displacement of small
  areas using GPU accelerated DIC

  Patches must be a list of tuples of length 4
  Each tuple contains the origin and the size of
  each patch along Y and X respectively (ie Oy,Ox,Ly,Lx)

  This block simply returns the displacement of each region along x and y
  in pixel.

  This block will not return the strain as it does not know
  how the patches are arranged.
  It should be done by another block or a condition if necessary
  LATEST VERSION IS UNTESTED
  """
  def __init__(self, camera, patches, **kwargs):
    self.ready = False
    self.patches = patches
    cam_kw = {}
    # Kwargs to be given to the camera BLOCK
    # ie save_folder, config, etc... but NOT the labels
    for k,v in default_cam_block_kw.items():
      if k == 'labels':
        continue
      cam_kw[k] = kwargs.pop(k,v)
    self.verbose = cam_kw['verbose'] # Also, we keep the verbose flag
    cam_kw.update(kwargs.pop('cam_kwargs',{}))
    Camera.__init__(self,camera,**cam_kw)
    self.transform = cam_kw.get("transform")

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
    Camera.prepare(self)
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
      self.save(self.save_folder + "img_ref_%.5f.tiff" % (t-self.t0))

  def loop(self):
    if self.verbose and self.loops%self.nloops == 0:
      t = time()
      print("[VE block] processed", self.nloops / (t-self.last_t), "ips")
      self.last_t = t
    self.loops += 1
    t,img = self.get_img()
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
