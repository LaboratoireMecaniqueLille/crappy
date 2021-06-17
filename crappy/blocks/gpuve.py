# coding: utf-8

from time import time
import numpy as np
try:
  from pycuda.tools import make_default_context
  from pycuda.driver import init as cuda_init
except (ModuleNotFoundError, ImportError):
  def cuda_init():
    print("PyCUDA is could not be imported, cannot use GPUVE block")
    raise ModuleNotFoundError("pycuda")

from ..tool import GPUCorrel as GPUCorrel_tool
from .camera import Camera


class GPUVE(Camera):
  """An optical Videoextensometry measuring the displacement of small areas
  using GPU accelerated DIC.

  This block simply returns the displacement of each region along `x` and `y`
  in pixel. It will not return the strain as it does not know how the patches
  are arranged. It should be done by another block or a condition if necessary.

  Important:
    ``patches`` must be a :obj:`list` of :obj:`tuple` of length `4`. Each tuple
    contains the origin and the size of each patch along `Y` and `X`
    respectively (i.e. `Oy, Ox, Ly, Lx`).
  """

  def __init__(self,
               camera,
               patches,
               save_folder=None,
               verbose=False,
               labels=None,
               fps_label=False,
               img_name="{self.loops:06d}_{t-self.t0:.6f}",
               ext='tiff',
               save_period=1,
               save_backend=None,
               transform=None,
               input_label=None,
               config=True,
               cam_kwargs=None,
               **kwargs):
    self.ready = False
    cam_kw = {}
    self.patches = patches
    # Kwargs to be given to the camera BLOCK
    # ie save_folder, config, etc... but NOT the labels

    cam_kw['save_folder'] = save_folder
    cam_kw['verbose'] = verbose
    cam_kw['fps_label'] = fps_label
    cam_kw['img_name'] = img_name
    cam_kw['ext'] = ext
    cam_kw['save_period'] = save_period
    cam_kw['save_backend'] = save_backend
    cam_kw['transform'] = transform
    cam_kw['input_label'] = input_label
    cam_kw['config'] = config

    self.verbose = cam_kw['verbose']  # Also, we keep the verbose flag
    if cam_kwargs is not None:
      cam_kw.update(cam_kwargs)
    Camera.__init__(self, camera, **cam_kw)
    self.transform = cam_kw.get("transform")

    # Creating the tuple of labels (to name the outputs)
    if labels is not None:
      assert len(labels) == len(patches) * 2,\
          "The number of labels must be twice the number of patches (x and y)"
      self.labels = ['t(s)'] + list(labels)
    # Else if we got a default field as a string,
    # use this string (ex: fields=('x', 'y', 'r', 'exx', 'eyy'))
    else:
      self.labels = ['t(s)'] + sum([
          [f'p{i}x', f'p{i}y'] for i in range(len(patches))], [])
    # Handle res parameters: if true, also return the residual
    self.res = kwargs.get("res", True)
    if self.res:
      self.labels += ("res",)

    self.cam_kwargs = {} if cam_kwargs is None else cam_kwargs
    self.save_folder = save_folder
    self.save_period = 1
    self.kwargs = kwargs

  def prepare(self):
    cuda_init()
    self.context = make_default_context()
    Camera.prepare(self, send_img=False)
    t, img = self.camera.read_image()
    if self.transform is not None:
      img = self.transform(img)
    self.correl = []
    for oy, ox, h, w in self.patches:
      self.correl.append(GPUCorrel_tool((h, w),
        fields=['x', 'y'], context=self.context, levels=1, **self.kwargs))
    self.loops = 0
    self.nloops = 50

  def begin(self):
    t, img = self.camera.read_image()
    for c, (oy, ox, h, w) in zip(self.correl, self.patches):
      c.set_orig(
          self.transform(img[oy:oy + h, ox:ox + w]).astype(np.float32))
      c.prepare()
    self.last_t = time() - 1
    if self.save_folder:
      self.save(self.save_folder + "img_ref_%.5f.tiff" % (t - self.t0))

  def loop(self):
    if self.verbose and self.loops % self.nloops == 0:
      t = time()
      print("[VE block] processed", self.nloops / (t - self.last_t), "ips")
      self.last_t = t
    self.loops += 1
    t, img = self.get_img()
    out = [t - self.t0]
    # + self.correl.get_disp(
    #      self.transform(img).astype(np.float32)).tolist()
    for c, (oy, ox, h, w) in zip(self.correl, self.patches):
      out.extend(c.get_disp(self.transform(
          img[oy:oy + h, ox:ox + w]).astype(np.float32)).tolist())
      # out.extend([np.sum(self.transform(
      #   img[oy:oy+h, ox:ox+w]).astype(np.float32)),0])
    if self.res:
      pass  # TODO
    self.send(out)
