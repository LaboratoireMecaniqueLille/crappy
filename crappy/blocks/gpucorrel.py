# coding: utf-8

from time import time
import numpy as np

from ..tool import GPUCorrel as GPUCorrel_tool
from .camera import Camera


class GPUCorrel(Camera):
  """This block uses the :ref:`GPU Correl` class.

  See the documentation of :ref:`GPU Correl` to have more information about the
  arguments specific to this class.

  It will try to identify the deformation parameters for each fields. If you
  use custom fields, you can use ``labels=(...)`` to name the data sent through
  the link. If no labels are specified, custom fields will be named by their
  position.

  The reference image is only taken once, when the :meth:`start` method is
  called (after dropping the first image).
  """

  def __init__(self,
               camera,
               fields,
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
               discard_lim=3,
               discard_ref=5,
               imgref=None,
               **kwargs):
    self.ready = False
    cam_kw = {}
    self.fields = fields
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
    # A function to apply to the image
    self.transform = cam_kw.get("transform")
    self.discard_lim = discard_lim
    self.discard_ref = discard_ref
    # If the residual of the image exceeds <discard_lim> times the
    # average of the residual of the last <discard_ref> images,
    # do not send the result (requires res=True)

    # Creating the tuple of labels (to name the outputs)
    self.labels = ('t(s)',)
    for i in range(len(self.fields)):
      # If explicitly named with labels=(...)
      if labels is not None:
        self.labels += (labels[i],)
      # Else if we got a default field as a string,
      # use this string (ex: fields=('x', 'y', 'r', 'exx', 'eyy'))
      elif isinstance(fields[i], str):
        self.labels += (fields[i],)
      # Custom field and no label given: name it by its position...
      else:
        self.labels += (str(i),)

    # Handle res parameters: if true, also return the residual
    self.res = kwargs.get("res", True)
    if self.res:
      self.labels += ("res",)
    self.imgref = imgref
    self.gpu_correl_kwargs = kwargs
    self.gpu_correl_kwargs['fields'] = self.fields

  def prepare(self):
    Camera.prepare(self, send_img=False)
    t, img = self.camera.read_image()
    if self.transform is not None:
      img = self.transform(img)
    self.correl = GPUCorrel_tool(img.shape, **self.gpu_correl_kwargs)
    self.loops = 0
    self.nloops = 50
    self.res_hist = [np.inf]
    if self.imgref is not None:
      if self.transform is not None:
        self.correl.set_orig(self.transform(self.imgref.astype(np.float32)))
      else:
        self.correl.set_orig(self.imgref.astype(np.float32))
      self.correl.prepare()

  def begin(self):
    self.last_t = time() - 1
    if self.imgref is not None:
      return
    t, img = self.camera.read_image()
    if self.transform is not None:
      self.correl.set_orig(self.transform(img).astype(np.float32))
    else:
      self.correl.set_orig(img.astype(np.float32))
    self.correl.prepare()
    if self.save_folder:
      self.save(img, self.save_folder + "img_ref_%.6f.tiff" % (t - self.t0))

  def loop(self):
    if self.verbose and self.loops % self.nloops == 0:
      t = time()
      print("[Correl block] processed", self.nloops / (t - self.last_t), "ips")
      self.last_t = t
    t, img = self.get_img()
    out = [t - self.t0] + self.correl.get_disp(img.astype(np.float32)).tolist()
    if self.res:
      out += [self.correl.get_res()]
      if self.discard_lim:
        self.res_hist = self.res_hist + [out[-1]]
        self.res_hist = self.res_hist[-self.discard_ref - 1:]
        if self.res_hist[-1] > \
                self.discard_lim * np.average(self.res_hist[:-1]):
          print("[Correl block] Residual too high, not sending values")
          return
    self.send(out)

  def finish(self):
    self.correl.clean()
    Camera.finish(self)
