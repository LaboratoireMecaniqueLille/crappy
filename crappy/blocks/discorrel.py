# coding: utf-8

"""More documentation coming soon !"""

import numpy as np

from ..tool import DISCorrel as Dis
from ..tool import DISConfig
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


def draw_box(box, img):
  for s in [
      (box[0], slice(box[1], box[3])),
      (box[2], slice(box[1], box[3])),
      (slice(box[0], box[2]), box[1]),
      (slice(box[0], box[2]), box[3])
   ]:
    # Turn these pixels white or black for highest possible contrast
    img[s] = 255 * int(np.mean(img[s]) < 128)


class DISCorrel(Camera):
  def __init__(self, camera,
               fields=None,
               labels=None,
               alpha=3,
               delta=1,
               gamma=0,
               finest_scale=1,
               iterations=1,
               gditerations=10,
               init=True,
               patch_size=8,
               patch_stride=3,
               show_image=False,
               residual=False,
               residual_full=False,
               **kwargs):
    self.niceness = -5
    self.cam_kwargs = kwargs
    kwargs['config'] = False  # We have our own config window
    Camera.__init__(self, camera, **kwargs)
    self.fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    self.labels = ['t(s)', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)'] \
      if labels is None else labels
    self.show_image = show_image
    self.residual = residual
    self.residual_full = residual_full
    self.dis_kw = {"alpha": alpha,
                   "delta": delta,
                   "gamma": gamma,
                   "finest_scale": finest_scale,
                   "init": init,
                   "iterations": iterations,
                   "gditerations": gditerations,
                   "patch_size": patch_size,
                   "patch_stride": patch_stride}
    if self.residual:
      self.labels.append('res')
    if self.residual_full:
      self.labels.append('res_full')

  def prepare(self):
    Camera.prepare(self, send_img=False)
    config = DISConfig(self.camera)
    config.main()
    self.bbox = config.box
    if not all(i > 0 for i in self.bbox):
      raise AttributeError("Incorrect bounding box sepcified in DISCorrel. "
          "Was the region selected on the configuration Window ?")
    t, img0 = self.camera.get_image()
    self.correl = Dis(img0, bbox=self.bbox, fields=self.fields, **self.dis_kw)
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("DISCorrel", flags)

  def begin(self):
    t, self.img0 = self.camera.read_image()
    if self.transform is not None:
      self.correl.img0 = self.transform(self.img0)
    else:
      self.correl.img0 = self.img0
    if self.save_folder:
      self.save(self.img0,
          self.save_folder + "img_ref_%.6f.tiff" % (t - self.t0))

  def loop(self):
    t, img = self.get_img()
    if self.inputs and not self.input_label and self.inputs[0].poll():
      self.inputs[0].clear()
      self.img0 = img
      if self.transform is not None:
        self.correl.img0 = self.transform(self.img0)
      else:
        self.correl.img0 = self.img0
      print("[CORREL block] : Resetting L0")

    d = self.correl.calc(img)
    if self.show_image:
      draw_box(self.bbox, img)
      cv2.imshow("DISCorrel", img)
      cv2.waitKey(5)
    if self.residual:
      d.append(self.correl.dis_res_scal())
    self.send([t - self.t0] + d)

  def finish(self):
    if self.show_image:
      cv2.destroyAllWindows()
    Camera.finish(self)
