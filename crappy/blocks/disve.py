# coding: utf-8

"""More documentation coming soon !"""

from ..tool import DISVE as VE, Camera_config_with_boxes
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class DISVE(Camera):
  """
  Used to track the motion of specific regions using Disflow

  It uses the class crappy.tool.DISVE to compute the displacement of regions
  """
  def __init__(self,
               camera: str,
               patches: list,
               fields: list = None,
               labels: list = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gditerations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               show_image: bool = False,
               border: float = .1,
               safe: bool = True,
               follow: bool = True,
               **kwargs) -> None:
    self.niceness = -5
    self.cam_kwargs = kwargs
    Camera.__init__(self, camera, **kwargs)
    self.patches = patches
    self.show_image = show_image
    self.fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    if labels is None:
      self.labels = ['t(s)'] + sum(
          [[f'p{i}x', f'p{i}y'] for i in range(len(self.patches))], [])
    else:
      self.labels = labels
    self.ve_kw = {"alpha": alpha,
                  "delta": delta,
                  "gamma": gamma,
                  "finest_scale": finest_scale,
                  "iterations": iterations,
                  "gditerations": gditerations,
                  "patch_size": patch_size,
                  "patch_stride": patch_stride,
                  "border": border,
                  "show_image": show_image,
                  "safe": safe,
                  "follow": follow}

  def prepare(self, *_, **__) -> None:
    config = self.config
    self.config = False
    Camera.prepare(self, send_img=False)
    self.config = config
    if config:
      conf = Camera_config_with_boxes(self.camera, self.patches)
      conf.main()

  def begin(self) -> None:
    t, self.img0 = self.camera.read_image()
    self.ve = VE(self.img0, self.patches, **self.ve_kw)

  def loop(self) -> None:
    t, img = self.get_img()
    if self.inputs and not self.input_label and self.inputs[0].poll():
      self.inputs[0].clear()
      self.ve = VE(img, self.patches, **self.ve_kw)
      self.img0 = img
      print("[DISVE block] : Resetting L0")
    d = self.ve.calc(img)
    self.send([t - self.t0] + d)

  def finish(self) -> None:
    self.ve.close()
    Camera.finish(self)
