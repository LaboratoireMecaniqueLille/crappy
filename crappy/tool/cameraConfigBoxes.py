# coding: utf-8

import numpy as np
from .cameraConfig import Camera_config
from .._global import OptionalModule

try:
  from PIL import ImageTk, Image
except (ModuleNotFoundError, ImportError):
  ImageTk = OptionalModule("pillow")
  Image = OptionalModule("pillow")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Camera_config_with_boxes(Camera_config):
  """
  Config window for camera, with lines to highlight boxes on the image

  Used for VE blocks when the selection is not interactive (unlike ve_config)
  """
  def __init__(self, camera, boxes):
    self.boxes = boxes
    Camera_config.__init__(self, camera)

  def clamp(self, t: tuple) -> tuple:
    if isinstance(t[0], slice):
      return t[0], min(max(0, t[1]), self.img_shape[1] - 1)
    else:
      return min(max(0, t[0]), self.img_shape[0] - 1), t[1]

  def draw_box(self, box, img):
    miny, minx, h, w = box
    maxy = miny + h
    maxx = minx + w
    for s in [
        (miny, slice(minx, maxx)),
        (maxy, slice(minx, maxx)),
        (slice(miny, maxy), minx),
        (slice(miny, maxy), maxx)
     ]:
      # Turn these pixels white or black for highest possible contrast
      s = self.clamp(s)
      img[s] = 255 * int(np.mean(img[s]) < 128)

  def resize_img(self, sl: tuple) -> None:
    rimg = cv2.resize(self.img8[sl[1], sl[0]], tuple(reversed(self.img_shape)),
                      interpolation=0)
    for b in self.boxes:
      lbox = [0] * 4
      for i in range(4):
        n = b[i] - self.zoom_window[i % 2] * self.img.shape[i % 2]
        n /= (self.zoom_window[2 + i % 2] - self.zoom_window[i % 2])
        lbox[i] = int(n / self.img.shape[i % 2] * self.img_shape[i % 2])
      self.draw_box(lbox, rimg)

    self.c_img = ImageTk.PhotoImage(Image.fromarray(rimg))
