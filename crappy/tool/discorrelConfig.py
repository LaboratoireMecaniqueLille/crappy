# coding: utf-8

"""More documentation coming soon !"""

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


class DISConfig(Camera_config):
  def __init__(self, camera):
    self.box = (-1, -1, -1, -1)
    Camera_config.__init__(self, camera)

  def clamp(self, t):
    if isinstance(t[0], slice):
      return t[0], min(max(0, t[1]), self.img_shape[1] - 1)
    else:
      return min(max(0, t[0]), self.img_shape[0] - 1), t[1]

  def create_window(self):
    Camera_config.create_window(self)
    self.img_label.bind('<1>', self.start_select)
    self.img_label.bind('<B1-Motion>', self.update_box)
    self.img_label.bind('<ButtonRelease-1>', self.stop_select)

  def start_select(self, event):
    self.box_origin = self.get_img_coord(event.y, event.x)

  def update_box(self, event):
    oy, ox = self.box_origin
    y, x = self.get_img_coord(event.y, event.x)
    self.box = (min(oy, y), min(ox, x), max(oy, y), max(ox, x))
    #print(self.box)

  def stop_select(self, event):
    pass

  def draw_box(self, box, img):
    for s in [(box[0], slice(box[1], box[3])),
              (box[2], slice(box[1], box[3])),
              (slice(box[0], box[2]), box[1]),
              (slice(box[0], box[2]), box[3])]:
      # Turn these pixels white or black for highest possible contrast
      s = self.clamp(s)
      img[s] = 255 * int(np.mean(img[s]) < 128)

  def resize_img(self, sl):
    rimg = cv2.resize(self.img8[sl[1], sl[0]], tuple(reversed(self.img_shape)),
        interpolation=0)
    if self.box[0] > 0:
      lbox = [0] * 4
      for i in range(4):
        n = self.box[i] - self.zoom_window[i % 2] * self.img.shape[i % 2]
        n /= (self.zoom_window[2 + i % 2] - self.zoom_window[i % 2])
        lbox[i] = int(n / self.img.shape[i % 2] *
                      self.img_shape[i % 2])
      self.draw_box(lbox, rimg)

    self.c_img = ImageTk.PhotoImage(Image.fromarray(rimg))
