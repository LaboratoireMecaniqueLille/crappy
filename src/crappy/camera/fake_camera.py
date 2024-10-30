# coding: utf-8

from time import time
from typing import Optional
import numpy as np
import logging

from .meta_camera import Camera


class FakeCamera(Camera):
  """This camera class generates images without requiring any actual camera or
  existing image file.

  The generated images are just a gradient of grey levels, with a line moving
  as a function of time. It is possible to tune the dimension of the image, the
  frame rate and the speed of the moving line.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Fake_camera* to *FakeCamera*
  """

  def __init__(self) -> None:
    """Initializes the parent class and instantiates the settings."""

    super().__init__()
    self._frame_nr = -1

    self.add_scale_setting('width', 1, 4096, None, self._gen_image, 1280, 1)
    self.add_scale_setting('height', 1, 4096, None, self._gen_image, 720, 1)
    self.add_scale_setting('speed', 0., 800., None, None, 400., 0.8)
    self.add_scale_setting('fps', 0.1, 100., None, None, 50., 0.1)

    self._t0 = time()
    self._t = self._t0

  def open(self,
           width: int = 1280,
           height: int = 720,
           speed: float = 100.,
           fps: float = 50.) -> None:
    """Sets the settings, initializes the first image and starts the time
    counter.

    Args:
      width: The width of the image to generate in pixels.
      height: The height of the image to generate in pixels.
      speed: The evolution speed of the image, in pixels per second.
      fps: The maximum update frequency of the generated images.
    
    .. versionadded:: 2.0.0
       *width*, *height*, *speed* and *fps* arguments explicitly listed
    """

    self.set_all(width=width, height=height, speed=speed, fps=fps)

    self._gen_image()

  def get_image(self) -> tuple[float, np.ndarray]:
    """Returns the updated image, depending only on the current timestamp.

    Also includes a waiting loop in order to achieve the right frame rate.
    """

    # Waiting in order to achieve the right frame rate
    while time() - self._t < 1 / self.fps:
      pass

    self._t = time()
    self._frame_nr += 1

    # Splitting the image to make a moving line
    row = int(self.speed * (self._t - self._t0)) % self.height
    return self._t, np.concatenate((self._img[row:], self._img[:row]), axis=0)

  def _gen_image(self, _: Optional[float] = None) -> None:
    """Generates the base gradient image, that will be split and returned
    in the :meth:`get_image` method."""

    self.log(logging.DEBUG, "Generating the image")
    self._img = np.arange(self.height) * 255. / self.height
    self._img = np.repeat(self._img.reshape(self.height, 1),
                          self.width, axis=1).astype(np.uint8)
