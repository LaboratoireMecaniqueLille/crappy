# coding: utf-8

"""
Demonstration of how to create a custom Camera in Crappy.

This camera is intended to be used as a template, it doesn't actually display
any image from a real camera.

No hardware required.
"""

import crappy
import numpy as np
from time import time

# This class can be used as a starting point to create a new Camera.
# To add it to crappy, make the imports relative (refer to any other camera),
# move the class to a file in crappy/camera and add the corresponding line
# in crappy/camera/__init__.py


class My_cam(crappy.camera.Camera):
  """A basic example of Camera.

  It will simply send a gray image, the level can be set using a Camera_setting.
  """

  def __init__(self, resolution=(480, 640)):
    # Do not forget to init Camera !
    super().__init__()
    self.resolution = resolution
    self.frame = None

    # Optional: add settings to the camera
    # Here we will set the gray level of the image
    self.add_setting('level', self._get_lvl, self._set_lvl, limits=(0, 255))

  def _get_lvl(self):
    """The gray level getter."""

    return self.frame[0, 0]

  def _set_lvl(self, val):
    """The gray level setter.

    Recreates the image with the new level.
    """

    self.frame = np.ones(self.resolution, dtype=np.uint8) * val
    return val

  def open(self, **kwargs):
    """Will be called in the .prepare() of the block."""

    # Let's create our image
    self.frame = np.zeros(self.resolution, dtype=np.uint8)
    # Allow delegation of generic camera args such as max_fps
    self.set_all(**kwargs)

  def get_image(self):
    """The method that returns the frame. It must also return the time."""

    return time(), self.frame

  def close(self):
    """Will be called on exit or crash."""

    del self.frame


if __name__ == '__main__':
  cam = crappy.blocks.Camera('My_cam', max_fps=60, no_loop=True)

  crappy.start()
