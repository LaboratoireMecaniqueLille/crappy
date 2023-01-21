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


class MyCam(crappy.Camera):
  """A basic example of Camera.

  It will simply send a gray image, the level can be set using a
  Camera_setting.
  """

  def __init__(self, resolution=(480, 640)):
    # Do not forget to init Camera !
    super().__init__()
    self._resolution = resolution
    self._frame = None

    # Optional: add settings to the camera
    # Here we will set the gray level of the image
    self.add_scale_setting('level', 0, 255, self._get_lvl, self._set_lvl, 128)

  def _get_lvl(self):
    """The gray level getter."""

    return self._frame[0, 0]

  def _set_lvl(self, val):
    """The gray level setter.

    Recreates the image with the new level.
    """

    self._frame = np.ones(self._resolution, dtype=np.uint8) * val
    return val

  def open(self, **kwargs):
    """Will be called in the .prepare() of the block."""

    # Let's create our image
    self._frame = np.zeros(self._resolution, dtype=np.uint8)
    # Allow delegation of generic camera args such as max_fps
    self.set_all(**kwargs)

  def get_image(self):
    """The method that returns the frame. It must also return the time."""

    return time(), self._frame

  def close(self):
    """Will be called on exit or crash."""

    del self._frame


if __name__ == '__main__':

  # Instantiating the custom Camera
  cam = crappy.blocks.Camera('MyCam',
                             freq=60,
                             display_images=True,
                             display_freq=True)

  # Starting the test
  crappy.start()
