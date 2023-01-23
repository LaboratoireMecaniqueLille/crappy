# coding: utf-8

from typing import Optional
import tkinter as tk
import numpy as np
from io import BytesIO
from pkg_resources import resource_string
from time import sleep
import logging

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box, SpotsBoxes
from ...camera.meta_camera import Camera
from ..._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class DICVEConfig(CameraConfigBoxes):
  """Class similar to :ref:`Camera Configurator` but also displaying the
  bounding boxes of the regions defined as patches.

  It is meant to be used for configuring the :ref:`DIS VE` block.
  """

  def __init__(self, camera: Camera, patches: SpotsBoxes) -> None:
    """Sets the patches and initializes the parent class.

    Args:
      camera: The camera object in charge of acquiring the images.
      patches: The patches to follow for image correlation.
    """

    super().__init__(camera)

    # Setting the patches
    self._spots = patches

  def _on_img_resize(self, _: Optional[tk.Event] = None) -> None:
    """Same as in the parent class except it also draws the patches on top of
    the displayed image."""

    self.log(logging.DEBUG, "The image canvas was resized")

    self._draw_spots()
    self._resize_img()
    self._display_img()
    self.update()

  def _update_img(self, init: bool = False) -> None:
    """Same as in the parent class except it also draws the patches on top of
    the displayed image.

    Args:
      init: If :obj:`True`, means that the method is called during
        :meth:`__init__` and if the image cannot be obtained it should be
        replaced with a dummy one.
    """

    self.log(logging.DEBUG, "Updating the image")

    ret = self._camera.get_image()

    # If no frame could be grabbed from the camera
    if ret is None:
      # If it's the first call, generate error image to initialize the window
      if init:
        self.log(logging.WARNING, "Could not get an image from the camera, "
                                  "displaying an error image instead")
        ret = None, np.array(Image.open(BytesIO(resource_string(
          'crappy', 'tool/data/no_image.png'))))
      # Otherwise, just pass
      else:
        self.log(logging.DEBUG, "No image returned by the camera")
        self.update()
        sleep(0.001)
        return

    self._n_loops += 1
    _, img = ret

    if img.dtype != self.dtype:
      self.dtype = img.dtype
    if self.shape != img.shape:
      self.shape = img.shape

    self._cast_img(img)
    self._draw_spots()
    self._resize_img()

    self._calc_hist()
    self._resize_hist()

    self._display_img()
    self._display_hist()

    self._update_pixel_value()

    self.update()

  def _handle_box_outside_img(self, box: Box) -> None:
    """If a patch is outside the image, maybe the user entered a wrong value or
    the image size has been modified. Raising an exception as the DICVE can't
    run in this situation."""

    raise ValueError(f'The patch {box.get_patch()} does not fit in the '
                     f'image !')
