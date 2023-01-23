# coding: utf-8

from typing import Optional
from tkinter.messagebox import showerror
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

  def _bind_canvas_left_click(self) -> None:
    """"""

    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _extend_box(self, event: tk.Event) -> None:
    """"""

    super()._extend_box(event)

    if not self._select_box.no_points():
      min_x, max_x, min_y, max_y = self._select_box.sorted()
      if max_x - min_x >= 8 and max_y - min_y >= 8:
        self._spots.spot_1 = Box(
            min_x, min_x + (max_x - min_x) // 4,
            min_y + 3 * (max_y - min_y) // 8, min_y + 5 * (max_y - min_y) // 8)
        self._spots.spot_2 = Box(
            min_x + 3 * (max_x - min_x) // 4, max_x,
            min_y + 3 * (max_y - min_y) // 8, min_y + 5 * (max_y - min_y) // 8)
        self._spots.spot_3 = Box(
            min_x + 3 * (max_x - min_x) // 8, min_x + 5 * (max_x - min_x) // 8,
            min_y, min_y + (max_y - min_y) // 4)
        self._spots.spot_4 = Box(
            min_x + 3 * (max_x - min_x) // 8, min_x + 5 * (max_x - min_x) // 8,
            min_y + 3 * (max_y - min_y) // 4, max_y)

  def _stop_box(self, _: tk.Event) -> None:
    """"""

    # This box is not needed anymore
    self._select_box.reset()

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

    self.log(logging.WARNING, f"The patch {box} is outside the image, "
                              f"resetting the patches")
    self._spots.reset()

  def _stop(self) -> None:
    """"""

    if self._spots.empty():
      self.log(logging.WARNING, "No patches selected ! Not exiting the "
                                "configuration window")
      showerror('Error !',
                message="Please select patches before exiting the config "
                        "window !\nOr hit CTRL+C to exit Crappy")
      return

    super()._stop()
