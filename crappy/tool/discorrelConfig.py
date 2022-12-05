# coding: utf-8

import tkinter as tk
from typing import Optional
from copy import deepcopy
import numpy as np
from io import BytesIO
from pkg_resources import resource_string
from time import sleep
from .cameraConfigBoxes import Camera_config_with_boxes
from .cameraConfigTools import Box
from .._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class DISConfig(Camera_config_with_boxes):
  """Class similar to :ref:`Camera configuration` but also allowing to select
  the area on which the correlation will be performed.

  It is meant to be used for configuring the :ref:`Discorrel` block.
  """

  def __init__(self, camera) -> None:
    """Initializes the parent class and sets the correl box."""

    self._correl_box = Box()
    self._draw_correl_box = True

    super().__init__(camera)

  @property
  def box(self) -> Box:
    """Returns the Box object containing the region of interest."""

    return self._correl_box

  def _bind_canvas_left_click(self) -> None:
    """Binds the left mouse button click for drawing the box on which the
    correlation will be performed."""

    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _start_box(self, event: tk.Event) -> None:
    """Simply saves the position of the user click, and disables the display of
    the current correl box."""

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self._select_box.x_start, \
        self._select_box.y_start = self._coord_to_pix(event.x, event.y)

    self._draw_correl_box = False

  def _stop_box(self, _: tk.Event) -> None:
    """Makes sure that the selected region is valid, sets it as the new correl
    box, and enables the display of the correl box."""

    # If it's just a regular click with no dragging, do nothing
    if self._img is None or self._select_box.no_points():
      self._select_box.reset()
      self._draw_correl_box = True
      return

    # The sides need to be sorted before slicing numpy array
    y_left, y_right, x_top, x_bottom = self._select_box.sorted()

    # If the box is flat, resetting it
    if y_left == y_right or x_top == x_bottom:
      self._select_box.reset()
      self._draw_correl_box = True
      return

    # The new correl box is just the copy of the select box
    self._correl_box = deepcopy(self._select_box)
    self._select_box.reset()

    self._draw_correl_box = True

  def _on_img_resize(self, _: Optional[tk.Event] = None) -> None:
    """Same as in the parent class except it also draws the select box on top
    of the displayed image."""

    # Do not draw the correl box if the user is creating the select box
    if self._draw_correl_box:
      self._draw_box(self._correl_box)
    self._draw_box(self._select_box)

    self._resize_img()
    self._display_img()
    self.update()

  def _update_img(self, init: bool = False) -> None:
    """Same as in the parent class except it also draws the select box on top
    of the displayed image.

    Args:
      init: If :obj:`True`, means that the method is called during
        :meth:`__init__` and if the image cannot be obtained it should be
        replaced with a dummy one.
    """

    ret = self._camera.get_image()

    # If no frame could be grabbed from the camera
    if ret is None:
      # If it's the first call, generate error image to initialize the window
      if init:
        ret = None, np.array(Image.open(BytesIO(resource_string(
          'crappy', 'tool/data/no_image.png'))))
      # Otherwise, just pass
      else:
        self.update()
        sleep(0.001)
        return

    _, img = ret

    if img.dtype != self.dtype:
      self.dtype = img.dtype
    if self.shape != img.shape:
      self.shape = img.shape

    self._cast_img(img)
    # Do not draw the correl box if the user is creating the select box
    if self._draw_correl_box:
      self._draw_box(self._correl_box)
    self._draw_box(self._select_box)
    self._resize_img()

    self._calc_hist()
    self._resize_hist()

    self._display_img()
    self._display_hist()

    self._update_pixel_value()

    self.update()

  def _handle_box_outside_img(self, _: Box) -> None:
    """If the correl box is outside the image, it means that the image size has
    been modified. Simply resetting the correl box then."""

    self._correl_box.reset()
