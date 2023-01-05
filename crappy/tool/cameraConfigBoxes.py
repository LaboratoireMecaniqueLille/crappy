# coding: utf-8

import numpy as np
import tkinter as tk
import logging

from .cameraConfig import Camera_config
from .cameraConfigTools import Box, Spot_boxes


class Camera_config_with_boxes(Camera_config):
  """This class is a basis for the configuration GUIs featuring boxes to
  display or to draw.

  It implements useful methods for drawing the boxes. If instantiated, this
  class behaves the exact same way as its parent class. It is not used as is by
  any block in Crappy.
  """

  def __init__(self, camera) -> None:
    """Initializes the parent class and sets the spots container."""

    self._spots = Spot_boxes()
    self._select_box = Box()
    super().__init__(camera)

  def _draw_box(self, box: Box) -> None:
    """Draws one line of the box after the other, making sure they fit in the
    image."""

    if self._img is None or box.no_points():
      return

    self.log(logging.DEBUG, f"Drawing the box: {box}")

    # The sides need to be sorted before slicing numpy array
    y_left, y_right, x_top, x_bottom = box.sorted()

    # Drawing one line after the other
    for slice_ in ((slice(x_top, x_bottom), y_left),
                   (slice(x_top, x_bottom), y_right),
                   (x_top, slice(y_left, y_right)),
                   (x_bottom, slice(y_left, y_right))):
      try:
        # The color of the line is adjusted according to the background
        # The original image must be used as no lines are already drawn on it
        if np.size(self._original_img[slice_]) > 0:
          self._img[slice_] = 255 * np.rint(np.mean(self._img[slice_] < 128))
      except IndexError:
        self._handle_box_outside_img(box)
        return

  def _handle_box_outside_img(self, _: Box) -> None:
    """This method is meant to simplify the customization of the action to
    perform when a patch is outside the image in subclasses."""

    pass

  def _draw_spots(self) -> None:
    """Simply draws every spot on top of the image."""

    if self._img is None:
      return

    for spot in self._spots:
      if spot is not None:
        self._draw_box(spot)

  def _start_box(self, event: tk.Event) -> None:
    """Simply saves the position of the user click."""

    self.log(logging.DEBUG, "Starting the selection box")

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self._select_box.x_start, \
        self._select_box.y_start = self._coord_to_pix(event.x, event.y)

  def _extend_box(self, event: tk.Event) -> None:
    """Draws a box as the user drags the mouse while maintaining the left
    button clicked."""

    self.log(logging.DEBUG, "Extending the selection box")

    # If the mouse is on the canvas but not on the image, do nothing
    if not self._check_event_pos(event):
      return

    self._select_box.x_end, \
        self._select_box.y_end = self._coord_to_pix(event.x, event.y)
