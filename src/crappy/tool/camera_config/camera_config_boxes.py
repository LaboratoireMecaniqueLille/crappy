# coding: utf-8

import numpy as np
import tkinter as tk
import logging
from typing import Optional
from multiprocessing.queues import Queue

from .camera_config import CameraConfig
from .config_tools import Box, SpotsBoxes
from ...camera.meta_camera import Camera


class CameraConfigBoxes(CameraConfig):
  """This class is a basis for the configuration GUIs featuring boxes to
  display or to draw.
  
  It is a child of the base :class:`~crappy.tool.camera_config.CameraConfig`,
  and relies on the :class:`~crappy.tool.camera_config.config_tools.Box` and
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` tools. It
  implements useful methods for drawing one or several Boxes. If instantiated,
  this class behaves the exact same way as its parent class. It is not used as
  is by any Block in Crappy.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0
     renamed from *Camera_config_with_boxes* to *CameraConfigBoxes*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: Optional[int],
               max_freq: Optional[float]) -> None:
    """Initializes the parent class and sets the spots container.

    Args:
      camera: The :class:`~crappy.camera.Camera` object in charge of acquiring 
        the images.
      log_queue: A :obj:`multiprocessing.Queue` for sending the log messages to 
        the main :obj:`~logging.Logger`, only used in Windows.

        .. versionadded:: 2.0.0
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.

        .. versionadded:: 2.0.0
      max_freq: The maximum frequency this window is allowed to loop at. It is
        simply the ``freq`` attribute of the :class:`~crappy.blocks.Camera`
        Block.

        .. versionadded:: 2.0.0
    """

    self._spots = SpotsBoxes()
    self._select_box = Box()
    super().__init__(camera, log_queue, log_level, max_freq)

  def _draw_box(self, box: Box) -> None:
    """Draws one line of the box after the other, making sure they fit in the
    image."""

    if self._img is None or box.no_points():
      return

    self.log(logging.DEBUG, f"Drawing the box: {box}")

    # Determining the number of lines to draw
    x_top, x_bottom, y_left, y_right = box.sorted()
    canvas_width = self._img_canvas.winfo_width()
    canvas_height = self._img_canvas.winfo_height()
    max_fact = max(self._img.shape[0] // canvas_height,
                   self._img.shape[1] // canvas_width, 1)

    try:
      for line in (line for i in range(max_fact) for line in
                   ((box.y_start + i, slice(x_top, x_bottom)),
                    (box.y_end - i, slice(x_top, x_bottom)),
                    (slice(y_left, y_right), x_top + i),
                    (slice(y_left, y_right), x_bottom - i))):
        if np.size(self._original_img[line]) > 0:
          self._img[line] = 255 * int(np.mean(self._img[line]) < 128)
    except IndexError:
      self._handle_box_outside_img(box)
      return

  def _handle_box_outside_img(self, _: Box) -> None:
    """This method is meant to simplify the customization of the action to
    perform when a patch is outside the image in subclasses."""

    ...

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
