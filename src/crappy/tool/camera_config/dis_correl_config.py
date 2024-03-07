# coding: utf-8

import tkinter as tk
from tkinter.messagebox import showerror
from typing import Optional
import logging
from multiprocessing.queues import Queue

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box
from ...camera.meta_camera import Camera
from ..._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class DISCorrelConfig(CameraConfigBoxes):
  """Class similar to :class:`~crappy.tool.camera_config.CameraConfig` but also 
  allowing to select the area on which the correlation will be performed.
  
  It relies on the :class:`~crappy.tool.camera_config.config_tools.Box` tool. 
  It is meant to be used for configuring the :class:`~crappy.blocks.DISCorrel` 
  Block.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *DISConfig* to *DISCorrelConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: Optional[int],
               max_freq: Optional[float],
               patch: Box) -> None:
    """Initializes the parent class and sets the correlation Box.

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
      patch: The :class:`~crappy.tool.camera_config.config_tools.Box` container
        that will save the information on the patch where to perform image
        correlation.

        .. versionadded:: 2.0.0
    """

    self._correl_box = patch
    self._draw_correl_box = True

    super().__init__(camera, log_queue, log_level, max_freq)

  @property
  def box(self) -> Box:
    """Returns the :class:`~crappy.tool.camera_config.config_tools.Box` object
    containing the region of interest.
    
    .. versionadded:: 1.5.10
    """

    return self._correl_box

  def finish(self) -> None:
    """Method called when the user tries to close the configuration window.

    Checks that a patch was selected on the image. If not, warns the user and
    prevents him from exiting except with CTRL+C.
    
    .. versionadded:: 2.0.0
    """

    if self.box.no_points():
      self.log(logging.WARNING, "No ROI selected ! Not exiting the "
                                "configuration window")
      showerror('Error !',
                message="Please select a ROI before exiting the config "
                        "window !\nOr hit CTRL+C to exit Crappy")
      return

    super().stop()

  def _set_bindings(self) -> None:
    """Binds the left mouse button click for drawing the box on which the
    correlation will be performed."""

    super()._set_bindings()

    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _start_box(self, event: tk.Event) -> None:
    """Simply saves the position of the user click, and disables the display of
    the current correl box."""

    super()._start_box(event)

    self._draw_correl_box = False

  def _stop_box(self, _: tk.Event) -> None:
    """Makes sure that the selected region is valid, sets it as the new correl
    box, and enables the display of the correl box."""

    self.log(logging.DEBUG, "Ending the selection box")

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
    self._correl_box.update(self._select_box)
    self._select_box.reset()

    self._draw_correl_box = True

  def _draw_overlay(self) -> None:
    """Draws the box to use for performing correlation on top of the last
    acquired image.

    Does not draw the correl box is the user is using the selection box.
    """

    if self._draw_correl_box:
      self._draw_box(self._correl_box)
    self._draw_box(self._select_box)

  def _handle_box_outside_img(self, _: Box) -> None:
    """If the correl box is outside the image, it means that the image size has
    been modified. Simply resetting the correl box then."""

    self._correl_box.reset()
