# coding: utf-8

from typing import Optional
from tkinter.messagebox import showerror
import tkinter as tk
import logging
from multiprocessing.queues import Queue

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box, SpotsBoxes
from ...camera.meta_camera import Camera
from ...camera.meta_camera.camera_setting import CameraScaleSetting
from ..._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class DICVEConfig(CameraConfigBoxes):
  """Class similar to :class:`~crappy.tool.camera_config.CameraConfig` but also 
  displaying the bounding boxes of the regions defined as patches.
  
  It relies on the :class:`~crappy.tool.camera_config.config_tools.Box` and 
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` tools. It is
  meant to be used for configuring the :class:`~crappy.blocks.DICVE` Block.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *DISVE_config* to *DICVEConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: Optional[int],
               max_freq: Optional[float],
               patches: SpotsBoxes) -> None:
    """Sets the patches and initializes the parent class.

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
      patches: An instance of
        :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` containing
        the patches to follow for image correlation.
    """

    self._patch_size: Optional[CameraScaleSetting] = None

    super().__init__(camera, log_queue, log_level, max_freq)

    # Setting the patches
    self._spots = patches

  def finish(self) -> None:
    """Method called when the user tries to close the configuration window.

    Check that patches were selected on the image. If not, warns the user and
    prevents him from exiting except with CTRL+C. If not already done by the
    user, also saves the initial length between the patches.
    
    .. versionadded:: 2.0.0
    """

    if self._spots.empty():
      self.log(logging.WARNING, "No patches selected ! Not exiting the "
                                "configuration window")
      showerror('Error !',
                message="Please select patches before exiting the config "
                        "window !\nOr hit CTRL+C to exit Crappy")
      return

    self._spots.save_length()
    self.log(logging.INFO,
             f"Successfully saved L0 ! L0 x : {self._spots.x_l0}, "
             f"L0 y : {self._spots.y_l0}")

    super().stop()

  def _add_settings(self) -> None:
    """Same as in the parent class except it also adds a Path size setting to
    the list of possible settings."""

    self._patch_size = CameraScaleSetting("Patch size (px)", 2, 1024,
                                          default=128)
    self._add_slider_setting(self._patch_size)

    super()._add_settings()

  def _update_settings(self) -> None:
    """Same as in the parent class except it also updates the Path size setting
    in addition to all the other settings."""

    if self._patch_size.value != self._patch_size.tk_var.get():
      self._patch_size.value = self._patch_size.tk_var.get()
    self._patch_size.tk_var.set(self._patch_size.value)

    super()._update_settings()

  def _set_bindings(self) -> None:
    """Binds the left mouse button click to drawing the patches on which to
    perform the image correlation."""

    super()._set_bindings()

    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _extend_box(self, event: tk.Event) -> None:
    """When the user drags the selection box, updating the four patches being
    drawn."""

    super()._extend_box(event)

    if not self._select_box.no_points() and self._patch_size is not None:
      min_x, max_x, min_y, max_y = self._select_box.sorted()
      size = self._patch_size.value
      if max_x - min_x >= 3 * size and max_y - min_y >= 3 * size:
        self._spots.spot_1 = Box(min_x, min_x + size,
                                 (min_y + max_y - size) // 2,
                                 (min_y + max_y + size) // 2)
        self._spots.spot_2 = Box(max_x - size, max_x,
                                 (min_y + max_y - size) // 2,
                                 (min_y + max_y + size) // 2)
        self._spots.spot_3 = Box((min_x + max_x - size) // 2,
                                 (min_x + max_x + size) // 2,
                                 min_y, min_y + size)
        self._spots.spot_4 = Box((min_x + max_x - size) // 2,
                                 (min_x + max_x + size) // 2,
                                 max_y - size, max_y)

  def _stop_box(self, _: tk.Event) -> None:
    """Simply resets the selection box."""

    # This box is not needed anymore
    self._select_box.reset()

  def _draw_overlay(self) -> None:
    """Draws the detected spots to track on top of the last acquired image."""

    self._draw_spots()

  def _handle_box_outside_img(self, box: Box) -> None:
    """If a patch is outside the image, warning the user and resetting the
    patches."""

    self.log(logging.WARNING, f"The patch {box} is outside the image, "
                              f"resetting the patches")
    self._spots.reset()
