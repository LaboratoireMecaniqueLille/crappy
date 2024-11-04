# coding: utf-8

import tkinter as tk
from tkinter.messagebox import showerror
from typing import Optional
import logging
from multiprocessing.queues import Queue

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box, SpotsDetector
from ...camera.meta_camera import Camera
from ..._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class VideoExtensoConfig(CameraConfigBoxes):
  """Class similar to :class:`~crappy.tool.camera_config.CameraConfig` but also
  displaying the bounding boxes of the detected spots, and allowing to select
  the area where to detect the spots by drawing a box with the left mouse
  button.

  It relies on the :class:`~crappy.tool.camera_config.config_tools.Box` and
  :class:`~crappy.tool.camera_config.config_tools.SpotsDetector` tools. It is
  meant to be used for configuring the :class:`~crappy.blocks.VideoExtenso`
  Block.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *VE_config* to *VideoExtensoConfig*
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: Optional[int],
               max_freq: Optional[float],
               detector: SpotsDetector) -> None:
    """Sets the args and initializes the parent class.

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
      detector: An instance of
        :class:`~crappy.tool.camera_config.config_tools.SpotsDetector` used for
        detecting spots on the images received from the
        :class:`~crappy.camera.Camera`.
        
        .. versionadded:: 2.0.0

    .. versionchanged:: 1.5.10 renamed *ve* argument to *video_extenso*
    .. versionremoved:: 2.0.0 *video_extenso* argument
    """

    super().__init__(camera, log_queue, log_level, max_freq)
    self._detector = detector
    self._spots = detector.spots

  def finish(self) -> None:
    """Method called when the user tries to close the configuration window.

    Checks that spots were detected on the image. If not, warns the user and
    prevents him from exiting except with CTRL+C. Also, saves the initial
    length if not already done by the user.
    
    .. versionadded:: 2.0.0
    """

    if self._detector.spots.empty():
      self.log(logging.WARNING, "No spots were selected ! Not exiting the "
                                "configuration window")
      showerror("Error !",
                message="Please select spots before exiting the config "
                        "window !\nOr hit CTRL+C to exit Crappy")
      return

    if self._detector.spots.x_l0 is None or self._detector.spots.y_l0 is None:
      self._detector.spots.save_length()
      self.log(logging.INFO,
               f"Successfully saved L0 ! L0 x : {self._detector.spots.x_l0}, "
               f"L0 y : {self._detector.spots.y_l0}")

    super().stop()

  def _set_bindings(self) -> None:
    """Binds the left mouse button click for drawing the box in which the spots
    will be searched."""

    super()._set_bindings()

    self._img_canvas.bind('<ButtonPress-1>', self._start_box)
    self._img_canvas.bind('<B1-Motion>', self._extend_box)
    self._img_canvas.bind('<ButtonRelease-1>', self._stop_box)

  def _create_buttons(self) -> None:
    """Compared with the parent class, creates an extra button for saving the
    original position of the spots."""

    super()._create_buttons()

    self._update_button = tk.Button(self._sets_frame, text="Save L0",
                                    command=self._save_l0)
    self._update_button.pack(expand=False, fill='none', ipadx=5, ipady=5,
                             padx=5, pady=5, anchor='n', side='top')

  def _stop_box(self, _: tk.Event) -> None:
    """When the user releases the mouse, searches for spots in the selected
    area and displays them if any were found."""

    # If it's just a regular click with no dragging, do nothing
    if self._img is None or self._select_box.no_points():
      self._select_box.reset()
      return

    # The sides need to be sorted before slicing numpy array
    y_left, y_right, x_top, x_bottom = self._select_box.sorted()

    # If the box is flat, resetting it
    if y_left == y_right or x_top == x_bottom:
      self._select_box.reset()
      return

    # Now actually trying to detect the spots
    try:
      self._detector.detect_spots(self._original_img[x_top: x_bottom,
                                                     y_left: y_right],
                                  x_top, y_left)
    except IndexError:
      # Highly unlikely but always better to be careful
      self._detector.spots.reset()
      return

    # This box is not needed anymore
    self._select_box.reset()

  def _save_l0(self) -> None:
    """Saves the original positions of the spots on the image."""

    if self._detector.spots.empty():
      self.log(logging.WARNING, "Cannot save L0, there are no spots !")
    else:
      self._detector.spots.save_length()
      self.log(logging.INFO,
               f"Successfully saved L0 ! L0 x : {self._detector.spots.x_l0}, "
               f"L0 y : {self._detector.spots.y_l0}")

  def _draw_overlay(self) -> None:
    """Draws the detected spots to track on top of the last acquired image.

    Also draws the selection box if the user is currently drawing one.
    """

    self._draw_box(self._select_box)
    self._draw_spots()

  def _handle_box_outside_img(self, _: Box) -> None:
    """If a patch is outside the image, it means that the image size has been
    modified. Simply resetting the spots then."""

    self._spots.reset()
