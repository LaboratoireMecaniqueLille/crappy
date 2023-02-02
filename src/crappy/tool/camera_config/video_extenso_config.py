# coding: utf-8

import tkinter as tk
from tkinter.messagebox import showerror
from typing import Optional
import numpy as np
from io import BytesIO
from pkg_resources import resource_string
from time import sleep
import logging

from .camera_config_boxes import CameraConfigBoxes
from .config_tools import Box, SpotsDetector
from ...camera.meta_camera import Camera
from ..._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class VideoExtensoConfig(CameraConfigBoxes):
  """Class similar to :ref:`Camera Configurator` but also displaying the
  bounding boxes of the detected spots, and allowing to select the area where
  to detect the spots by drawing a box with the left mouse button.

  It is meant to be used for configuring the :ref:`Video Extenso` block.
  """

  def __init__(self, camera: Camera, detector: SpotsDetector) -> None:
    """Sets the args and initializes the parent class.

    Args:
      camera: The camera object in charge of acquiring the images.
    """

    super().__init__(camera)
    self._detector = detector
    self._spots = detector.spots

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

    self._update_button = tk.Button(self._sets_frame, text="Apply Settings",
                                    command=self._update_settings)
    self._update_button.pack(expand=False, fill='none', ipadx=5, ipady=5,
                             padx=5, pady=5, anchor='n', side='top')

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

  def _on_img_resize(self, _: Optional[tk.Event] = None) -> None:
    """Same as in the parent class except it also draws the patches and the
    select box on top of the displayed image."""

    self.log(logging.DEBUG, "The image canvas was resized")

    self._draw_box(self._select_box)
    self._draw_spots()
    self._resize_img()
    self._display_img()
    self.update()

  def _update_img(self, init: bool = False) -> None:
    """Same as in the parent class except it also draws the patches and the
    select box on top of the displayed image.

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
    self._draw_box(self._select_box)
    self._draw_spots()
    self._resize_img()

    self._calc_hist()
    self._resize_hist()

    self._display_img()
    self._display_hist()

    self._update_pixel_value()

    self.update()

  def _handle_box_outside_img(self, _: Box) -> None:
    """If a patch is outside the image, it means that the image size has been
    modified. Simply resetting the spots then."""

    self._spots.reset()

  def _stop(self) -> None:
    """"""

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

    super()._stop()
