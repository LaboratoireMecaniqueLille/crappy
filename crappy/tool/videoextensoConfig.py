# coding: utf-8

import tkinter as tk
from typing import Optional
from .cameraConfigBoxes import Camera_config_with_boxes
from .cameraConfigTools import Box


class VE_config(Camera_config_with_boxes):
  """Class similar to :ref:`Camera configuration` but also displaying the
  bounding boxes of the detected spots, and allowing to select the area where
  to detect the spots by drawing a box with the left mouse button.

  It is meant to be used for configuring the :ref:`VideoExtenso` block.
  """

  def __init__(self, camera, video_extenso) -> None:
    """Sets the args and initializes the parent class.

    Args:
      camera: The camera object in charge of acquiring the images.
      video_extenso: The video extenso tool in charge of tracking the spots.
    """

    self._video_extenso = video_extenso
    super().__init__(camera)

  def _bind_canvas_left_click(self) -> None:
    """Binds the left mouse button click for drawing the box in which the spots
    will be searched."""

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
      spots = self._video_extenso.detect_spots(
          self._original_img[x_top: x_bottom, y_left: y_right], x_top, y_left)
      if spots is not None:
        self._spots = spots
    except IndexError:
      # Highly unlikely but always better to be careful
      self._spots.reset()
      return

    # This box is not needed anymore
    self._select_box.reset()

  def _save_l0(self) -> None:
    """Saves the original positions of the spots on the image."""

    if self._video_extenso.save_length():
      print(f"[VideoExtenso] Successfully saved L0 :\n"
            f"L0 x : {self._video_extenso.x_l0}\n"
            f"L0 y : {self._video_extenso.y_l0}")

  def _on_img_resize(self, _: Optional[tk.Event] = None) -> None:
    """Same as in the parent class except it also draws the patches and the
    select box on top of the displayed image."""

    self._draw_box(self._select_box)
    self._draw_spots()
    self._resize_img()
    self._display_img()
    self.update()

  def _update_img(self) -> None:
    """Same as in the parent class except it also draws the patches and the
    select box on top of the displayed image."""

    _, img = self._camera.get_image()
    if img is None:
      return

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
