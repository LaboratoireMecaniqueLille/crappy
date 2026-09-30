# coding: utf-8

"""Camera configuration state and interactions shared by GUI backends."""

import logging
import importlib.resources
from collections.abc import Callable
from io import BytesIO
from typing import Any
import numpy as np

from ..config_tools.zoom import Zoom
from .display_state import DisplayGeometry, DisplayState
from .selection_behavior import ConfigAction
from .setting_manager import SettingManager
from .configuration_lifecycle import ExceptionInfo
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import CameraSetting
from ...._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


class CameraConfigCore:
  """Own the camera model, image state, and toolkit-free user interactions.

  A backend initializes this class after creating its root window, then maps
  native events to the coordinate methods below. It renders ``_display_state``
  and ``_img`` using its own widgets. Resource and window lifetime remain the
  backend's responsibility.

  Args:
    camera: Camera providing settings and preview images.
    max_freq: Maximum preview acquisition frequency.
    transform: Optional transformation applied to acquired images.
  """

  def __init__(self,
               camera: Camera,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
    """Initialize ordinary state without constructing GUI controls."""

    self._camera: Camera = camera
    self._display_state: DisplayState = DisplayState()
    self._display_geometry: DisplayGeometry = DisplayGeometry()
    self._setting_manager: SettingManager = SettingManager(
        camera.settings, self._create_local_settings())

    self.shape: tuple[int, int] | tuple[int, int, int] | None = None
    self.dtype: str | None = None
    self._transform: Callable[[np.ndarray], np.ndarray] | None = transform
    self._img: np.ndarray | None = None
    self._original_img: np.ndarray | None = None
    self._low_thresh: float | None = None
    self._high_thresh: float | None = None
    self._move_x: float | None = None
    self._move_y: float | None = None
    self._max_freq: float | None = max_freq
    self._got_first_img: bool = False
    self._n_loops: int = 0

    self._zoom_values: Zoom = Zoom()
    self._zoom_ratio: float = 0.9
    self._zoom_step: int = 0
    self._max_zoom_step: int = 15

  def log(self,
          level: int,
          msg: str,
          *,
          exc_info: ExceptionInfo | None = None) -> None:
    """Send a message or explicit exception through the backend's logger.

    Args:
      level: Severity of an ordinary message.
      msg: Message to record.
      exc_info: Original exception tuple, if this is an exception record.
    """

    raise NotImplementedError

  def _create_local_settings(self) -> tuple[CameraSetting, ...]:
    """Create configurator-specific settings before building GUI controls.

    Subclasses can return local settings here. They are applied before camera
    settings and displayed before the sorted camera controls.
    """

    return tuple()

  def _extra_actions(self) -> tuple[ConfigAction, ...]:
    """Return semantic actions for the backend to present."""

    return tuple()

  def _validate_close(self) -> str | None:
    """Return a reason to keep the window open, or ``None`` when valid.

    A backend presents the reason to the user before attempting to close.
    """

    return None

  def _on_valid_close(self) -> None:
    """Finalize a validated configuration before the backend closes.

    Specialized configurations can save their selection or derived data here.
    """

    ...

  def get_config(self) -> tuple[Any, ...] | None:
    """Export values consumed by the paired image-processing CameraProcess.

    The owning Block calls this after the configuration window closes and
    before starting its processing Process. Subclasses return a tuple matching
    the positional parameters of the paired ``CameraProcess.set_config()``.

    Returns:
      Positional arguments for ``set_config()``, or ``None`` if no processing
      configuration is required.
    """

    return None

  def _zoom_at(self, x: int, y: int, direction: int) -> bool:
    """Zoom at a display position using a signed, toolkit-free direction."""

    if not direction or not self._is_on_image(x, y):
      return False

    self.log(logging.DEBUG, "Zooming on the image")
    next_step = min(max(self._zoom_step + direction, 0), self._max_zoom_step)
    if next_step == self._zoom_step:
      return False
    self._zoom_step = next_step
    self._display_state.zoom_percent = (100 * (1 / self._zoom_ratio) **
                                        self._zoom_step)

    if self._zoom_step == 0:
      self._zoom_values.reset()
      return True

    relative_x, relative_y = self._display_geometry.relative(x, y)
    geometry = self._display_geometry
    x_ratio = (relative_x * (self._zoom_values.x_high -
                             self._zoom_values.x_low) / geometry.image_width)
    y_ratio = (relative_y * (self._zoom_values.y_high -
                             self._zoom_values.y_low) / geometry.image_height)
    ratio = self._zoom_ratio if direction < 0 else 1 / self._zoom_ratio
    self._zoom_values.update_zoom(x_ratio, y_ratio, ratio)
    return True

  def _point_at(self, x: int, y: int) -> bool:
    """Update the reticle from plain display coordinates."""

    if not self._is_on_image(x, y):
      return False

    self.log(logging.DEBUG, "Updating the coordinates of the current pixel")
    (self._display_state.reticle_x,
     self._display_state.reticle_y) = self._coord_to_pix(x, y)
    self._update_pixel_value()
    return True

  def _update_pixel_value(self) -> None:
    """Read the original-image value at the current reticle position."""

    self.log(logging.DEBUG, "Updating the value of the current pixel")

    if self._original_img is None or not self._original_img.size:
      return

    try:
      self._display_state.reticle_value = int(np.average(
        self._original_img[self._display_state.reticle_y,
                           self._display_state.reticle_x]))
    except IndexError:
      self._display_state.reticle_x = 0
      self._display_state.reticle_y = 0
      self._display_state.reticle_value = int(np.average(
        self._original_img[0, 0]))

  def _coord_to_pix(self, x: int, y: int) -> tuple[int, int]:
    """Convert display coordinates to full-image pixel coordinates."""

    if self._img is None:
      return 0, 0

    img_height, img_width, *_ = self._img.shape
    return self._display_geometry.to_pixel(x, y, img_width, img_height,
                                           self._zoom_values)

  def _begin_pan(self, x: int, y: int) -> None:
    """Start panning from a display coordinate on the image."""

    self._move_x = None
    self._move_y = None
    if not self._is_on_image(x, y):
      return

    self.log(logging.DEBUG, "Drag started")
    self._move_x, self._move_y = self._display_geometry.relative(x, y)

  def _pan_to(self, x: int, y: int) -> None:
    """Pan the image to a display coordinate after a valid press."""

    if (self._move_x is None or self._move_y is None or
        not self._is_on_image(x, y)):
      return

    self.log(logging.DEBUG, "Dragging the image")

    geometry = self._display_geometry
    zoom_x_low, zoom_x_high = self._zoom_values.x_low, self._zoom_values.x_high
    zoom_y_low, zoom_y_high = self._zoom_values.y_low, self._zoom_values.y_high
    relative_x, relative_y = geometry.relative(x, y)
    delta_x_disp = self._move_x - relative_x
    delta_y_disp = self._move_y - relative_y
    delta_x = delta_x_disp * (zoom_x_high - zoom_x_low) / geometry.image_width
    delta_y = delta_y_disp * (zoom_y_high - zoom_y_low) / geometry.image_height
    self._zoom_values.update_move(delta_x, delta_y)
    self._move_x, self._move_y = relative_x, relative_y

  def _is_on_image(self, x: int, y: int) -> bool:
    """Check image hit-testing from plain display coordinates."""

    return self._display_geometry.contains(x, y)

  def _acquire_image(self) -> bool:
    """Acquire, transform, and convert one frame for a backend to render.

    Returns:
      Whether a new preview image is available. After the first image, a
      camera returning no frame leaves the previous preview unchanged.
    """

    ret = self._camera.get_image()
    no_img = ret is None

    if no_img:
      if not self._got_first_img:
        self.log(logging.WARNING, "Could not get an image from the camera, "
                                  "displaying an error image instead")
        no_img_path = importlib.resources.files('crappy').joinpath(
            'tool/data/no_image.png')
        ret = None, np.array(Image.open(BytesIO(no_img_path.read_bytes())))
      else:
        self.log(logging.DEBUG, "No image returned by the camera")
        return False

    if ret is None:
      raise RuntimeError("The returned metadata and image shouldn't be None "
                         "at that point")

    self._got_first_img = True
    self._n_loops += 1
    _, img = ret

    if not no_img and self._transform is not None:
      img = self._transform(img)

    if not no_img and img.dtype.name != self.dtype:
      self.dtype = img.dtype.name
    if not no_img and img.shape != self.shape:
      self.shape = img.shape

    self._cast_img(img)
    return True

  def _cast_img(self, img: np.ndarray) -> None:
    """Convert a camera image to 8 bits and update ordinary display state."""

    if len(img.shape) not in (2, 3):
      raise ValueError(f"Cannot handle images of shape {img.shape} !")

    if len(img.shape) == 3 and img.shape[2] > 4:
      raise ValueError(f"Cannot handle images of shape {img.shape} !")

    if len(img.shape) == 3 and img.shape[2] == 1:
      img = img[:, :, 0]
    if len(img.shape) == 3 and img.shape[2] == 2:
      img = img[:, :, 0]
    if len(img.shape) == 3 and img.shape[2] == 4:
      img = img[:, :, :3]
    if len(img.shape) == 3:
      img = img[:, :, ::-1]

    if self._display_state.auto_range:
      self.log(logging.DEBUG, "Applying auto range to the image")
      low_thresh, high_thresh = map(float, np.percentile(img, (3, 97)))
      self._low_thresh, self._high_thresh = low_thresh, high_thresh
      self._img = ((np.clip(img, low_thresh, high_thresh) - low_thresh) * 255 /
                   (high_thresh - low_thresh)).astype('uint8')
      bit_depth = int(np.ceil(np.log2(int(np.max(img)) + 1)))
      self._original_img = (img / 2 ** (bit_depth - 8)).astype('uint8')
    elif img.dtype != np.uint8:
      self.log(logging.DEBUG, "Casting the image to 8 bits")
      bit_depth = int(np.ceil(np.log2(int(np.max(img)) + 1)))
      self._img = (img / 2 ** (bit_depth - 8)).astype('uint8')
      self._original_img = np.copy(self._img)
    else:
      self._img = img
      self._original_img = np.copy(img)

    self._display_state.detected_bits = int(np.ceil(np.log2(int(np.max(img))
                                                            + 1)))
    self._display_state.max_pixel = int(np.max(img))
    self._display_state.min_pixel = int(np.min(img))
