# coding: utf-8

""":class:`~crappy.camera.meta_camera.camera.Camera` configuration state and
interactions shared by GUI backends."""

import logging
import importlib.resources
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from multiprocessing import current_process
from multiprocessing.queues import Queue
from typing import Any
import numpy as np

from ..config_tools.zoom import Zoom
from ._display_state import DisplayGeometry, DisplayState
from ._setting_manager import SettingManager
from ._configuration_lifecycle import ExceptionInfo
from ....camera.meta_camera import Camera
from ....camera.meta_camera.camera_setting import CameraSetting
from ...._global import OptionalModule

try:
  from PIL import Image
except (ModuleNotFoundError, ImportError):
  Image = OptionalModule("pillow")


@dataclass(frozen=True)
class ConfigAction:
  """A semantic action that a GUI backend can present as a button.

  Attributes:
    id: Stable key used to find the action's backend control.
    label: Text displayed on the control.
    callback: Operation to run when the action is activated.
  """

  id: str
  label: str
  callback: Callable[[], None]


class CameraConfig(ABC):
  """Abstract camera configuration shared by the Tkinter and PyQt6 backends.

  This class owns the :class:`~crappy.camera.meta_camera.camera.Camera`
  reference, setting models, preview indicators, image conversion, zoom, and
  pointer coordinates. It does not create widgets or start a histogram worker.
  Subclasses implement :meth:`run() <crappy.tool.camera_config.base.\
camera_config.CameraConfig.run>`, :meth:`stop() <crappy.tool.camera_config.\
base.camera_config.CameraConfig.stop>`, and :meth:`watch_shutdown() <crappy.\
tool.camera_config.base.camera_config.CameraConfig.watch_shutdown>` in a GUI.

  Attributes:
    shape: Shape of the latest acquired image after transform, before preview
      conversion. :obj:`None` until a
      :class:`~crappy.camera.meta_camera.camera.Camera` image has been
      acquired.
    dtype: NumPy dtype of that image, or :obj:`None` before acquisition.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               camera: Camera,
               log_queue: Queue,
               log_level: int | None,
               max_freq: float | None,
               transform: Callable[[np.ndarray], np.ndarray] | None,
               *_,
               **__) -> None:
    """Initializes the shared models without starting acquisition.

    Args:
      camera: Open :class:`~crappy.camera.meta_camera.camera.Camera` object
        providing preview images and adjustable settings.
      log_queue: Crappy logging queue, forwarded to the histogram worker.
      log_level: Script logging level, or :obj:`None` to disable worker
        logging. The window uses the logger configured by its owning
        :class:`~crappy.blocks.meta_block.block.Block`.
      max_freq: Maximum preview acquisition rate in hertz. :obj:`None` removes
        this limit, but acquisition and rendering may reduce the achieved rate.
      transform: :obj:`~collections.abc.Callable` applied to acquired images
        before preview conversion and image-format reporting, or :obj:`None` to
        leave them unchanged.
    """

    # Logging must be available during GUI and local settings initialization
    self._logger: logging.Logger | None = None
    self._log_queue: Queue = log_queue
    self._log_level: int | None = log_level

    # Continue through the MRO so the selected GUI backend creates its window
    super().__init__()

    self._camera: Camera = camera

    # Objects in charge of managing the display and settings
    self._display_state: DisplayState = DisplayState()
    self._display_geometry: DisplayGeometry = DisplayGeometry()
    self._setting_manager: SettingManager = SettingManager(
        camera.settings, self._create_local_settings())

    # Other useful attributes
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

    # Attributes specifically handling the zoom feature
    self._zoom_values: Zoom = Zoom()
    self._zoom_ratio: float = 0.9
    self._zoom_step: int = 0
    self._max_zoom_step: int = 15

  @abstractmethod
  def run(self) -> None:
    """Runs configuration until validation, cancellation, or failure.

    GUI implementations must release resources before returning or raising.
    Callback failures must propagate after the event loop ends.
    :exc:`KeyboardInterrupt` must propagate after cleanup without an error
    dialog or error log.
    """

    ...

  @abstractmethod
  def stop(self) -> None:
    """Closes configuration without validating or exporting a selection.

    GUI implementations must release their resources, including after a
    partial start, and allow repeated calls. This method does not close the
    :class:`~crappy.camera.meta_camera.camera.Camera` object, which remains
    owned by the :class:`~crappy.blocks.meta_block.block.Block`.
    """

    ...

  @abstractmethod
  def watch_shutdown(self, requested: Callable[[], bool]) -> None:
    """Registers a predicate for canceling configuration during preparation.

    GUI implementations check it while :meth:`run() <crappy.tool.\
camera_config.base.camera_config.CameraConfig.run>` executes. A :obj:`True`
    result must close without selection validation or finalization.

    Args:
      requested: :obj:`~collections.abc.Callable` returning :obj:`True` when
        the owning :class:`~crappy.blocks.meta_block.block.Block` must stop,
        for example when its stop event is set or preparation has failed.
    """

    ...

  def log(self,
          level: int,
          msg: str,
          exc_info: ExceptionInfo | None = None) -> None:
    """Records a message using the owning process's logging configuration.

    The logger is created on first use and named after the process and concrete
    configurator class. This method does not install logging handlers.

    Args:
      level: Logging level of the message.
      msg: Message to record.
      exc_info: Exception type, instance, and traceback to log at error level.
        An explicit :obj:`tuple` preserves callback tracebacks outside their
        original exception handler. :obj:`None` records a normal message at
        level.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    if exc_info is None:
      self._logger.log(level, msg)
    else:
      self._logger.exception(msg, exc_info=exc_info)

  def _create_local_settings(self) -> tuple[CameraSetting, ...]:
    """Creates settings owned by the configurator rather than the
    :class:`~crappy.camera.meta_camera.camera.Camera`.

    Called during initialization, before GUI controls exist. Override this hook
    to supply additional settings, and combine the parent's result with your
    own when extending a specialized configuration.

    Returns:
      Settings displayed and applied before the
      :class:`~crappy.camera.meta_camera.camera.Camera` settings. The default
      is an empty :obj:`tuple`.
    """

    return tuple()

  def _extra_actions(self) -> tuple[ConfigAction, ...]:
    """Supplies additional actions for the backend's button panel.

    Called while the backend builds its layout. Callbacks may use the
    initialized shared state, but should not depend on controls that have not
    yet been built.

    Returns:
      Actions with unique identifiers within this window, in display
      order. The default is an empty :obj:`tuple`.
    """

    return tuple()

  def _validate_close(self) -> str | None:
    """Checks whether a user-requested close can accept the configuration.

    Returns:
      A message to display when closing must be refused, or :obj:`None`
      to accept it. The default accepts closing.
      :class:`~crappy.blocks.meta_block.block.Block` shutdown and callback
      failures bypass this check.
    """

    ...

  def _on_valid_close(self) -> None:
    """Finalizes a selection after close validation succeeds.

    Override this hook to save derived values before the backend releases its
    resources. The default does nothing. It is not called during cancellation.
    """

    ...

  def get_config(self) -> tuple[Any, ...] | None:
    """Returns the configuration values after successful configuration.

    The all-in-one :class:`Camera Block <crappy.blocks.Camera>` unpacks the
    :obj:`tuple` into
    :meth:`CameraProcess.set_config() <crappy.blocks.camera_processes.\
CameraProcess.set_config>` before starting its processing worker.
    :class:`~crappy.blocks.vision.CameraSource` returns it to the requesting
    :class:`~crappy.blocks.vision.block.VisionBlock`. Keep the :obj:`tuple`
    compatible with that consumer, and return serializable data.

    Returns:
      Processing-specific configuration values, or :obj:`None` when none
      are needed. The default returns :obj:`None`.
    """

    ...

  def _zoom_at(self, x: int, y: int, direction: int) -> bool:
    """:class:`~crappy.tool.camera_config.config_tools.Zoom` at a display
    position using a signed, backend-independent direction."""

    # Only proceed if the point to zoom on is on the image
    if not direction or not self._is_on_image(x, y):
      return False

    # There is a limit to how much the user can zoom in
    self.log(logging.DEBUG, "Zooming on the image")
    next_step = min(max(self._zoom_step + direction, 0), self._max_zoom_step)
    if next_step == self._zoom_step:
      self.log(logging.DEBUG, "Not zooming, the requested zoom limit is "
                              "reached")
      return False

    self._zoom_step = next_step
    self._display_state.zoom_percent = (100 * (1 / self._zoom_ratio) **
                                        self._zoom_step)

    if self._zoom_step == 0:
      self._zoom_values.reset()
      self.log(logging.DEBUG, "Back to normal display, zoom level is 0")
      return True

    # Compute the new coordinates to display after zooming in or out
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

    # Only proceed if the pixel being pointed is on the image
    if not self._is_on_image(x, y):
      return False

    self.log(logging.DEBUG, "Updating the coordinates of the current pixel")
    (self._display_state.reticle_x,
     self._display_state.reticle_y) = self._coord_to_pix(x, y)
    self._update_pixel_value()
    return True

  def _update_pixel_value(self) -> None:
    """Reads the pre-contrast preview value at the current reticle position."""

    self.log(logging.DEBUG, "Updating the value of the current pixel")

    # Only relevant if there is a displayed image in the first place
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
      # In case no image is acquired yet, indicate it with an error image
      if not self._got_first_img:
        self.log(logging.WARNING, "Could not get an image from the camera, "
                                  "displaying an error image instead")
        no_img_path = importlib.resources.files('crappy').joinpath(
            'tool/data/no_image.png')
        ret = None, np.array(Image.open(BytesIO(no_img_path.read_bytes())))
      # Otherwise just leave the last received image on display
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
      self.log(logging.DEBUG, f"Preview image dtype changed to {self.dtype}")
    if not no_img and img.shape != self.shape:
      self.shape = img.shape
      self.log(logging.DEBUG, f"Preview image shape changed to {self.shape}")

    self._cast_img(img)
    return True

  def _cast_img(self, img: np.ndarray) -> None:
    """Convert a camera image to 8 bits and update ordinary display state."""

    if len(img.shape) not in (2, 3):
      raise ValueError(f"Cannot handle images of shape {img.shape} !")

    if len(img.shape) == 3 and img.shape[2] > 4:
      raise ValueError(f"Cannot handle images of shape {img.shape} !")

    # All rendered images are 2D, either single-channel or RGB
    if len(img.shape) == 3 and img.shape[2] == 1:
      img = img[:, :, 0]
    if len(img.shape) == 3 and img.shape[2] == 2:
      img = img[:, :, 0]
    if len(img.shape) == 3 and img.shape[2] == 4:
      img = img[:, :, :3]
    if len(img.shape) == 3:
      img = img[:, :, ::-1]

    # With auto-range, image values are altered for a better contrast
    if self._display_state.auto_range:
      self.log(logging.DEBUG, "Applying auto range to the image")
      low_thresh, high_thresh = map(float, np.percentile(img, (3, 97)))
      self._low_thresh, self._high_thresh = low_thresh, high_thresh
      self._img = ((np.clip(img, low_thresh, high_thresh) - low_thresh) * 255 /
                   (high_thresh - low_thresh)).astype('uint8')
      bit_depth = int(np.ceil(np.log2(int(np.max(img)) + 1)))
      self._original_img = (img / 2 ** (bit_depth - 8)).astype('uint8')
    # Cast the image to 8-bits for better performance
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
