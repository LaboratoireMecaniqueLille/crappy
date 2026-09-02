# coding: utf-8

import logging
from multiprocessing import Queue, current_process, queues
from time import monotonic, sleep, time
from typing import Any, Callable
import unittest
from unittest.mock import patch

import numpy as np

try:
  import tkinter as tk
except (ImportError, ModuleNotFoundError) as exc:
  raise unittest.SkipTest("tkinter is required for camera configuration tests") \
      from exc

try:
  from PIL import Image as _PILImage
except (ImportError, ModuleNotFoundError) as exc:
  raise unittest.SkipTest("Pillow is required for camera configuration tests") \
      from exc


def _check_graphical_environment() -> None:
  """Skip the graphical suite when Tk cannot connect to a display server."""

  root = None
  try:
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
  except tk.TclError as exc:
    raise unittest.SkipTest(
        f"a working graphical environment is required ({exc})") from exc
  finally:
    if root is not None:
      root.destroy()


_check_graphical_environment()

from crappy.tool.camera_config.camera_config import CameraConfig
from crappy.tool.camera_config.camera_config_boxes import CameraConfigBoxes
from crappy.tool.camera_config.dic_ve_config import DICVEConfig
from crappy.tool.camera_config.dis_correl_config import DISCorrelConfig
from crappy.tool.camera_config.video_extenso_config import VideoExtensoConfig
from crappy.camera.meta_camera.camera import Camera
import crappy.tool.camera_config.camera_config as camera_config_module
import crappy.tool.camera_config.dic_ve_config as dic_ve_config_module
import crappy.tool.camera_config.dis_correl_config as dis_correl_config_module
import crappy.tool.camera_config.video_extenso_config as video_extenso_config_module


class DummyCamera(Camera):
  """Subclass of Camera that does nothing, but Camera cannot be instantiated
  directly."""

  def get_image(self) -> tuple[dict[str, Any] | float, np.ndarray] | None:
    """Mandatory to implement in subclasses."""

    return None


class ConfigurationWindowTestBase(unittest.TestCase):
  """Base test class for testing the
  :class:`~crappy.tool.camera_config.CameraConfig` of Crappy.

  Basically implements setup and teardown methods shared by many test classes.

  .. versionadded:: 2.0.8
  """

  start_histogram_process = False

  def __init__(self,
               *args,
               freq: float | None = None,
               log_level: int | None = None,
               log_queue: queues.Queue | None = None,
               camera: Camera | None = None,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      *args: Positional arguments to pass to the base
        :class:`~unittest.TestCase`.
      freq: The maximum looping frequency the configuration window is allowed
        to loop at.
      log_level: The minimum logging level for the configuration window.
      log_queue: A :obj:`~queues.Queue` for sending the log messages of the
        histogram process to the main logger.
      camera: The :class:`~crappy.camera.Camera` object producing the images
        for the configuration window.
      **kwargs: Keyword arguments to pass to the base
        :class:`~unittest.TestCase`.
    """

    super().__init__(*args, **kwargs)

    self._log_queue: queues.Queue | None = log_queue
    self._log_level: int | None = log_level
    self._freq: float | None = freq
    self._camera: Camera | None = camera

    self._config: (CameraConfig | CameraConfigBoxes | DICVEConfig |
                   DISCorrelConfig | VideoExtensoConfig | None) = None

  def setUp(self) -> None:
    """Defines the arguments to pass to the configuration window if not already
    given."""

    # Patch the references imported directly by the implementation. This keeps
    # expected validation dialogs from blocking a test without replacing a
    # stdlib module globally in sys.modules.
    for module in (camera_config_module, dic_ve_config_module,
                   dis_correl_config_module, video_extenso_config_module):
      patcher = patch.object(module, 'showerror', return_value=None)
      patcher.start()
      self.addCleanup(patcher.stop)

    # Negative-path GUI tests deliberately emit warnings. Silence only the
    # loggers owned by these fixtures and restore their previous state later.
    for class_name in ('CameraConfig', 'CameraConfigBoxes', 'DICVEConfig',
                       'DISCorrelConfig', 'VideoExtensoConfig', 'DummyCamera',
                       'FakeTestCameraSimple', 'FakeTestCameraSpots',
                       'FakeTestCameraParams', 'CameraBoolSetting',
                       'CameraChoiceSetting', 'CameraScaleSetting',
                       'SpotsDetector'):
      logger = logging.getLogger(f"{current_process().name}.{class_name}")
      previous_disabled = logger.disabled
      logger.disabled = True
      self.addCleanup(setattr, logger, 'disabled', previous_disabled)

    # CameraConfig.start contains a defensive fixed sleep for user sessions.
    # Test methods explicitly control elapsed time, so it is unnecessary here.
    sleep_patcher = patch.object(camera_config_module, 'sleep', return_value=None)
    sleep_patcher.start()
    self.addCleanup(sleep_patcher.stop)

    if self._log_queue is None:
      self._log_queue = Queue()
    self.addCleanup(self._close_log_queue)

    if self._log_level is None:
      self._log_level = logging.CRITICAL
    if self._freq is None:
      self._freq = 30
    if self._camera is None:
      self._camera = DummyCamera()

    # Register cleanup before constructing or starting the GUI so that a setup
    # failure cannot leave a Tk window or HistogramProcess behind.
    self.addCleanup(self._close_configuration)
    self.customSetUp()

  def customSetUp(self) -> None:
    """Instantiates the configuration window and starts it."""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = True
    self.start_configuration()

  def start_configuration(self) -> None:
    """Initialize scheduling and start the histogram process when relevant."""

    if self.start_histogram_process:
      self._config.start()
      return

    # Most GUI tests exercise controls, drawing, or image conversion and do not
    # assert histogram behavior. Keep that independent subsystem mocked there.
    histogram_patcher = patch.object(self._config, '_calc_hist',
                                     return_value=None)
    histogram_patcher.start()
    self.addCleanup(histogram_patcher.stop)
    join_patcher = patch.object(self._config._histogram_process, 'join',
                                return_value=None)
    join_patcher.start()
    self.addCleanup(join_patcher.stop)
    self._config._n_loops = 0
    self._config._last_upd_t = time()

  def tearDown(self) -> None:
    """Runs subclass-specific cleanup before registered fail-safe cleanup."""

    self.customTearDown()

  def _close_configuration(self) -> None:
    """Stops the GUI and its child process, including after a test failure."""

    if self._config is None:
      return

    process = self._config._histogram_process
    try:
      window_exists = bool(self._config.winfo_exists())
    except tk.TclError:
      window_exists = False

    if window_exists:
      if process.pid is None:
        # CameraConfig.stop cannot join a process that never started.
        for queue in (self._config._img_in, self._config._img_out):
          queue.cancel_join_thread()
          queue.close()
        try:
          self._config.destroy()
        except tk.TclError:
          pass
      else:
        self._config.finish()

    if process.pid is not None:
      process.join(1.0)
      if process.is_alive():
        process.kill()
        process.join(1.0)

    self.assertFalse(process.is_alive())

  def _close_log_queue(self) -> None:
    """Release the logging queue without waiting for its feeder thread."""

    if self._log_queue is not None:
      try:
        self._log_queue.cancel_join_thread()
        self._log_queue.close()
      except (OSError, ValueError):
        pass

  def run_config_cycle(self, elapsed: float = 0.05) -> None:
    """Run one deterministic acquisition/update cycle."""

    self._config._last_upd_t -= elapsed
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

  def wait_until(self,
                 predicate: Callable[[], bool],
                 timeout: float = 3.0) -> bool:
    """Poll a condition with a deadline instead of sleeping a fixed duration."""

    deadline = monotonic() + timeout
    while monotonic() < deadline:
      if predicate():
        return True
      sleep(0.01)
    return predicate()

  def wait_for_histogram(self, timeout: float = 3.0) -> bool:
    """Wait for CameraConfig to receive a histogram from its child process."""

    def received_histogram() -> bool:
      self._config._calc_hist()
      return self._config._hist is not None

    return self.wait_until(received_histogram, timeout)

  def customTearDown(self) -> None:
    """Meant to be overwritten in subclasses for custom behavior."""

    ...


class FakeTestCameraSimple(Camera):
  """Fake :class:`~crappy.camera.Camera` used for tests, generating a
  grey-level gradient image.

  .. versionadded:: 2.0.8
  """

  def __init__(self, min_val: int = 0, max_val: int = 255) -> None:
    """Initializes the parent class.

    Args:
      min_val: Minimum value in the generated image.
      max_val: Maximum value in the generated image.
    """

    super().__init__()

    self._min = min_val
    self._max = max_val

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a grey-level image containing a gradient from the specified
    minimum to the specified maximum."""

    x, y = np.mgrid[0:240, 0:320]
    ret = (self._min + (x + y) / np.max(x + y) *
           (self._max - self._min)).astype(np.uint8)
    return time(), ret


class FakeTestCameraSpots(Camera):
  """Fake :class:`~crappy.camera.Camera` used for test of the
  video-extensometry configurator, generating a white image with four round
  spots.

  .. versionadded:: 2.0.8
  """

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a white image with four round black spots."""

    ret = np.full((240, 320), 255, dtype=np.uint8)
    y, x = np.ogrid[:ret.shape[0], :ret.shape[1]]
    for x_center, y_center in ((80, 80), (80, 160),
                               (160, 80), (160, 160)):
      ret[(x - x_center) ** 2 + (y - y_center) ** 2 <= 20 ** 2] = 0

    return time(), ret


class FakeTestCameraParams(Camera):
  """Fake :class:`~crappy.camera.Camera` used for testing the parameter
  handling in the configuration interface.

  .. versionadded:: 2.0.8
  """

  def __init__(self) -> None:
    """Initializes the parent class and sets the attributes."""

    super().__init__()

    self._bool_getter_called: bool = False
    self._bool_setter_called: bool = False
    self._scale_int_getter_called: bool = False
    self._scale_int_setter_called: bool = False
    self._scale_float_getter_called: bool = False
    self._scale_float_setter_called: bool = False
    self._choice_getter_called: bool = False
    self._choice_setter_called: bool = False

    self._scale_int_bounds = (-100, 100, 2)
    self._scale_float_bounds = (-10.0, 10.0, 0.1)
    self._choices = ('choice_1', 'choice_2', 'choice_3')

  def open(self) -> None:
    """Instantiates the four camera parameters to test."""

    self.add_bool_setting('bool_setting',
                          self._bool_getter,
                          self._bool_setter,
                          True)
    
    self.add_scale_setting('scale_int_setting',
                           self._scale_int_bounds[0],
                           self._scale_int_bounds[1],
                           self._scale_int_getter,
                           self._scale_int_setter,
                           default=0,
                           step=self._scale_int_bounds[2])

    self.add_scale_setting('scale_float_setting',
                           self._scale_float_bounds[0],
                           self._scale_float_bounds[1],
                           self._scale_float_getter,
                           self._scale_float_setter,
                           default=0.,
                           step=self._scale_float_bounds[2])

    self.add_choice_setting('choice_setting',
                            self._choices,
                            self._choice_getter,
                            self._choice_setter,
                            self._choices[0])

    # Left out on purpose
    # self.set_all()

  def get_image(self) -> tuple[dict[str, Any] | float, np.ndarray] | None:
    """Added because this method must be defined by children classes."""

    return super().get_image()

  def _bool_setter(self, value: bool) -> None:
    """Setter for the boolean parameter."""

    self._bool_setter_called = True
    self.settings['bool_setting']._value_no_getter = value

  def _bool_getter(self) -> bool:
    """Getter for the boolean parameter."""

    self._bool_getter_called = True
    return self.settings['bool_setting']._value_no_getter

  def _scale_int_setter(self, value: int) -> None:
    """Setter for the integer parameter."""

    self._scale_int_setter_called = True
    self.settings['scale_int_setting']._value_no_getter = value

  def _scale_int_getter(self) -> int:
    """Getter for the integer parameter."""

    self._scale_int_getter_called = True
    return self.settings['scale_int_setting']._value_no_getter
  
  def _scale_float_setter(self, value: float) -> None:
    """Setter for the float parameter."""

    self._scale_float_setter_called = True
    self.settings['scale_float_setting']._value_no_getter = value

  def _scale_float_getter(self) -> float:
    """Getter for the float parameter."""

    self._scale_float_getter_called = True
    return self.settings['scale_float_setting']._value_no_getter

  def _choice_setter(self, value: str) -> None:
    """Setter for the string parameter."""

    self._choice_setter_called = True
    self.settings['choice_setting']._value_no_getter = value

  def _choice_getter(self) -> str:
    """Getter for the string parameter."""

    self._choice_getter_called = True
    return self.settings['choice_setting']._value_no_getter
