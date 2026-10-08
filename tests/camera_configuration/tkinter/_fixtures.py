# coding: utf-8

"""Resources and deterministic update helpers for Tkinter window tests."""

import logging
from collections.abc import Callable
from multiprocessing import Queue, current_process
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

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

from crappy.tool.camera_config.tkinter import TkinterCameraConfig
from crappy.camera.meta_camera import Camera
import crappy.tool.camera_config.tkinter.camera_config as camera_config_module
from .._fixtures import (DummyCamera, FakeTestCameraSimple, FakeTestCameraSpots,
                        FakeTestCameraParams)


class TkinterConfigTestCase(unittest.TestCase):
  """Create each window and Camera during setup, with fail-safe cleanup."""

  start_histogram_process = False

  @classmethod
  def setUpClass(cls) -> None:
    """Skip Tk tests if no display is available, without affecting other layers."""

    super().setUpClass()
    root = None
    try:
      root = tk.Tk()
      root.withdraw()
      root.update_idletasks()
    except tk.TclError as error:
      raise unittest.SkipTest(
          f"a working graphical environment is required ({error})") from error
    finally:
      if root is not None:
        root.destroy()

  def make_camera(self) -> Camera:
    """Return the hardware-free Camera used by this test."""

    return DummyCamera()

  def setUp(self) -> None:
    """Create isolated Cameras, log queues, and configuration windows."""

    super().setUp()
    patcher = patch.object(camera_config_module, 'showerror', return_value=None)
    patcher.start()
    self.addCleanup(patcher.stop)

    # Restore only the fixture-owned loggers after expected negative-path tests
    for class_name in ('TkinterCameraConfig', 'TkinterCameraConfigBoxes',
                       'TkinterDICVEConfig', 'TkinterDISCorrelConfig',
                       'TkinterVideoExtensoConfig', 'DummyCamera',
                       'FakeTestCameraSimple', 'FakeTestCameraSpots',
                       'FakeTestCameraParams', 'CameraBoolSetting',
                       'CameraChoiceSetting', 'CameraScaleSetting',
                       'SpotsDetector'):
      logger = logging.getLogger(f"{current_process().name}.{class_name}")
      previous_disabled = logger.disabled
      logger.disabled = True
      self.addCleanup(setattr, logger, 'disabled', previous_disabled)

    self._log_queue = Queue()
    self.addCleanup(self._close_log_queue)
    self._log_level = logging.CRITICAL
    self._freq = 30
    self._camera = self.make_camera()
    self._config = None
    self.addCleanup(self._close_configuration)
    self.customSetUp()
    # Windows delivers native mapping and resize events outside idle processing
    self._config.update()

  def customSetUp(self) -> None:
    """Instantiates the configuration window and starts it."""

    self._config = TkinterCameraConfig(self._camera, self._log_queue,
                                       self._log_level, self._freq, None)

    self._config._testing = True
    self.start_configuration()

  def start_configuration(self) -> None:
    """Initialize scheduling and start the histogram process when relevant."""

    if self.start_histogram_process:
      self._config.start()
      return

    # Most GUI tests exercise controls, drawing, or image conversion and do not
    # assert worker behavior. Keep it mocked, including when run() calls start().
    # Spec the class to avoid reading an unstarted process's sentinel property
    process = Mock(spec=type(self._config._histogram_process))
    process.is_alive.return_value = False
    self._config._histogram_process = process
    self._config._lifecycle._histogram_process = process
    histogram_patcher = patch.object(self._config, '_calc_hist',
                                     return_value=None)
    histogram_patcher.start()
    self.addCleanup(histogram_patcher.stop)
    self._config._n_loops = 0
    self._config._last_upd_t = monotonic()

  def _close_configuration(self) -> None:
    """Stops the GUI and its child process, including after a test failure."""

    if self._config is None:
      return

    self._config.stop()

    self.assertTrue(self._config._lifecycle._process_closed)

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
    self._config._next_acq_t = -float('inf')
    self._config._img_acq_sched()
    self._config._upd_var_sched()

  def setting_control(self, name: str):
    """Return the Tk control owned by this configuration window."""

    return self._config._setting_controls[self._camera.settings[name]]

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
