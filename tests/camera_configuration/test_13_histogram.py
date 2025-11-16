# coding: utf-8

from time import sleep
import numpy as np

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.camera_config import CameraConfig


class TestHistogram(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def customSetUp(self) -> None:
    """"""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = True

  def test_histogram(self) -> None:
    """"""

    # Default configuration
    self.assertFalse(self._config._histogram_process.is_alive())
    self.assertFalse(self._config._histogram_process._stop_event.is_set())
    self.assertFalse(
        self._config._histogram_process._processing_event.is_set())
    self.assertTrue(self._config._histogram_process._img_in.empty())
    self.assertTrue(self._config._img_out.empty())

    # Acquire an image and put it in the queue for processing
    _, img = self._camera.get_image()
    self._config._img_in.put_nowait((img, False, 0, 255))
    sleep(1)

    # The flags and queue should be similar to default
    self.assertFalse(self._config._histogram_process.is_alive())
    self.assertFalse(self._config._histogram_process._stop_event.is_set())
    self.assertFalse(
        self._config._histogram_process._processing_event.is_set())
    self.assertFalse(self._config._histogram_process._img_in.empty())
    self.assertTrue(self._config._img_out.empty())

    # Start the histogram process and wait for it to work
    self._config._histogram_process.start()
    sleep(2)

    try:
      # The process should now be alive
      self.assertTrue(self._config._histogram_process.is_alive())
      self.assertFalse(self._config._histogram_process._stop_event.is_set())
      self.assertFalse(
          self._config._histogram_process._processing_event.is_set())
      self.assertTrue(self._config._histogram_process._img_in.empty())
      # There should be a histogram image available in the queue
      self.assertFalse(self._config._img_out.empty())

      # Read the generated histogram, that should be an array
      img = self._config._img_out.get_nowait()
      self.assertIsInstance(img, np.ndarray)

      # Raise the stop flag and wait for the histogram process to stop
      self._config._stop_event.set()
      sleep(1)

      # The process should now be stopped
      self.assertFalse(self._config._histogram_process.is_alive())
      # The stop event should have been set
      self.assertTrue(self._config._histogram_process._stop_event.is_set())
      self.assertFalse(
          self._config._histogram_process._processing_event.is_set())
      self.assertTrue(self._config._histogram_process._img_in.empty())
      self.assertTrue(self._config._img_out.empty())

    # Ensure the histogram process stops in case a test fails
    finally:
      if self._config._histogram_process.is_alive():
        self._config._histogram_process.kill()
