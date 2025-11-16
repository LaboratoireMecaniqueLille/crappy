# coding: utf-8

from time import sleep, time
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.camera_config import CameraConfig


class TestNormalRun(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)
    self._exit = False

  def customSetUp(self) -> None:
    """"""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = False
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def test_normal_run(self) -> None:
    """"""

    n_loops = 0
    t0 = time()
    while time() - t0 < 2:
      self._config.update()
      n_loops = max(n_loops, self._config._n_loops)
      sleep(0.1)

    # There should have been images acquired
    self.assertGreater(n_loops, 0)

    # The histogram process should be alive and there should be a histogram
    self.assertTrue(self._config._histogram_process.is_alive())
    self.assertIsNotNone(self._config._pil_hist)

    # Delete the configuration window
    self._config.finish()

    # After 2 seconds, the histogram process should have stopped
    sleep(2)
    self.assertFalse(self._config._histogram_process.is_alive())
