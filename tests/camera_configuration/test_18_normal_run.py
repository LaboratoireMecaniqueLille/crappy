# coding: utf-8

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.camera_config import CameraConfig


class TestNormalRun(ConfigurationWindowTestBase):
  """Class for testing the normal operating mode of the configuration 
  window.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def customSetUp(self) -> None:
    """Used for setting the testing mode to :obj:`False`."""

    self._config = CameraConfig(self._camera, self._log_queue,
                                self._log_level, self._freq)

    self._config._testing = False
    self._config.start()

  def test_normal_run(self) -> None:
    """Tests whether the interface is able to start and finish correctly in
    normal operating mode."""

    n_loops = [0]

    def image_and_histogram_ready() -> bool:
      self._config.update()
      n_loops[0] = max(n_loops[0], self._config._n_loops)
      return n_loops[0] > 0 and self._config._pil_hist is not None

    self.assertTrue(self.wait_until(image_and_histogram_ready, timeout=5.0))

    # There should have been images acquired
    self.assertGreater(n_loops[0], 0)

    # The histogram process should be alive and there should be a histogram
    self.assertTrue(self._config._histogram_process.is_alive())
    self.assertIsNotNone(self._config._pil_hist)

    # Delete the configuration window
    self._config.finish()

    self._config._histogram_process.join(1.0)
    self.assertFalse(self._config._histogram_process.is_alive())
