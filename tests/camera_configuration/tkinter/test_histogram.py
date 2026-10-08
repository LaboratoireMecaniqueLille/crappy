# coding: utf-8

"""Tk histogram queue integration and receipt of completed results."""

from queue import Empty
from unittest.mock import Mock, patch
import numpy as np

from ._fixtures import TkinterConfigTestCase, FakeTestCameraSimple
from crappy.tool.camera_config.tkinter import TkinterCameraConfig


class TestHistogram(TkinterConfigTestCase):
  """Check queue receipt at the Tk histogram rendering boundary."""

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def customSetUp(self) -> None:
    """Used for instantiating the configuration window without starting it for
    now."""

    self._config = TkinterCameraConfig(self._camera, self._log_queue,
                                       self._log_level, self._freq, None)

    self._config._testing = True

  def test_calc_hist_receives_output_while_process_is_busy(self) -> None:
    """Checks completed output is read while another calculation is active."""

    histogram = np.ones((80, 512), dtype=np.uint8)
    processing_event = Mock()
    processing_event.is_set.return_value = True
    img_in = Mock()
    img_out = Mock()
    img_out.get_nowait.side_effect = (histogram, Empty)
    self._config._original_img = np.zeros((240, 320), dtype=np.uint8)

    with (patch.object(self._config, '_processing_event', processing_event),
          patch.object(self._config, '_img_in', img_in),
          patch.object(self._config, '_img_out', img_out)):
      self._config._calc_hist()

    img_in.put_nowait.assert_not_called()
    np.testing.assert_array_equal(self._config._hist, histogram)


class TestNormalRun(TkinterConfigTestCase):
  """Class for testing the normal operating mode of the configuration
  window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def customSetUp(self) -> None:
    """Used for setting the testing mode to :obj:`False`."""

    self._config = TkinterCameraConfig(self._camera, self._log_queue,
                                       self._log_level, self._freq, None)

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

    self.assertTrue(self._config._lifecycle._process_closed)
