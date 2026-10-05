# coding: utf-8

"""Qt histogram queue integration and an actual acquisition-loop smoke test."""

from queue import Empty
from time import monotonic
from unittest.mock import patch
import numpy as np

from ._fixtures import PyQtConfigTestCase


class TestHistogram(PyQtConfigTestCase):
  def test_calc_hist_receives_output_while_process_is_busy(self) -> None:
    """Reading completed histograms is independent of a pending calculation."""

    config = self.make_config()
    histogram = np.ones((80, 512), dtype=np.uint8)
    config._original_img = np.zeros((100, 100), dtype=np.uint8)
    with (patch.object(config, '_processing_event') as processing,
          patch.object(config, '_img_in') as img_in,
          patch.object(config, '_img_out') as img_out):
      processing.is_set.return_value = True
      img_out.get_nowait.side_effect = (histogram, Empty)
      config._calc_hist()

    img_in.put_nowait.assert_not_called()
    np.testing.assert_array_equal(config._hist, histogram)

  def test_normal_run_updates_preview_and_histogram(self) -> None:
    """Qt timers and the actual histogram process work together and shut down."""

    from PyQt6.QtCore import QTimer

    config = self.make_config(histogram_process=True)
    complete = []
    deadline = monotonic() + 5.0
    timer = QTimer(config)

    def check_preview() -> None:
      if config._hist is not None and config._img_canvas.pixmap() is not None:
        complete.append(True)
        config.stop()
      elif monotonic() >= deadline:
        config.stop()

    timer.timeout.connect(check_preview)
    timer.start(10)
    try:
      config.run()
    finally:
      timer.stop()

    self.assertTrue(complete, 'Timed out waiting for the Qt image and histogram')
    self.assertEqual(config.shape, (100, 100))
    self.assertIsNotNone(config._hist_canvas.pixmap())
    self.assertFalse(config._histogram_process.is_alive())
    self.assertTrue(config._img_in._closed)
    self.assertTrue(config._img_out._closed)
