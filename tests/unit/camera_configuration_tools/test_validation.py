"""Public validation and protected state for this feature."""

from unittest import TestCase
import numpy as np
from crappy.tool.camera_config import Zoom


class TestZoomValidation(TestCase):
  def test_zoom_rejects_singular_ratios(self):
    zoom = Zoom()
    for ratio in (0, -1, np.nan, np.inf):
      with self.subTest(ratio=ratio):
        with self.assertRaises((TypeError, ValueError)):
          zoom.update_zoom(0.5, 0.5, ratio)
    self.assertEqual(zoom, Zoom())
