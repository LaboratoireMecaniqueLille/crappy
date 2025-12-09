# coding: utf-8

import unittest
import numpy as np
from itertools import combinations

from crappy.tool.camera_config.config_tools import Zoom


class TestZoom(unittest.TestCase):
  """Class for testing the
  :class:`~crappy.tool.camera_config.config_tools.Zoom` class.

  .. versionadded:: 2.0.8
  """

  def setUp(self) -> None:
    """Instantiates the Zoom to test."""

    self._zoom = Zoom()

  def test_01_zoom(self) -> None:
    """Tests whether the coordinates are updated consistently when zooming in
    and out and when moving on the image."""

    # At the beginning, the zoom should cover the entire image
    self.assertEqual(self._zoom.x_low, 0.0)
    self.assertEqual(self._zoom.y_low, 0.0)
    self.assertEqual(self._zoom.x_high, 1.0)
    self.assertEqual(self._zoom.y_high, 1.0)

    # Check if zooming in and out works as expected
    for ratio in np.linspace(0.1, 2.0, 20):
      with self.subTest(ratio=ratio):
        self._zoom.update_zoom(0.5, 0.5, ratio)

        # Cannot zoom out when the image is already fully zoomed out
        if ratio < 1.:
          self.assertEqual(self._zoom.x_low, 0.0)
          self.assertEqual(self._zoom.y_low, 0.0)
          self.assertEqual(self._zoom.x_high, 1.0)
          self.assertEqual(self._zoom.y_high, 1.0)

        # In this context the position of the edges is easy to compute
        else:
          self.assertAlmostEqual(self._zoom.x_low, 0.0 + (1 - 1 / ratio) / 2)
          self.assertAlmostEqual(self._zoom.y_low, 0.0 + (1 - 1 / ratio) / 2)
          self.assertAlmostEqual(self._zoom.x_high, 1.0 - (1 - 1 / ratio) / 2)
          self.assertAlmostEqual(self._zoom.y_high, 1.0 - (1 - 1 / ratio) / 2)

        self._zoom.reset()

    # Reset zoom for next range of tests
    self._zoom.reset()
    self.assertEqual(self._zoom.x_low, 0.0)
    self.assertEqual(self._zoom.y_low, 0.0)
    self.assertEqual(self._zoom.x_high, 1.0)
    self.assertEqual(self._zoom.y_high, 1.0)

    # Check that the values are as expected when zooming on edge of the image
    for x, y in combinations((0.0, 0.0, 1.0, 1.0), 2):
      with self.subTest(x=x, y=y):
        self._zoom.update_zoom(x, y, 1.1)
        if x == 0.0:
          self.assertEqual(self._zoom.x_low, 0.0)
          self.assertAlmostEqual(self._zoom.x_high, 1.0 - (1.0 - 1 / 1.1))
        if y == 0.0:
          self.assertEqual(self._zoom.y_low, 0.0)
          self.assertAlmostEqual(self._zoom.y_high, 1.0 - (1.0 - 1 / 1.1))
        if x == 1.0:
          self.assertEqual(self._zoom.x_high, 1.0)
          self.assertAlmostEqual(self._zoom.x_low, 0.0 + (1.0 - 1 / 1.1))
        if y == 1.0:
          self.assertEqual(self._zoom.y_high, 1.0)
          self.assertAlmostEqual(self._zoom.y_low, 0.0 + (1.0 - 1 / 1.1))
        self._zoom.reset()

    # Check that moving to the corner of the image has the expected effect
    self._zoom.update_zoom(0.5, 0.5, 2.0)
    self._zoom.update_move(1.0, 1.0)
    self.assertEqual(self._zoom.x_low, 0.5)
    self.assertEqual(self._zoom.y_low, 0.5)
    self.assertEqual(self._zoom.x_high, 1.0)
    self.assertEqual(self._zoom.y_high, 1.0)

    self._zoom.reset()

    # Check that moving to the corner of the image has the expected effect
    self._zoom.update_zoom(0.5, 0.5, 2.0)
    self._zoom.update_move(-1.0, -1.0)
    self.assertEqual(self._zoom.x_low, 0.0)
    self.assertEqual(self._zoom.y_low, 0.0)
    self.assertEqual(self._zoom.x_high, 0.5)
    self.assertEqual(self._zoom.y_high, 0.5)

    last_x_low = 0.0
    last_y_low = 0.0
    last_x_high = 0.5
    last_y_high = 0.5

    # Check that moving the zoom behaves as expected, and that the edges are
    # correctly handled
    for x, y in combinations((-0.1, -0.1, 0.0, 0.0, 0.1, 0.1), 2):
      with self.subTest(x=x, y=y):
        if max(last_x_low + x, 0.0) == 0.0 or min(last_x_high + x, 1.0) == 1.0:
          dx = 0.0
        else:
          dx = x
        if max(last_y_low + y, 0.0) == 0.0 or min(last_y_high + y, 1.0) == 1.0:
          dy = 0.0
        else:
          dy = y
        self._zoom.update_move(x, y)
        self.assertAlmostEqual(self._zoom.x_low, last_x_low + dx)
        self.assertAlmostEqual(self._zoom.y_low, last_y_low + dy)
        self.assertAlmostEqual(self._zoom.x_high, last_x_high + dx)
        self.assertAlmostEqual(self._zoom.y_high, last_y_high + dy)
        last_x_low = self._zoom.x_low
        last_y_low = self._zoom.y_low
        last_x_high = self._zoom.x_high
        last_y_high = self._zoom.y_high
