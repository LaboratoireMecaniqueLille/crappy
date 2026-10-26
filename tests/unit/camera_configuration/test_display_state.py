# coding: utf-8

import unittest

from crappy.tool.camera_config.config_tools.zoom import Zoom
from crappy.tool.camera_config.base._display_state import (DisplayGeometry,
                                                         DisplayState)


class TestDisplayState(unittest.TestCase):
  """Headless checks for preview state and display-coordinate geometry."""

  def test_defaults_are_plain_values(self) -> None:
    state = DisplayState()

    self.assertEqual(state.fps, 0.0)
    self.assertEqual(state.zoom_percent, 100.0)
    self.assertFalse(state.auto_range)
    self.assertFalse(state.auto_apply)
    self.assertEqual((state.reticle_x, state.reticle_y,
                      state.reticle_value), (0, 0, 0))

  def test_centered_image_hit_test_and_coordinate_conversion(self) -> None:
    geometry = DisplayGeometry(width=400, height=300,
                               image_width=320, image_height=240)

    self.assertEqual((geometry.left, geometry.top), (40, 30))
    self.assertFalse(geometry.contains(39, 30))
    self.assertTrue(geometry.contains(40, 30))
    self.assertTrue(geometry.contains(360, 270))
    self.assertFalse(geometry.contains(361, 270))
    self.assertEqual(geometry.to_pixel(40, 30, 320, 240, Zoom()), (0, 0))
    self.assertEqual(geometry.to_pixel(360, 270, 320, 240, Zoom()),
                     (319, 239))

    zoom = Zoom(x_low=0.25, x_high=0.75, y_low=0.25, y_high=0.75)
    self.assertEqual(geometry.to_pixel(200, 150, 320, 240, zoom),
                     (160, 120))

  def test_fit_and_zero_dimensions(self) -> None:
    geometry = DisplayGeometry(width=400, height=300)

    self.assertEqual(geometry.fit(320, 240), (400, 300))
    self.assertEqual(geometry.fit(100, 200), (150, 300))
    self.assertEqual(geometry.fit(200, 100), (400, 200))
    self.assertFalse(geometry.contains(0, 0))
    self.assertEqual(geometry.to_pixel(0, 0, 320, 240, Zoom()), (0, 0))

    geometry.width = 0
    self.assertEqual(geometry.fit(320, 240), (0, 0))
