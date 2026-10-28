# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from crappy.camera.fake_camera import FakeCamera
import crappy.camera.fake_camera as fake_camera_module


from tests.support.clock import Clock


class TestFakeCamera(TestCase):
  """Generated frames and rate limiting through the public Camera interface."""

  def setUp(self):
    self.clock = Clock()
    self.enterContext(patch.object(fake_camera_module, 'time', self.clock))

  def make_camera(self, **kwargs):
    camera = FakeCamera()
    self.addCleanup(camera.close)
    camera.open(**kwargs)
    return camera

  def test_constructor_adds_settings_with_expected_defaults(self):
    camera = FakeCamera()
    self.assertTrue({'width', 'height', 'speed', 'fps'}.issubset(camera.settings))
    self.assertEqual((camera.width, camera.height, camera.speed, camera.fps),
                     (1280, 720, 100, 50))
    camera.open()
    _, frame = camera.get_image()
    self.assertEqual(frame.shape, (camera.height, camera.width))
    self.assertEqual(frame.dtype, np.uint8)

  def test_open_sets_settings_resets_state_and_generates_image(self):
    camera = self.make_camera(width=4, height=3, speed=1, fps=25)
    _, original = camera.get_image()
    self.clock.now = 1
    _, moved = camera.get_image()
    self.assertFalse(np.array_equal(moved, original))
    self.clock.now = 10
    camera.open(width=4, height=3, speed=2, fps=25)
    timestamp, restarted = camera.get_image()
    self.assertEqual((camera.width, camera.height, camera.speed, camera.fps), (4, 3, 2, 25))
    self.assertEqual(timestamp, 10)
    np.testing.assert_array_equal(restarted, original)

  def test_width_and_height_changes_regenerate_image(self):
    camera = self.make_camera(width=4, height=3)
    camera.width, camera.height = 2, 4
    _, frame = camera.get_image()
    self.assertEqual(frame.shape, (4, 2))

  def test_generated_image_is_vertical_gradient(self):
    camera = self.make_camera(width=3, height=4, speed=0)
    _, frame = camera.get_image()
    self.assertTrue(np.all(np.diff(frame[:, 0].astype(int)) > 0))
    np.testing.assert_array_equal(frame[:, 1], frame[:, 0])

  def test_get_image_returns_none_when_called_too_soon(self):
    camera = self.make_camera(width=3, height=4, fps=2)
    self.clock.now = 10
    self.assertIsNotNone(camera.get_image())
    self.clock.now = 10.4
    self.assertIsNone(camera.get_image())
    self.clock.now = 10.6
    self.assertIsNotNone(camera.get_image())

  def test_get_image_returns_shifted_frame_when_due(self):
    camera = self.make_camera(width=3, height=4, speed=1, fps=100)
    _, first = camera.get_image()
    self.clock.now = 2
    timestamp, frame = camera.get_image()
    self.assertEqual(timestamp, 2)
    np.testing.assert_array_equal(frame, np.roll(first, -2, axis=0))

  def test_get_image_wraps_shift_by_image_height(self):
    camera = self.make_camera(width=2, height=4, speed=3, fps=100)
    _, first = camera.get_image()
    self.clock.now = 3
    _, frame = camera.get_image()
    np.testing.assert_array_equal(frame, np.roll(first, -1, axis=0))
