# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from crappy.camera.fake_camera import FakeCamera
import crappy.camera.fake_camera as fake_camera_module


class TestFakeCamera(TestCase):
  """Unit tests for the FakeCamera object."""

  def test_constructor_adds_settings_with_expected_defaults(self) -> None:
    """Checks fake camera settings."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()

    self.assertEqual(set(camera.settings),
                     {'width', 'height', 'speed', 'fps'})
    self.assertEqual(camera.width, 1280)
    self.assertEqual(camera.height, 720)
    self.assertEqual(camera.speed, 100.)
    self.assertEqual(camera.fps, 50.)
    self.assertEqual(camera._frame_nr, -1)
    self.assertEqual(camera._t0, 0)
    self.assertEqual(camera._t, -float('inf'))

  def test_open_sets_settings_resets_state_and_generates_image(self) -> None:
    """Checks open initialization."""

    with patch.object(fake_camera_module, 'time',
                      side_effect=[0, 10]):
      camera = FakeCamera()
      camera._frame_nr = 12
      camera._t = 4
      camera.open(width=4, height=3, speed=2, fps=25)

    self.assertEqual(camera.width, 4)
    self.assertEqual(camera.height, 3)
    self.assertEqual(camera.speed, 2.)
    self.assertEqual(camera.fps, 25.)
    self.assertEqual(camera._frame_nr, -1)
    self.assertEqual(camera._t0, 10)
    self.assertEqual(camera._t, -float('inf'))
    self.assertEqual(camera._img.shape, (3, 4))
    self.assertEqual(camera._img.dtype, np.uint8)

  def test_width_and_height_changes_regenerate_image(self) -> None:
    """Checks image regeneration through scale setting setters."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()
      camera.open(width=4, height=3)

    camera.width = 2
    camera.height = 4

    self.assertEqual(camera._img.shape, (4, 2))

  def test_generated_image_is_vertical_gradient(self) -> None:
    """Checks base image contents."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()
      camera.open(width=3, height=4)

    np.testing.assert_array_equal(camera._img[:, 0],
                                  np.array([0, 63, 127, 191],
                                           dtype=np.uint8))
    np.testing.assert_array_equal(camera._img[:, 1], camera._img[:, 0])

  def test_get_image_returns_none_when_called_too_soon(self) -> None:
    """Checks FPS rate limiting."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()
      camera.open(width=3, height=4, fps=2)

    camera._t = 10

    with patch.object(fake_camera_module, 'time', return_value=10.4):
      self.assertIsNone(camera.get_image())

    self.assertEqual(camera._frame_nr, -1)

  def test_get_image_returns_shifted_frame_when_due(self) -> None:
    """Checks moving-line frame generation."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()
      camera.open(width=3, height=4, speed=1, fps=100)

    camera._t0 = 0
    camera._t = -float('inf')

    with patch.object(fake_camera_module, 'time', side_effect=[2, 2]):
      t, img = camera.get_image()

    self.assertEqual(t, 2)
    self.assertEqual(camera._t, 2)
    self.assertEqual(camera._frame_nr, 0)
    np.testing.assert_array_equal(img[:, 0],
                                  np.array([127, 191, 0, 63],
                                           dtype=np.uint8))

  def test_get_image_wraps_shift_by_image_height(self) -> None:
    """Checks cyclic line movement."""

    with patch.object(fake_camera_module, 'time', return_value=0):
      camera = FakeCamera()
      camera.open(width=2, height=4, speed=3, fps=100)

    camera._t0 = 0
    camera._t = -float('inf')

    with patch.object(fake_camera_module, 'time', side_effect=[3, 3]):
      _, img = camera.get_image()

    # int(3 px/s * 3 s) % 4 rows = 1 row shift.
    np.testing.assert_array_equal(img[:, 0],
                                  np.array([63, 127, 191, 0],
                                           dtype=np.uint8))
