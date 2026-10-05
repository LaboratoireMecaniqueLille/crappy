# coding: utf-8

"""Camera-image normalization is shared logic, not a GUI rendering feature."""

from itertools import product
import unittest
from unittest.mock import sentinel
import numpy as np

from .._fixtures import DummyCamera
from ._fixtures import RecordingCore


class TestImageConversion(unittest.TestCase):
  def setUp(self) -> None:
    self.config = RecordingCore(DummyCamera(), sentinel.log_queue,
                                None, None, None)

  def test_channel_layout_and_bit_depth_conversion(self) -> None:
    """Normalize gray, gray-alpha, BGR, and BGRA inputs to gray or RGB."""

    gray = np.linspace(0, 255, 16, dtype=np.uint8).reshape(4, 4)
    color = np.stack((gray, gray // 2, gray // 3, gray), axis=2)
    for channels, dtype in product((None, 1, 2, 3, 4), (np.uint8, np.uint16)):
      with self.subTest(channels=channels, dtype=dtype):
        image = gray if channels is None else color[:, :, :channels]
        expected = gray if channels is None or channels < 3 else \
            color[:, :, :3][:, :, ::-1]
        image = image.astype(dtype)
        if dtype is np.uint16:
          image *= 257
        self.config._cast_img(image)

        np.testing.assert_array_equal(self.config._img, expected)
        np.testing.assert_array_equal(self.config._original_img, expected)
        self.assertEqual(self.config._img.dtype, np.dtype('uint8'))
        self.assertEqual(self.config._display_state.detected_bits,
                         8 if dtype is np.uint8 else 16)
        self.assertFalse(np.shares_memory(self.config._img,
                                         self.config._original_img))

  def test_invalid_image_shapes_are_rejected(self) -> None:
    """Reject layouts that neither preview renderer supports."""

    for shape in ((4,), (2, 3, 4, 5), (2, 3, 5)):
      with self.subTest(shape=shape):
        with self.assertRaisesRegex(ValueError, 'Cannot handle images'):
          self.config._cast_img(np.ones(shape, dtype=np.uint8))

  def test_auto_range_preserves_source_pixels_and_indicators(self) -> None:
    """Only the display copy is stretched when Auto range is enabled."""

    image = np.linspace(3, 252, 100, dtype=np.uint8).reshape(10, 10)
    self.config._display_state.auto_range = True
    self.config._cast_img(image)

    np.testing.assert_array_equal(self.config._original_img, image)
    self.assertEqual((self.config._img.min(), self.config._img.max()), (0, 255))
    self.assertEqual((self.config._display_state.min_pixel,
                      self.config._display_state.max_pixel), (3, 252))
    self.assertAlmostEqual(self.config._low_thresh, np.percentile(image, 3))
    self.assertAlmostEqual(self.config._high_thresh, np.percentile(image, 97))

    self.config._display_state.auto_range = False
    self.config._cast_img(image)
    np.testing.assert_array_equal(self.config._img, image)
