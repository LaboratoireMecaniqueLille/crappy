# coding: utf-8

import unittest
import itertools
import numpy as np

from crappy.tool.camera_config.config_tools import Box


class TestBox(unittest.TestCase):
  """Class for testing the :class:`~crappy.tool.camera_config.config_tools.Box`
  class.

  .. versionadded:: 2.0.8
  """

  def setUp(self) -> None:
    """Instantiates the Box to test."""

    self._box = Box()

  def test_01_points(self) -> None:
    """Tests whether the points can be set and their defaults."""

    # All attributes should be None
    self.assertIsNone(self._box.x_start)
    self.assertIsNone(self._box.y_start)
    self.assertIsNone(self._box.x_end)
    self.assertIsNone(self._box.y_end)
    self.assertIsNone(self._box.x_centroid)
    self.assertIsNone(self._box.y_centroid)
    self.assertIsNone(self._box.x_disp)
    self.assertIsNone(self._box.y_disp)

    # no_points() should be False once all points are defined
    self.assertTrue(self._box.no_points())
    self._box.x_start = 0
    self.assertTrue(self._box.no_points())
    self._box.y_start = 0
    self.assertTrue(self._box.no_points())
    self._box.x_end = 10
    self.assertTrue(self._box.no_points())
    self._box.y_end = 10
    self.assertFalse(self._box.no_points())

    # The points should properly reset
    self._box.reset()
    self.assertTrue(self._box.no_points())

    self.assertIsNone(self._box.x_start)
    self.assertIsNone(self._box.y_start)
    self.assertIsNone(self._box.x_end)
    self.assertIsNone(self._box.y_end)
    self.assertIsNone(self._box.x_centroid)
    self.assertIsNone(self._box.y_centroid)
    self.assertIsNone(self._box.x_disp)
    self.assertIsNone(self._box.y_disp)

  def test_02_sorted(self) -> None:
    """Tests the Box.sorted() method."""

    # Check that the correct sorted sequence is obtained no matter the input
    # sequence
    for ((x_start, x_end),
         (y_start, y_end)) in itertools.product(((0, 10), (10, 0)), repeat=2):
      with self.subTest(x_start=x_start,
                        y_start=y_start,
                        x_end=x_end,
                        y_end=y_end):
        self._box.x_start = x_start
        self._box.y_start = y_start
        self._box.x_end = x_end
        self._box.y_end = y_end

        self.assertTupleEqual(self._box.sorted(), (0, 10, 0, 10))

        self._box.reset()

  def test_03_draw(self) -> None:
    """Tests the Box.draw() method."""

    # Start from a blank image, should be fully blank
    img = np.zeros((100, 100), dtype=np.uint8)
    self._box.x_start = 25
    self._box.y_start = 25
    self._box.x_end = 75
    self._box.y_end = 75
    self.assertEqual(np.count_nonzero(img), 0)

    # Draw the box, the image should no longer be blank
    self._box.draw(img)
    self.assertEqual(np.count_nonzero(img), 4 * (50 - 1) + 4 * (48 - 1))

    self._box.reset()

    # Test again with larger image
    img = np.zeros((1000, 1000), dtype=np.uint8)
    self._box.x_start = 250
    self._box.y_start = 250
    self._box.x_end = 750
    self._box.y_end = 750
    self.assertEqual(np.count_nonzero(img), 0)

    self._box.draw(img)
    self.assertEqual(np.count_nonzero(img),
                     4 * (500 - 1) + 4 * (498 - 1) + 4 * (496 - 1))

    self._box.reset()

  def test_04_update(self) -> None:
    """Test the Box.update() method."""

    # Define a second box object, different from the first one
    other = Box()
    other.x_start = 25
    other.y_start = 25
    other.x_end = 75
    other.y_end = 75
    other.x_centroid = 50
    other.y_centroid = 50
    other.x_disp = 0
    other.y_disp = 0

    self.assertNotEqual(other, self._box)

    # After update, both boxes should be equal
    self._box.update(other)

    self.assertEqual(self._box.x_start, 25)
    self.assertEqual(self._box.y_start, 25)
    self.assertEqual(self._box.x_end, 75)
    self.assertEqual(self._box.y_end, 75)
    self.assertEqual(self._box.x_centroid, 50)
    self.assertEqual(self._box.y_centroid, 50)
    self.assertEqual(self._box.x_disp, 0)
    self.assertEqual(self._box.y_disp, 0)

    self.assertEqual(other, self._box)

