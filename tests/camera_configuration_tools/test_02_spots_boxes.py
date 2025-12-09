# coding: utf-8

import unittest

from crappy.tool.camera_config.config_tools import Box, SpotsBoxes


class TestSpotBoxes(unittest.TestCase):
  """Class for testing the 
  :class:`~crappy.tool.camera_config.config_tools.SpotsBoxes` class.

  .. versionadded:: 2.0.8
  """

  def setUp(self) -> None:
    """Instantiates the SpotsBoxes to test."""

    self._spots = SpotsBoxes()

  def test_01_spots(self) -> None:
    """Tests whether the spots can be correctly set and their defaults."""

    # At the beginning, all spots should be empty
    self.assertIsNone(self._spots.spot_1)
    self.assertIsNone(self._spots.spot_2)
    self.assertIsNone(self._spots.spot_3)
    self.assertIsNone(self._spots.spot_4)
    self.assertIsNone(self._spots.x_l0)
    self.assertIsNone(self._spots.y_l0)
    self.assertTrue(self._spots.empty())

    # When adding at least one box, the spots should no longer be empty
    box = Box()
    self._spots.spot_1 = box
    self.assertFalse(self._spots.empty())

    self._spots.spot_2 = box
    self._spots.spot_3 = box
    self._spots.spot_4 = box
    self._spots.x_l0 = 0.0
    self._spots.y_l0 = 0.0

    # Reset the spots
    self.assertFalse(self._spots.empty())
    self._spots.reset()

    # The spots should now be empty
    self.assertTrue(self._spots.empty())
    self.assertIsNone(self._spots.spot_1)
    self.assertIsNone(self._spots.spot_2)
    self.assertIsNone(self._spots.spot_3)
    self.assertIsNone(self._spots.spot_4)

    # But the l0 should have been preserved
    self.assertIsNotNone(self._spots.x_l0)
    self.assertIsNotNone(self._spots.y_l0)

  def test_02_set_spots(self) -> None:
    """Tests the SpotsBoxes.set_spots() method."""

    # Defining spots with a regular pattern
    spots = list(zip(range(0, 13, 4), range(1, 14, 4),
                     range(2, 15, 4), range(3, 16, 4)))
    self._spots.set_spots(spots)

    # Check if the pattern is kept after the set_spots call
    for i in range(4):
      with self.subTest(i=i):
        self.assertEqual(self._spots[i], Box(1 + 4 * i, 4 + 8 * i,
                                             4 * i, 2 + 8 * i))

  def test_03_save_length(self) -> None:
    """Tests the SpotsBoxes.save_length() method."""

    # Populate the spots
    spots = list(zip(range(0, 13, 4), range(1, 14, 4),
                     range(2, 15, 4), range(3, 16, 4)))
    self._spots.set_spots(spots)

    # Check that the centroids and length are not initialized
    self.assertTrue(all(spot.x_centroid is None for spot in self._spots))
    self.assertTrue(all(spot.y_centroid is None for spot in self._spots))
    self.assertIsNone(self._spots.x_l0)
    self.assertIsNone(self._spots.y_l0)

    self._spots.save_length()

    # The centroids and lengths should now be defined
    self.assertTrue(all(spot.x_centroid is not None for spot in self._spots))
    self.assertTrue(all(spot.y_centroid is not None for spot in self._spots))
    self.assertEqual(self._spots.x_l0, 18.0)
    self.assertEqual(self._spots.x_l0, 18.0)
