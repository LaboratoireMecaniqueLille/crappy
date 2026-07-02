# coding: utf-8

from unittest import TestCase

from crappy.modifier import Median


class TestMedian(TestCase):
  """Unit tests for the Median Modifier."""

  def test_n_points_must_be_positive(self) -> None:
    """Checks validation of the median period."""

    with self.assertRaises(ValueError):
      Median(0)
    with self.assertRaises(ValueError):
      Median(-1)

  def test_n_points_one_returns_each_point_as_float(self) -> None:
    """Checks that a one-point median emits every received sample."""

    modifier = Median(1)

    self.assertEqual(modifier({'x': 1}), {'x': 1.0})
    self.assertEqual(modifier({'x': 2}), {'x': 2.0})

  def test_returns_none_until_buffer_is_full_then_clears(self) -> None:
    """Checks regular median calculation and buffer reset."""

    modifier = Median(3)

    self.assertIsNone(modifier({'x': 1}))
    self.assertIsNone(modifier({'x': 9}))
    self.assertEqual(modifier({'x': 2}), {'x': 2.0})
    self.assertEqual(modifier._buf, {'x': []})
    self.assertIsNone(modifier({'x': 10}))

  def test_calculates_median_for_all_labels_independently(self) -> None:
    """Checks multi-label median calculation."""

    modifier = Median(2)
    modifier({'x': 1, 'y': 10})

    self.assertEqual(modifier({'x': 3, 'y': 20}),
                     {'x': 2.0, 'y': 15.0})

  def test_non_numeric_values_fall_back_to_latest_value(self) -> None:
    """Checks fallback behavior for labels that cannot be sorted."""

    modifier = Median(2)
    modifier({'x': 'first'})

    self.assertEqual(modifier({'x': 'second'}), {'x': 'second'})
