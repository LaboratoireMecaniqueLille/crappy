# coding: utf-8

from unittest import TestCase

from crappy.modifier import MovingMed


class TestMovingMed(TestCase):
  """Unit tests for the MovingMed Modifier."""

  def test_n_points_must_be_positive(self) -> None:
    """Checks validation of the moving median window."""

    with self.assertRaises(ValueError):
      MovingMed(0)
    with self.assertRaises(ValueError):
      MovingMed(-1)

  def test_n_points_one_returns_latest_point_as_float(self) -> None:
    """Checks the one-point moving median."""

    modifier = MovingMed(1)

    self.assertEqual(modifier({'x': 1}), {'x': 1.0})
    self.assertEqual(modifier({'x': 3}), {'x': 3.0})
    self.assertEqual(modifier._buf, {'x': [3]})

  def test_returns_median_at_every_call_during_warmup(self) -> None:
    """Checks moving median before the window is full."""

    modifier = MovingMed(3)

    self.assertEqual(modifier({'x': 1}), {'x': 1.0})
    self.assertEqual(modifier({'x': 9}), {'x': 5.0})
    self.assertEqual(modifier({'x': 2}), {'x': 2.0})

  def test_trims_window_to_latest_values(self) -> None:
    """Checks moving window trimming."""

    modifier = MovingMed(3)

    for value in (1, 9, 2):
      modifier({'x': value})

    self.assertEqual(modifier({'x': 7}), {'x': 7.0})
    self.assertEqual(modifier._buf, {'x': [9, 2, 7]})

  def test_calculates_median_for_all_labels_independently(self) -> None:
    """Checks multi-label moving median calculation."""

    modifier = MovingMed(2)
    modifier({'x': 1, 'y': 10})

    self.assertEqual(modifier({'x': 3, 'y': 20}),
                     {'x': 2.0, 'y': 15.0})

  def test_non_numeric_values_fall_back_to_latest_value(self) -> None:
    """Checks fallback behavior for labels that cannot be sorted."""

    modifier = MovingMed(2)

    self.assertEqual(modifier({'x': 'first'}), {'x': 'first'})
    self.assertEqual(modifier({'x': 'second'}), {'x': 'second'})
