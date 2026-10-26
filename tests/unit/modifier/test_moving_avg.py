# coding: utf-8

from unittest import TestCase

from crappy.modifier import MovingAvg


class TestMovingAvg(TestCase):
  """Unit tests for the MovingAvg Modifier."""

  def test_n_points_must_be_positive(self) -> None:
    """Checks validation of the moving average window."""

    with self.assertRaises(ValueError):
      MovingAvg(0)
    with self.assertRaises(ValueError):
      MovingAvg(-1)

  def test_n_points_one_returns_latest_point_as_float(self) -> None:
    """Checks the one-point moving average."""

    modifier = MovingAvg(1)

    self.assertEqual(modifier({'x': 1}), {'x': 1.0})
    self.assertEqual(modifier({'x': 3}), {'x': 3.0})
    self.assertEqual(modifier._buf, {'x': [3]})

  def test_returns_average_at_every_call_during_warmup(self) -> None:
    """Checks moving average before the window is full."""

    modifier = MovingAvg(3)

    self.assertEqual(modifier({'x': 1}), {'x': 1.0})
    self.assertEqual(modifier({'x': 3}), {'x': 2.0})
    self.assertEqual(modifier({'x': 5}), {'x': 3.0})

  def test_trims_window_to_latest_values(self) -> None:
    """Checks moving window trimming."""

    modifier = MovingAvg(3)

    for value in (1, 3, 5):
      modifier({'x': value})

    self.assertEqual(modifier({'x': 7}), {'x': 5.0})
    self.assertEqual(modifier._buf, {'x': [3, 5, 7]})

  def test_averages_all_labels_independently(self) -> None:
    """Checks multi-label moving average calculation."""

    modifier = MovingAvg(2)
    modifier({'x': 1, 'y': 10})

    self.assertEqual(modifier({'x': 3, 'y': 20}),
                     {'x': 2.0, 'y': 15.0})

  def test_non_numeric_values_fall_back_to_latest_value(self) -> None:
    """Checks fallback behavior for labels that cannot be averaged."""

    modifier = MovingAvg(2)

    self.assertEqual(modifier({'x': 'first'}), {'x': 'first'})
    self.assertEqual(modifier({'x': 'second'}), {'x': 'second'})
