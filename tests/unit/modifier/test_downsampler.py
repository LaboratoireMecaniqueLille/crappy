# coding: utf-8

from unittest import TestCase

from crappy.modifier import DownSampler


class TestDownSampler(TestCase):
  """Unit tests for the DownSampler Modifier."""

  def test_n_points_must_be_positive(self) -> None:
    """Checks validation of the downsampling period."""

    with self.assertRaises(ValueError):
      DownSampler(0)
    with self.assertRaises(ValueError):
      DownSampler(-1)

  def test_n_points_one_returns_every_point(self) -> None:
    """Checks the pass-through case."""

    modifier = DownSampler(1)
    data_1 = {'x': 1}
    data_2 = {'x': 2}

    self.assertIs(modifier(data_1), data_1)
    self.assertIs(modifier(data_2), data_2)

  def test_returns_first_point_and_then_every_n_points(self) -> None:
    """Checks regular downsampling behavior."""

    modifier = DownSampler(3)
    data = [{'x': i} for i in range(5)]

    self.assertIs(modifier(data[0]), data[0])
    self.assertIsNone(modifier(data[1]))
    self.assertIsNone(modifier(data[2]))
    self.assertIs(modifier(data[3]), data[3])
    self.assertIsNone(modifier(data[4]))
