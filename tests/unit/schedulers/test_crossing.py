# coding: utf-8

from unittest import TestCase

from crappy.blocks.schedulers.conditions import Crossing


class TestCrossing(TestCase):
  """Directional threshold crossings and history reset."""

  def test_rising_crossing_across_calls_and_rearming(self) -> None:
    """A strictly below sample must precede a strictly above sample."""

    crossing = Crossing('x', 5, 'rising')
    self.assertFalse(crossing(0, {}))
    self.assertFalse(crossing(0, {'x': []}))
    self.assertFalse(crossing(0, {'x': [5, 7]}))
    self.assertFalse(crossing(0, {'x': [4, 5]}))
    self.assertTrue(crossing(0, {'x': [6]}))
    self.assertFalse(crossing(0, {'x': [7]}))
    self.assertTrue(crossing(0, {'x': [4, 6]}))

  def test_falling_crossing_and_reset(self) -> None:
    """Reset forgets the earlier side observation."""

    crossing = Crossing('x', 5, 'falling')
    self.assertFalse(crossing(0, {'x': [6]}))
    crossing.reset()
    self.assertFalse(crossing(0, {'x': [4]}))
    self.assertTrue(crossing(0, {'x': [6, 4]}))
    self.assertFalse(crossing(0, {'x': [3]}))

  def test_constructor_validation(self) -> None:
    """Labels, finite thresholds and directions are required."""

    for kwargs, error in (({'label': ''}, ValueError),
                          ({'label': 1}, TypeError),
                          ({'threshold': float('nan')}, ValueError),
                          ({'threshold': float('inf')}, ValueError),
                          ({'threshold': '5'}, TypeError),
                          ({'direction': 'up'}, ValueError),
                          ({'direction': 1}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Crossing(**{'label': 'x', 'threshold': 5,
                    'direction': 'rising', **kwargs})
