# coding: utf-8

from unittest import TestCase

from crappy.modifier import Diff


class TestDiff(TestCase):
  """Unit tests for the Diff Modifier."""

  def test_first_point_initializes_state_and_returns_zero(self) -> None:
    """Checks first derivative output and stored state."""

    modifier = Diff('x')
    data = {'t(s)': 1, 'x': 3}

    returned = modifier(data)

    self.assertIs(returned, data)
    self.assertEqual(returned, {'t(s)': 1, 'x': 3, 'd_x': 0})
    self.assertEqual(modifier._last_t, 1)
    self.assertEqual(modifier._last_val, 3)

  def test_following_points_return_derivative(self) -> None:
    """Checks derivative calculation with default labels."""

    modifier = Diff('x')
    modifier({'t(s)': 1, 'x': 3})

    returned = modifier({'t(s)': 3, 'x': 9})

    self.assertEqual(returned, {'t(s)': 3, 'x': 9, 'd_x': 3})
    self.assertEqual(modifier._last_t, 3)
    self.assertEqual(modifier._last_val, 9)

  def test_custom_time_and_output_labels_are_supported(self) -> None:
    """Checks derivative calculation with custom labels."""

    modifier = Diff(label='position', time_label='time', out_label='speed')
    modifier({'time': 0, 'position': 1})

    returned = modifier({'time': 4, 'position': 9})

    self.assertEqual(returned, {'time': 4, 'position': 9, 'speed': 2})

  def test_repeated_timestamp_returns_none_and_keeps_previous_state(self):
    """Checks zero-division avoidance."""

    modifier = Diff('x')
    modifier({'t(s)': 1, 'x': 3})

    self.assertIsNone(modifier({'t(s)': 1, 'x': 9}))
    self.assertEqual(modifier._last_t, 1)
    self.assertEqual(modifier._last_val, 3)

    returned = modifier({'t(s)': 3, 'x': 11})

    self.assertEqual(returned['d_x'], 4)
