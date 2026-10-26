# coding: utf-8

from unittest import TestCase

from crappy.modifier import Integrate


class TestIntegrate(TestCase):
  """Unit tests for the Integrate Modifier."""

  def test_first_point_initializes_state_and_returns_zero(self) -> None:
    """Checks first integration output and stored state."""

    modifier = Integrate('x')
    data = {'t(s)': 1, 'x': 3}

    returned = modifier(data)

    self.assertIs(returned, data)
    self.assertEqual(returned, {'t(s)': 1, 'x': 3, 'i_x': 0})
    self.assertEqual(modifier._last_t, 1)
    self.assertEqual(modifier._last_val, 3)
    self.assertEqual(modifier._integration, 0)

  def test_following_points_return_trapezoidal_integral(self) -> None:
    """Checks integration with default labels."""

    modifier = Integrate('x')
    modifier({'t(s)': 0, 'x': 2})

    first = modifier({'t(s)': 2, 'x': 4})
    second = modifier({'t(s)': 3, 'x': 6})

    self.assertEqual(first, {'t(s)': 2, 'x': 4, 'i_x': 6})
    self.assertEqual(second, {'t(s)': 3, 'x': 6, 'i_x': 11})

  def test_custom_time_and_output_labels_are_supported(self) -> None:
    """Checks integration with custom labels."""

    modifier = Integrate(label='speed',
                         time_label='time',
                         out_label='position')
    modifier({'time': 0, 'speed': 1})

    returned = modifier({'time': 4, 'speed': 3})

    self.assertEqual(returned, {'time': 4, 'speed': 3, 'position': 8})

  def test_repeated_timestamp_updates_state_without_changing_integral(self):
    """Checks zero-duration integration."""

    modifier = Integrate('x')
    modifier({'t(s)': 1, 'x': 3})

    returned = modifier({'t(s)': 1, 'x': 9})

    self.assertEqual(returned, {'t(s)': 1, 'x': 9, 'i_x': 0})
    self.assertEqual(modifier._last_t, 1)
    self.assertEqual(modifier._last_val, 9)
    self.assertEqual(modifier._integration, 0)
