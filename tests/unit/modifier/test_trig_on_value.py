# coding: utf-8

from unittest import TestCase

from crappy.modifier import TrigOnValue


class TestTrigOnValue(TestCase):
  """Unit tests for the TrigOnValue Modifier."""

  def test_single_string_value_is_not_split_into_characters(self) -> None:
    """Checks constructor normalization for strings."""

    modifier = TrigOnValue('state', 'ok')
    accepted = {'state': 'ok'}

    self.assertEqual(modifier._values, ('ok',))
    self.assertIs(modifier(accepted), accepted)
    self.assertIsNone(modifier({'state': 'o'}))

  def test_single_non_iterable_value_is_wrapped(self) -> None:
    """Checks constructor normalization for scalar values."""

    modifier = TrigOnValue('state', 1)
    accepted = {'state': 1}

    self.assertEqual(modifier._values, (1,))
    self.assertIs(modifier(accepted), accepted)
    self.assertIsNone(modifier({'state': 2}))

  def test_iterable_values_are_accepted(self) -> None:
    """Checks filtering against several accepted values."""

    modifier = TrigOnValue('state', (1, 3))
    first = {'state': 1}
    second = {'state': 3}

    self.assertEqual(modifier._values, (1, 3))
    self.assertIs(modifier(first), first)
    self.assertIsNone(modifier({'state': 2}))
    self.assertIs(modifier(second), second)
