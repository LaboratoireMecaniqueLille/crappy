# coding: utf-8

from unittest import TestCase

from crappy.modifier import TrigOnChange


class TestTrigOnChange(TestCase):
  """Unit tests for the TrigOnChange Modifier."""

  def test_first_value_is_returned(self) -> None:
    """Checks first received data is transmitted."""

    modifier = TrigOnChange('x')
    data = {'x': 1}

    self.assertIs(modifier(data), data)
    self.assertTrue(modifier._initialized)
    self.assertEqual(modifier._last, 1)

  def test_same_value_is_filtered_and_changed_value_is_returned(self) -> None:
    """Checks value-change filtering."""

    modifier = TrigOnChange('x')
    first = {'x': 1}
    same = {'x': 1}
    changed = {'x': 2}

    self.assertIs(modifier(first), first)
    self.assertIsNone(modifier(same))
    self.assertIs(modifier(changed), changed)
    self.assertEqual(modifier._last, 2)

  def test_none_value_is_not_used_as_initialization_sentinel(self) -> None:
    """Checks that None can be a legitimate monitored value."""

    modifier = TrigOnChange('x')
    first = {'x': None}
    same = {'x': None}
    changed = {'x': 1}
    back_to_none = {'x': None}

    self.assertIs(modifier(first), first)
    self.assertIsNone(modifier(same))
    self.assertIs(modifier(changed), changed)
    self.assertIs(modifier(back_to_none), back_to_none)
