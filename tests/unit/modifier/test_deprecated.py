# coding: utf-8

from unittest import TestCase

from crappy.modifier import Moving_avg, Moving_med
from crappy.modifier import Trig_on_change, Trig_on_value
from crappy.modifier import Modifier


class TestDeprecatedModifiers(TestCase):
  """Unit tests for deprecated Modifier aliases."""

  def test_deprecated_aliases_are_registered(self) -> None:
    """Checks that legacy names remain visible in the registry."""

    self.assertIs(Modifier.classes['Moving_avg'], Moving_avg)
    self.assertIs(Modifier.classes['Moving_med'], Moving_med)
    self.assertIs(Modifier.classes['Trig_on_change'], Trig_on_change)
    self.assertIs(Modifier.classes['Trig_on_value'], Trig_on_value)

  def test_moving_avg_alias_raises_with_new_name(self) -> None:
    """Checks the Moving_avg deprecation message."""

    with self.assertRaisesRegex(NotImplementedError, 'MovingAvg'):
      Moving_avg()

  def test_moving_med_alias_raises_with_new_name(self) -> None:
    """Checks the Moving_med deprecation message."""

    with self.assertRaisesRegex(NotImplementedError, 'MovingMed'):
      Moving_med()

  def test_trig_on_change_alias_raises_with_new_name(self) -> None:
    """Checks the Trig_on_change deprecation message."""

    with self.assertRaisesRegex(NotImplementedError, 'TrigOnChange'):
      Trig_on_change()

  def test_trig_on_value_alias_raises_with_new_name(self) -> None:
    """Checks the Trig_on_value deprecation message."""

    with self.assertRaisesRegex(NotImplementedError, 'TrigOnValue'):
      Trig_on_value()
