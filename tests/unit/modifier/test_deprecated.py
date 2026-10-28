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

  def test_retired_moving_avg_alias_rejects_construction(self) -> None:
    """The retired Moving_avg spelling rejects construction."""

    with self.assertRaises(NotImplementedError):
      Moving_avg()

  def test_retired_moving_med_alias_rejects_construction(self) -> None:
    """The retired Moving_med spelling rejects construction."""

    with self.assertRaises(NotImplementedError):
      Moving_med()

  def test_retired_trig_on_change_alias_rejects_construction(self) -> None:
    """The retired Trig_on_change spelling rejects construction."""

    with self.assertRaises(NotImplementedError):
      Trig_on_change()

  def test_retired_trig_on_value_alias_rejects_construction(self) -> None:
    """The retired Trig_on_value spelling rejects construction."""

    with self.assertRaises(NotImplementedError):
      Trig_on_value()
