# coding: utf-8

from unittest import TestCase

from crappy.blocks.generator_path.conditional import Conditional
from crappy.blocks.generator_path.meta_path import Path


class TestConditional(TestCase):
  """Unit tests for the Conditional Generator Path."""

  def setUp(self) -> None:
    Path.t0 = 0
    Path.last_cmd = None

  def test_condition1_has_priority_over_condition2(self) -> None:
    """Checks value selection priority."""

    path = Conditional(condition1='a>0',
                       condition2='b>0',
                       value1=1,
                       value2=2,
                       value0=0)

    self.assertEqual(path.get_cmd({'a': [1], 'b': [1]}), 1)

  def test_condition2_is_used_when_condition1_is_false(self) -> None:
    """Checks second-condition value selection."""

    path = Conditional(condition1='a>0',
                       condition2='b>0',
                       value1=1,
                       value2=2,
                       value0=0)

    self.assertEqual(path.get_cmd({'a': [0], 'b': [1]}), 2)

  def test_value0_is_used_when_no_condition_matches(self) -> None:
    """Checks fallback value selection."""

    path = Conditional(condition1='a>0',
                       condition2='b>0',
                       value1=1,
                       value2=2,
                       value0=-1)

    self.assertEqual(path.get_cmd({'a': [0], 'b': [0]}), -1)

  def test_empty_data_keeps_previous_value_without_rechecking(self) -> None:
    """Checks the intended hold-last-value behavior on empty input."""

    calls = list()

    def condition(data):
      calls.append(data)
      return True

    path = Conditional(condition1=condition,
                       condition2=condition,
                       value1=1,
                       value2=2,
                       value0=0)

    self.assertEqual(path.get_cmd({}), 0)
    self.assertEqual(calls, [])

    self.assertEqual(path.get_cmd({'x': [1]}), 1)
    self.assertEqual(path.get_cmd({}), 1)
    self.assertEqual(calls, [{'x': [1]}])
