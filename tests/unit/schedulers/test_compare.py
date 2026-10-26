# coding: utf-8

from unittest import TestCase

from crappy.blocks.schedulers.conditions import Compare


class TestCompare(TestCase):
  """Comparison operators, value modes and argument validation."""

  def test_constant_operators(self) -> None:
    """Every supported operator uses the received value as its left side."""

    cases = (('<', 3, [2], True), ('>', 3, [2], False),
             ('<=', 3, [3], True), ('>=', 3, [2], False),
             ('==', 3, [3], True), ('!=', 3, [3], False),
             ('in', [2, 4], [2], True),
             ('not in', [2, 4], [3], True))
    for operation, value, received, expected in cases:
      with self.subTest(operation=operation):
        self.assertIs(Compare('x', operation, value=value)(
          0, {'x': received}), expected)

  def test_modes_with_constant_and_second_label(self) -> None:
    """Any, all and last use the intended values and Cartesian product."""

    values = {'x': [1, 4], 'y': [2, 3]}
    self.assertTrue(Compare('x', '<', value=3, mode='any')(0, values))
    self.assertFalse(Compare('x', '<', value=3, mode='all')(0, values))
    self.assertFalse(Compare('x', '<', value=3, mode='last')(0, values))
    self.assertTrue(Compare('x', '<', label_2='y', mode='any')(0, values))
    self.assertFalse(Compare('x', '<', label_2='y', mode='all')(0, values))
    self.assertFalse(Compare('x', '<', label_2='y', mode='last')(0, values))
    self.assertTrue(Compare('x', '<', label_2='y', mode='all')(
      0, {'x': [1, 2], 'y': [3, 4]}))

  def test_missing_and_empty_operands_are_false(self) -> None:
    """A comparison is unavailable until both operands have values."""

    constant = Compare('x', '==', value=1)
    labels = Compare('x', '==', label_2='y')
    for data in ({}, {'x': []}):
      with self.subTest(data=data):
        self.assertFalse(constant(0, data))
    for data in ({'x': [1]}, {'x': [1], 'y': []}):
      with self.subTest(data=data):
        self.assertFalse(labels(0, data))

  def test_constructor_rejects_invalid_combinations(self) -> None:
    """Operand choice, operation and mode are validated."""

    cases = (({'label_1': ''}, ValueError),
             ({'label_1': 1}, TypeError),
             ({'operation': '?'}, ValueError),
             ({'operation': 1}, TypeError),
             ({'label_2': ''}, ValueError),
             ({'label_2': 1}, TypeError),
             ({'mode': '?'}, ValueError),
             ({'mode': 1}, TypeError),
             ({'value': None}, ValueError),
             ({'value': 1, 'label_2': 'y'}, ValueError),
             ({'operation': 'in', 'value': None, 'label_2': 'y'}, ValueError),
             ({'operation': 'in', 'value': 1}, ValueError),
             ({'operation': '<', 'value': '1'}, ValueError),
             ({'operation': '==', 'value': []}, ValueError))
    for kwargs, error in cases:
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Compare(**{'label_1': 'x', 'operation': '<',
                   'value': 1, **kwargs})

  def test_membership_static_helpers(self) -> None:
    """The operators reverse Python's container-first API."""

    self.assertTrue(Compare.contains('a', {'a', 'b'}))
    self.assertFalse(Compare.contains('c', {'a', 'b'}))
    self.assertTrue(Compare.not_contains('c', {'a', 'b'}))
