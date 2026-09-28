# coding: utf-8

from unittest import TestCase

from crappy.blocks.schedulers.conditions import (Condition, Delay, AllLabel,
                                                 AnyLabel, AllCondition,
                                                 AnyCondition)


class ProbeCondition(Condition):
  """Records evaluation and reset calls for condition composition."""

  def __init__(self, result: bool) -> None:
    super().__init__()
    self.result = result
    self.calls = []
    self.resets = 0

  def __call__(self, dt, data):
    self.calls.append((dt, data))
    return self.result

  def reset(self):
    self.resets += 1


class TestDelay(TestCase):
  """Elapsed-time transition tests."""

  def test_boundary_and_validation(self) -> None:
    """The condition becomes true exactly at the requested time."""

    delay = Delay(2)
    self.assertFalse(delay(1.999, {}))
    self.assertTrue(delay(2, {}))
    self.assertTrue(delay(3, {}))
    self.assertTrue(Delay(True)(1, {}))

    for value, error in ((0, ValueError), (-1, ValueError),
                         (float('inf'), ValueError),
                         (float('nan'), ValueError), ('2', TypeError)):
      with self.subTest(value=value), self.assertRaises(error):
        Delay(value)


class TestLabelConditions(TestCase):
  """Truthiness modes for all-label and any-label transitions."""

  def test_all_label_modes(self) -> None:
    """Every named label must have a matching non-empty value list."""

    data = {'a': [False, True], 'b': [True, True]}
    self.assertTrue(AllLabel('a', 'b')(0, data))
    self.assertFalse(AllLabel('a', 'b', mode='all')(0, data))
    self.assertTrue(AllLabel('a', 'b', mode='last')(0, data))
    self.assertFalse(AllLabel('a', 'b')(0, {'a': [True]}))
    self.assertFalse(AllLabel('a', 'b')(0, {'a': [True], 'b': []}))
    self.assertFalse(AllLabel('a', 'b', mode='last')(
      0, {'a': [True, False], 'b': [True]}))

  def test_any_label_modes_and_missing_values(self) -> None:
    """Missing or empty labels do not make another label fail."""

    data = {'a': [], 'b': [False, True]}
    snapshot = {'a': [], 'b': [False, True]}
    self.assertTrue(AnyLabel('a', 'b', 'missing')(0, data))
    self.assertFalse(AnyLabel('a', 'b', mode='all')(0, data))
    self.assertTrue(AnyLabel('a', 'b', mode='last')(0, data))
    self.assertFalse(AnyLabel('missing')(0, data))
    self.assertFalse(AnyLabel('a')(0, data))
    self.assertTrue(AnyLabel('a', 'b', mode='all')(
      0, {'a': [], 'b': [True, True]}))
    self.assertEqual(data, snapshot)

  def test_label_and_mode_validation(self) -> None:
    """Both combinators reject empty declarations and unknown modes."""

    for cls in (AllLabel, AnyLabel):
      for labels, mode, error in (((), 'any', ValueError),
                                  (('',), 'any', ValueError),
                                  ((1,), 'any', ValueError),
                                  (('a',), 'unknown', ValueError),
                                  (('a',), 1, TypeError)):
        with self.subTest(cls=cls, labels=labels, mode=mode):
          with self.assertRaises(error):
            cls(*labels, mode=mode)


class TestCombinedConditions(TestCase):
  """Short-circuit order and nested reset propagation."""

  def test_all_condition_short_circuits_and_resets(self) -> None:
    """A false condition prevents later evaluation, not later resets."""

    first, second = ProbeCondition(False), ProbeCondition(True)
    combined = AllCondition(first, second)
    data = {'a': [1]}
    self.assertFalse(combined(3, data))
    self.assertEqual(first.calls, [(3, data)])
    self.assertEqual(second.calls, [])
    first.result = True
    self.assertTrue(combined(4, data))
    combined.reset()
    self.assertEqual((first.resets, second.resets), (1, 1))

  def test_any_condition_short_circuits_and_accepts_callable(self) -> None:
    """A true result skips later conditions; plain functions are allowed."""

    first, second = ProbeCondition(True), ProbeCondition(False)
    combined = AnyCondition(first, second, lambda dt, data: dt > 3)
    self.assertTrue(combined(4, {}))
    self.assertEqual(len(first.calls), 1)
    self.assertEqual(second.calls, [])
    first.result = False
    self.assertTrue(combined(4, {}))
    self.assertEqual(len(second.calls), 1)
    combined.reset()
    self.assertEqual((first.resets, second.resets), (1, 1))

  def test_combinator_validation(self) -> None:
    """At least one callable condition is required."""

    for cls in (AllCondition, AnyCondition):
      with self.subTest(cls=cls), self.assertRaises(ValueError):
        cls()
      with self.subTest(cls=cls), self.assertRaises(TypeError):
        cls(lambda _dt, _data: True, 1)
