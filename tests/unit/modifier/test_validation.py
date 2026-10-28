"""Public validation and protected state for this feature."""

from unittest import TestCase
import numpy as np
from crappy.modifier import (Demux, Diff, DownSampler, Integrate, Mean, Median,
                             MovingAvg, MovingMed, Offset, TrigOnChange, TrigOnValue)


class TestModifierValidation(TestCase):
  def test_window_modifiers_require_positive_integer_counts(self):
    for cls in (Mean, Median, MovingAvg, MovingMed, DownSampler):
      for value in (True, 1.5, '2', None, []):
        with self.subTest(cls=cls.__name__, value=value):
          with self.assertRaises((TypeError, ValueError)):
            cls(value)
      for value in (0, -1):
        with self.subTest(cls=cls.__name__, value=value):
          with self.assertRaises((TypeError, ValueError)):
            cls(value)
      self.assertEqual(cls(np.int64(1))({'x': 2}), {'x': 2})

  def test_label_modifiers_reject_invalid_labels(self):
    constructors = (lambda label: Diff(label), lambda label: Integrate(label),
                    lambda label: TrigOnChange(label),
                    lambda label: TrigOnValue(label, 1))
    for constructor in constructors:
      for value, error in ((1, TypeError), (None, TypeError),
                            ('', ValueError), ('  ', ValueError)):
        with self.subTest(constructor=constructor, value=value):
          with self.assertRaises(error):
            constructor(value)
    with self.assertRaises((TypeError, ValueError)):
      Diff('x', out_label=1)
    with self.assertRaises((TypeError, ValueError)):
      Integrate('x', time_label='')

  def test_offsets_keep_iterables_and_check_numeric_targets(self):
    modifier = Offset(iter(('x', 'y')), iter((1.0, -2.0)))
    self.assertEqual(modifier({'x': 5, 'y': 10}), {'x': 1, 'y': -2})
    for offsets, error in ((('1',), TypeError), ((np.nan,), ValueError),
                            ((np.inf,), ValueError)):
      with self.subTest(offsets=offsets):
        with self.assertRaises(error):
          Offset('x', offsets)
    with self.assertRaises((TypeError, ValueError)):
      Offset(('x', 'x'), (1, 2))

  def test_demux_rejects_bad_shapes_before_mutating_data(self):
    modifier = Demux(('x', 'y'))
    for stream in (np.ones(2), np.ones((2, 1))):
      data = {'stream': stream, 't(s)': np.array([1, 2])}
      with self.subTest(shape=stream.shape):
        with self.assertRaises((TypeError, ValueError)):
          modifier(data)
        self.assertEqual(set(data), {'stream', 't(s)'})
    with self.assertRaises((TypeError, ValueError)):
      Demux('x', mean=1)
