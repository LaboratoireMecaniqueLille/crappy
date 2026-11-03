# coding: utf-8

from math import pi
from unittest import TestCase

from crappy.blocks.schedulers.outputs import (Constant, Ramp, Triangle,
                                              Square, Sine)


class TestSimpleOutputs(TestCase):
  """Waveform generation and argument validation for Scheduler outputs."""

  def test_all_outputs_validate_label(self) -> None:
    """Every predefined output requires a non-empty string label."""

    factories = (lambda label: Constant(label, 1),
                 lambda label: Ramp(label, 0, 1),
                 lambda label: Triangle(label, 1, 2),
                 lambda label: Square(label, 2, 0, 2),
                 lambda label: Sine(label, 1, 1, 0))
    for factory in factories:
      for label, error in (('', ValueError), ('  ', ValueError), (1, TypeError)):
        with self.subTest(factory=factory, label=label):
          with self.assertRaises(error):
            factory(label)

  def test_constant_preserves_any_value(self) -> None:
    """The configured object is returned under its label unchanged."""

    value = object()
    output = Constant('command', value)
    self.assertIs(output(0, {})['command'], value)
    self.assertIs(output(10, {'input': [2]})['command'], value)

  def test_ramp_linear_values_and_validation(self) -> None:
    """A signed slope applies to elapsed State time."""

    output = Ramp('command', 3, -2)
    self.assertEqual(output(0, {}), {'command': 3})
    self.assertEqual(output(1.5, {}), {'command': 0})

    for kwargs, error in (({'init_value': float('nan')}, ValueError),
                          ({'init_value': float('inf')}, ValueError),
                          ({'init_value': '0'}, TypeError),
                          ({'slope': float('-inf')}, ValueError),
                          ({'slope': '1'}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Ramp('command', **{'init_value': 0, 'slope': 1, **kwargs})

  def test_triangle_phase_duty_cycle_and_validation(self) -> None:
    """The waveform reaches its extrema and repeats after a period."""

    output = Triangle('command', amplitude=2, period=4, offset=3,
                      duty_cycle=0.25)
    for dt, expected in ((0, 1), (0.5, 3), (1, 5), (2.5, 3),
                         (4, 1), (5, 5)):
      with self.subTest(dt=dt):
        self.assertAlmostEqual(output(dt, {})['command'], expected)

    for kwargs, error in (({'amplitude': 0}, ValueError),
                          ({'amplitude': float('inf')}, ValueError),
                          ({'period': -1}, ValueError),
                          ({'period': '2'}, TypeError),
                          ({'offset': float('nan')}, ValueError),
                          ({'duty_cycle': 0}, ValueError),
                          ({'duty_cycle': 1}, ValueError),
                          ({'duty_cycle': '0.5'}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Triangle('command', **{'amplitude': 2, 'period': 4, **kwargs})

  def test_square_levels_boundaries_and_validation(self) -> None:
    """Both duty-cycle extremes and the switch point behave as documented."""

    output = Square('command', high=8, low=-2, period=4,
                    duty_cycle=0.25)
    for dt, expected in ((0, 8), (0.999, 8), (1, -2),
                         (4, 8), (5, -2)):
      with self.subTest(dt=dt):
        self.assertEqual(output(dt, {})['command'], expected)
    self.assertEqual(Square('x', 8, -2, 4, duty_cycle=0)(0, {}), {'x': -2})
    self.assertEqual(Square('x', 8, -2, 4, duty_cycle=1)(2, {}), {'x': 8})

    for kwargs, error in (({'high': float('nan')}, ValueError),
                          ({'low': float('inf')}, ValueError),
                          ({'low': '0'}, TypeError),
                          ({'period': 0}, ValueError),
                          ({'period': '4'}, TypeError),
                          ({'duty_cycle': -0.1}, ValueError),
                          ({'duty_cycle': 1.1}, ValueError),
                          ({'duty_cycle': '0.5'}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Square('x', **{'high': 8, 'low': 0, 'period': 4, **kwargs})

  def test_sine_phase_offset_and_validation(self) -> None:
    """Frequency and phase are applied in radians."""

    output = Sine('command', amplitude=2, frequency=0.5, offset=3,
                  phase=pi / 2)
    for dt, expected in ((0, 5), (0.5, 3), (1, 1), (2, 5)):
      with self.subTest(dt=dt):
        self.assertAlmostEqual(output(dt, {})['command'], expected)

    for kwargs, error in (({'amplitude': 0}, ValueError),
                          ({'amplitude': '1'}, TypeError),
                          ({'frequency': 0}, ValueError),
                          ({'frequency': float('inf')}, ValueError),
                          ({'offset': float('nan')}, ValueError),
                          ({'phase': float('-inf')}, ValueError),
                          ({'phase': '0'}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        Sine('command', **{'amplitude': 2, 'frequency': 0.5,
                           'offset': 3, **kwargs})
