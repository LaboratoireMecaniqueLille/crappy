# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.generator_path.ramp as ramp_module
import crappy.blocks.generator_path.sine as sine_module
from crappy.blocks.generator_path.constant import Constant
from crappy.blocks.generator_path.meta_path import Path
from crappy.blocks.generator_path.ramp import Ramp
from crappy.blocks.generator_path.sine import Sine


class PathTestMixin:
  """Shared setup for Path tests using class-level state."""

  def setUp(self) -> None:
    Path.t0 = 10
    Path.last_cmd = None


class TestConstant(PathTestMixin, TestCase):
  """Unit tests for the Constant Generator Path."""

  def test_first_constant_requires_a_value(self) -> None:
    """Checks that the first path cannot infer a previous command."""

    with self.assertRaises(ValueError):
      Constant(condition=None)

  def test_constant_uses_last_command_when_value_is_omitted(self) -> None:
    """Checks implicit value inheritance from the previous path."""

    Path.last_cmd = 7
    path = Constant(condition=None)

    self.assertEqual(path.get_cmd({}), 7)

  def test_constant_returns_value_until_condition_is_met(self) -> None:
    """Checks value emission and transition on condition."""

    condition_met = False

    def condition(_):
      return condition_met

    path = Constant(condition=condition, value=3)

    self.assertEqual(path.get_cmd({}), 3)
    condition_met = True
    with self.assertRaises(StopIteration):
      path.get_cmd({})


class TestRamp(PathTestMixin, TestCase):
  """Unit tests for the Ramp Generator Path."""

  def test_first_ramp_requires_an_initial_value(self) -> None:
    """Checks that the first path cannot infer a previous command."""

    with self.assertRaises(ValueError):
      Ramp(condition=None, speed=1)

  def test_ramp_uses_last_command_when_initial_value_is_omitted(self) -> None:
    """Checks implicit initial value inheritance."""

    Path.last_cmd = 4
    path = Ramp(condition=None, speed=2)

    with patch.object(ramp_module, 'time', return_value=12):
      self.assertEqual(path.get_cmd({}), 8)

  def test_ramp_returns_linear_signal_until_condition_is_met(self) -> None:
    """Checks ramp output and transition on condition."""

    condition_met = False

    def condition(_):
      return condition_met

    path = Ramp(condition=condition, speed=2, init_value=1)

    with patch.object(ramp_module, 'time', return_value=12):
      self.assertEqual(path.get_cmd({}), 5)

    condition_met = True
    with self.assertRaises(StopIteration):
      path.get_cmd({})


class TestSine(PathTestMixin, TestCase):
  """Unit tests for the Sine Generator Path."""

  def test_sine_uses_peak_to_peak_amplitude(self) -> None:
    """Checks sine value calculation."""

    path = Sine(condition=None,
                freq=0.25,
                amplitude=4,
                offset=10,
                phase=0)

    with patch.object(sine_module, 'time', return_value=10):
      self.assertEqual(path.get_cmd({}), 10)
    with patch.object(sine_module, 'time', return_value=11):
      self.assertAlmostEqual(path.get_cmd({}), 12)

  def test_sine_honors_phase(self) -> None:
    """Checks phase sign convention."""

    path = Sine(condition=None,
                freq=0.25,
                amplitude=4,
                offset=0,
                phase=sine_module.pi / 2)

    with patch.object(sine_module, 'time', return_value=10):
      self.assertAlmostEqual(path.get_cmd({}), -2)

  def test_sine_stops_when_condition_is_met(self) -> None:
    """Checks transition on condition."""

    path = Sine(condition=lambda _: True, freq=1, amplitude=2)

    with self.assertRaises(StopIteration):
      path.get_cmd({})
