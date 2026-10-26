# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.generator_path.cyclic as cyclic_module
import crappy.blocks.generator_path.cyclic_ramp as cyclic_ramp_module
from crappy.blocks.generator_path.cyclic import Cyclic
from crappy.blocks.generator_path.cyclic_ramp import CyclicRamp
from crappy.blocks.generator_path.meta_path import Path


class PathTestMixin:
  """Shared setup for cyclic Path tests."""

  def setUp(self) -> None:
    Path.t0 = 10
    Path.last_cmd = None


class TestCyclic(PathTestMixin, TestCase):
  """Unit tests for the Cyclic Generator Path."""

  def test_cycles_must_be_full_or_half_cycles(self) -> None:
    """Checks cycle-count validation."""

    with self.assertRaises(ValueError):
      Cyclic(condition1=None,
             condition2=None,
             value1=1,
             value2=2,
             cycles=1.25)

  def test_cyclic_alternates_values_and_then_stops(self) -> None:
    """Checks finite cycle transitions."""

    path = Cyclic(condition1='switch>0',
                  condition2='switch<0',
                  value1=1,
                  value2=2,
                  cycles=1)

    self.assertEqual(path.get_cmd({}), 1)
    self.assertEqual(path.get_cmd({'switch': [1]}), 2)

    with self.assertRaises(StopIteration):
      path.get_cmd({'switch': [-1]})

  def test_half_cycle_stops_before_second_value(self) -> None:
    """Checks half-cycle exhaustion."""

    path = Cyclic(condition1='switch>0',
                  condition2='switch<0',
                  value1=1,
                  value2=2,
                  cycles=0.5)

    self.assertEqual(path.get_cmd({}), 1)

    with self.assertRaises(StopIteration):
      path.get_cmd({'switch': [1]})

  def test_zero_or_negative_cycles_loop_forever(self) -> None:
    """Checks documented endless-cycle behavior."""

    for cycles in (0, -1):
      with self.subTest(cycles=cycles):
        path = Cyclic(condition1=lambda _: True,
                      condition2=lambda _: True,
                      value1=1,
                      value2=2,
                      cycles=cycles)

        self.assertEqual([path.get_cmd({}) for _ in range(4)],
                         [2, 1, 2, 1])

  def test_cyclic_resets_phase_time_on_transition(self) -> None:
    """Checks that phase t0 is updated when conditions switch."""

    path = Cyclic(condition1='switch>0',
                  condition2='switch<0',
                  value1=1,
                  value2=2,
                  cycles=1)
    path.get_cmd({})

    with patch.object(cyclic_module, 'time', return_value=42):
      self.assertEqual(path.get_cmd({'switch': [1]}), 2)

    self.assertEqual(path.t0, 42)


class TestCyclicRamp(PathTestMixin, TestCase):
  """Unit tests for the CyclicRamp Generator Path."""

  def test_first_cyclic_ramp_requires_initial_value(self) -> None:
    """Checks that the first path cannot infer a previous command."""

    with self.assertRaises(ValueError):
      CyclicRamp(condition1=None,
                 condition2=None,
                 speed1=1,
                 speed2=-1)

  def test_cycles_must_be_full_or_half_cycles(self) -> None:
    """Checks cycle-count validation."""

    with self.assertRaises(ValueError):
      CyclicRamp(condition1=None,
                 condition2=None,
                 speed1=1,
                 speed2=-1,
                 cycles=1.25,
                 init_value=0)

  def test_cyclic_ramp_alternates_slopes_and_then_stops(self) -> None:
    """Checks finite ramp transitions and value continuity."""

    path = CyclicRamp(condition1='switch>0',
                      condition2='switch<0',
                      speed1=2,
                      speed2=-1,
                      cycles=1,
                      init_value=0)

    with patch.object(cyclic_ramp_module, 'time', return_value=11):
      self.assertEqual(path.get_cmd({}), 2)

    with patch.object(cyclic_ramp_module,
                      'time',
                      side_effect=(12, 13)):
      self.assertEqual(path.get_cmd({'switch': [1]}), 3)

    with patch.object(cyclic_ramp_module, 'time', return_value=14):
      with self.assertRaises(StopIteration):
        path.get_cmd({'switch': [-1]})

  def test_cyclic_ramp_uses_last_command_when_initial_value_is_omitted(
      self) -> None:
    """Checks implicit initial value inheritance."""

    Path.last_cmd = 5
    path = CyclicRamp(condition1=None,
                      condition2=None,
                      speed1=2,
                      speed2=-1)

    with patch.object(cyclic_ramp_module, 'time', return_value=11):
      self.assertEqual(path.get_cmd({}), 7)
