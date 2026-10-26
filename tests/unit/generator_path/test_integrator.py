# coding: utf-8

from unittest import TestCase

from crappy.blocks.generator_path.integrator import Integrator
from crappy.blocks.generator_path.meta_path import Path


class TestIntegrator(TestCase):
  """Unit tests for the Integrator Generator Path."""

  def setUp(self) -> None:
    Path.t0 = 0
    Path.last_cmd = None

  def test_first_integrator_requires_initial_value(self) -> None:
    """Checks that the first path cannot infer a previous command."""

    with self.assertRaises(ValueError):
      Integrator(condition=None, inertia=1, func_label='f')

  def test_inertia_must_be_non_zero(self) -> None:
    """Checks inertia validation."""

    with self.assertRaises(ValueError):
      Integrator(condition=None,
                 inertia=0,
                 func_label='f',
                 init_value=0)

  def test_integrator_uses_last_command_when_initial_value_is_omitted(
      self) -> None:
    """Checks implicit initial value inheritance."""

    Path.last_cmd = 5
    path = Integrator(condition=None, inertia=2, func_label='f')

    self.assertEqual(path.get_cmd({}), 5)

  def test_integrator_accumulates_trapezoidal_integral(self) -> None:
    """Checks integration across consecutive data batches."""

    path = Integrator(condition=None,
                      inertia=2,
                      func_label='f',
                      init_value=1)

    self.assertEqual(path.get_cmd({'t(s)': [0, 1], 'f': [2, 2]}), 2)
    self.assertEqual(path.get_cmd({'t(s)': [2], 'f': [4]}), 3.5)

  def test_integrator_ignores_missing_labels(self) -> None:
    """Checks that incomplete data keeps the current value."""

    path = Integrator(condition=None,
                      inertia=2,
                      func_label='f',
                      init_value=1)

    self.assertEqual(path.get_cmd({'t(s)': [0, 1]}), 1)
    self.assertEqual(path.get_cmd({'f': [2, 2]}), 1)

  def test_integrator_stops_when_condition_is_met(self) -> None:
    """Checks transition on condition."""

    path = Integrator(condition='stop>0',
                      inertia=1,
                      func_label='f',
                      init_value=0)

    with self.assertRaises(StopIteration):
      path.get_cmd({'stop': [1], 't(s)': [0, 1], 'f': [1, 1]})
