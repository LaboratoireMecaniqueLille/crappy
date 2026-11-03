# coding: utf-8

import logging
from tests.support.generator_paths import GeneratorPathTestBase
from unittest.mock import patch

import crappy.blocks.generator_path as generator_path_package
import crappy.blocks.generator_path.meta_path.path as path_module
from crappy._exceptions import DefinitionError
from crappy.blocks.generator_path.meta_path import Path


Path.classes.pop('UnitTestPath', None)


class UnitTestPath(Path):
  """Concrete Path used for base class tests."""

  def get_cmd(self, data):
    return Path.get_cmd(self, data)


class TestPath(GeneratorPathTestBase):
  """Unit tests for the base Generator Path class."""

  path_t0 = 0

  def tearDown(self) -> None:
    """Removes dynamic classes registered during tests."""

    for name in ('RegisteredGeneratorPath', 'DuplicateGeneratorPath'):
      Path.classes.pop(name, None)

  def test_base_path_is_abstract(self) -> None:
    """Checks that Path cannot be instantiated directly."""

    with self.assertRaises(TypeError):
      Path()

  def test_subclasses_are_registered_by_name(self) -> None:
    """Checks Path subclass registration."""

    Path.classes.pop('RegisteredGeneratorPath', None)

    cls = type('RegisteredGeneratorPath',
               (Path,),
               {'get_cmd': lambda self, data: 0})

    self.assertIs(Path.classes['RegisteredGeneratorPath'], cls)

  def test_duplicate_subclass_names_are_rejected(self) -> None:
    """Checks that two Paths cannot share a class name."""

    Path.classes.pop('DuplicateGeneratorPath', None)
    type('DuplicateGeneratorPath',
         (Path,),
         {'get_cmd': lambda self, data: 0})

    with self.assertRaises(DefinitionError):
      type('DuplicateGeneratorPath',
           (Path,),
           {'get_cmd': lambda self, data: 0})

  def test_init_accepts_unused_arguments(self) -> None:
    """Checks base initialization with ignored arguments."""

    path = UnitTestPath('unused', option='ignored')

    self.assertFalse(path.parse_condition(None)({}))

  def test_log_forwards_levels_and_messages(self) -> None:
    """User messages retain their severity on first and subsequent calls."""

    path = UnitTestPath()
    with self.assertLogs(level='INFO') as captured:
      path.log(logging.INFO, 'message')
      path.log(logging.WARNING, 'follow-up')
    records = [(record.levelno, record.getMessage()) for record in captured.records]
    self.assertIn((logging.INFO, 'message'), records)
    self.assertIn((logging.WARNING, 'follow-up'), records)

  def test_default_get_cmd_warns_waits_and_returns_last_command(self) -> None:
    """An undefined output warns, backs off, and preserves the last command."""

    Path.last_cmd = 12
    path = UnitTestPath()
    with (patch.object(path_module, 'sleep') as sleep_mock,
          patch.object(path, 'log') as log_mock):
      self.assertEqual(path.get_cmd({}), 12)
    self.assertTrue(any(call.args[0] == logging.WARNING
                        for call in log_mock.call_args_list))
    self.assertTrue(any(call.args[0] > 0 for call in sleep_mock.call_args_list))

  def test_parse_none_condition_always_returns_false(self) -> None:
    """Checks the None condition helper."""

    condition = UnitTestPath().parse_condition(None)

    self.assertFalse(condition({}))
    self.assertFalse(condition({'x': [1]}))

  def test_parse_callable_condition_returns_it_unchanged(self) -> None:
    """Checks custom callable conditions."""

    def condition(data):
      return data.get('x') == [1]

    self.assertIs(UnitTestPath().parse_condition(condition), condition)

  def test_parse_lower_than_condition(self) -> None:
    """Checks less-than condition parsing and any-value semantics."""

    condition = UnitTestPath().parse_condition(' value < 3 ')

    self.assertTrue(condition({'value': [5, 2]}))
    self.assertFalse(condition({'value': [3, 4]}))
    self.assertFalse(condition({'other': [1]}))

  def test_parse_greater_than_condition(self) -> None:
    """Checks greater-than condition parsing and any-value semantics."""

    condition = UnitTestPath().parse_condition(' value > 3 ')

    self.assertTrue(condition({'value': [1, 4]}))
    self.assertFalse(condition({'value': [2, 3]}))
    self.assertFalse(condition({}))

  def test_parse_delay_condition(self) -> None:
    """Checks delay condition parsing against Path.t0."""

    Path.t0 = 10
    condition = UnitTestPath().parse_condition('delay = 2')

    with patch.object(path_module, 'time', return_value=12):
      self.assertFalse(condition({}))
    with patch.object(path_module, 'time', return_value=12.1):
      self.assertTrue(condition({}))

  def test_parse_invalid_conditions(self) -> None:
    """Checks validation of malformed conditions."""

    path = UnitTestPath()

    for condition in ('x = 1', 'x < bad', 'x > bad', 'delay = bad'):
      with self.subTest(condition=condition):
        with self.assertRaises((TypeError, ValueError)):
          path.parse_condition(condition)

  def test_package_registry_is_base_registry(self) -> None:
    """Checks package-level exports for all bundled Path classes."""

    self.assertIs(generator_path_package.meta_path.paths_dict, Path.classes)
    for name in ('Conditional', 'Constant', 'Custom', 'Cyclic',
                 'CyclicRamp', 'Integrator', 'Ramp', 'Sine'):
      self.assertIs(Path.classes[name],
                    getattr(generator_path_package, name))
