# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.generator_path as generator_path_package
import crappy.blocks.generator_path.meta_path.path as path_module
from crappy._global import DefinitionError
from crappy.blocks.generator_path.meta_path import Path


Path.classes.pop('UnitTestPath', None)


class UnitTestPath(Path):
  """Concrete Path used for base class tests."""

  def get_cmd(self, data):
    return Path.get_cmd(self, data)


class TestPath(TestCase):
  """Unit tests for the base Generator Path class."""

  def setUp(self) -> None:
    """Resets shared Path state before each test."""

    logging.disable(logging.NOTSET)
    Path.t0 = 0
    Path.last_cmd = None

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

    with self.assertRaisesRegex(DefinitionError, 'Generator Path'):
      type('DuplicateGeneratorPath',
           (Path,),
           {'get_cmd': lambda self, data: 0})

  def test_init_accepts_unused_arguments(self) -> None:
    """Checks base initialization with ignored arguments."""

    path = UnitTestPath('unused', option='ignored')

    self.assertIsNone(path._logger)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and first-message forwarding."""

    path = UnitTestPath()
    logger_name = f"{current_process().name}.UnitTestPath"

    with self.assertLogs(logger_name, level='INFO') as captured:
      path.log(logging.INFO, "message")

    self.assertIs(path._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_default_get_cmd_warns_waits_and_returns_last_command(self) -> None:
    """Checks the fallback get_cmd implementation exposed via super."""

    Path.last_cmd = 12
    path = UnitTestPath()

    with (patch.object(path_module, 'sleep') as sleep_mock,
          patch.object(path, 'log') as log_mock):
      self.assertEqual(path.get_cmd({}), 12)

    log_mock.assert_called_once_with(logging.WARNING, "The get_cmd was called "
                                     "but is not defined ! Please define a "
                                     "get_cmd method for your Generator "
                                     "path ! Returning the last sent command")
    sleep_mock.assert_called_once_with(1)

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
        with self.assertRaises(ValueError):
          path.parse_condition(condition)

  def test_package_registry_is_base_registry(self) -> None:
    """Checks package-level exports for all bundled Path classes."""

    self.assertIs(generator_path_package.meta_path.paths_dict, Path.classes)
    for name in ('Conditional', 'Constant', 'Custom', 'Cyclic',
                 'CyclicRamp', 'Integrator', 'Ramp', 'Sine'):
      self.assertIs(Path.classes[name],
                    getattr(generator_path_package, name))
