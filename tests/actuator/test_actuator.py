# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase
from unittest.mock import patch
from crappy._global import DefinitionError
from crappy.actuator.meta_actuator.actuator import Actuator
import crappy.actuator.meta_actuator.actuator as actuator_module


class TestActuator(TestCase):
  """Unit tests for the Actuator base class."""

  def tearDown(self) -> None:
    """Removes dynamic classes registered during tests."""

    for name in ('TestRegisteredActuator', 'TestDuplicateActuator',
                 'TestStoppingActuator'):
      Actuator.classes.pop(name, None)

  def test_subclasses_are_registered_by_name(self) -> None:
    """Checks that new Actuator subclasses are added to the class registry."""

    Actuator.classes.pop('TestRegisteredActuator', None)

    cls = type('TestRegisteredActuator', (Actuator,), {})

    self.assertIs(Actuator.classes['TestRegisteredActuator'], cls)

  def test_duplicate_subclass_names_are_rejected(self) -> None:
    """Checks that two Actuators cannot share a class name."""

    Actuator.classes.pop('TestDuplicateActuator', None)
    type('TestDuplicateActuator', (Actuator,), {})

    with self.assertRaises(DefinitionError):
      type('TestDuplicateActuator', (Actuator,), {})

  def test_default_attributes_and_ft232h_flag(self) -> None:
    """Checks base instance initialization and class defaults."""

    actuator = Actuator('unused', option='ignored')

    self.assertFalse(actuator.ft232h)
    self.assertIsNone(actuator._logger)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and lazy initialization."""

    actuator = Actuator()
    logger_name = f"{current_process().name}.Actuator"

    with self.assertLogs(logger_name, level='INFO') as captured:
      actuator.log(logging.INFO, "message")

    self.assertIs(actuator._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_open_and_close_are_noops_by_default(self) -> None:
    """Checks that optional lifecycle methods can be left undefined."""

    actuator = Actuator()

    self.assertIsNone(actuator.open())
    self.assertIsNone(actuator.close())

  def test_default_set_speed_logs_and_waits(self) -> None:
    """Checks fallback behavior for undefined speed commands."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.set_speed(1.5))

    mocked_sleep.assert_called_once_with(1)
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn("set_speed", logs[0][1])

  def test_default_set_position_logs_and_waits(self) -> None:
    """Checks fallback behavior for undefined position commands."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.set_position(1.5, None))

    mocked_sleep.assert_called_once_with(1)
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn("set_position", logs[0][1])

  def test_default_get_speed_logs_waits_and_returns_none(self) -> None:
    """Checks fallback behavior for undefined speed acquisition."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.get_speed())

    mocked_sleep.assert_called_once_with(1)
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn("get_speed", logs[0][1])

  def test_default_get_position_logs_waits_and_returns_none(self) -> None:
    """Checks fallback behavior for undefined position acquisition."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.get_position())

    mocked_sleep.assert_called_once_with(1)
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn("get_position", logs[0][1])

  def test_default_stop_delegates_to_set_speed_zero(self) -> None:
    """Checks the default stopping strategy."""

    Actuator.classes.pop('TestStoppingActuator', None)

    def init(self) -> None:
      Actuator.__init__(self)
      self.commands = list()

    def set_speed(self, speed: float) -> None:
      self.commands.append(speed)

    cls = type('TestStoppingActuator',
               (Actuator,),
               {'__init__': init, 'set_speed': set_speed})
    actuator = cls()

    actuator.stop()

    self.assertEqual(actuator.commands, [0])
