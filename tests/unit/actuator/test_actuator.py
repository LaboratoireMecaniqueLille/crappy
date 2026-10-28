# coding: utf-8

import logging
from unittest import TestCase
from unittest.mock import patch
from crappy._exceptions import DefinitionError
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

  def test_unused_constructor_arguments_allow_normal_lifecycle(self) -> None:
    """Checks base instance initialization."""

    actuator = Actuator('unused', option='ignored')

    self.assertIsNone(actuator.open())
    self.assertIsNone(actuator.close())

  def test_log_forwards_levels_and_messages(self) -> None:
    """User messages retain their severity on first and subsequent calls."""

    actuator = Actuator()
    with self.assertLogs(level='INFO') as captured:
      actuator.log(logging.INFO, 'message')
      actuator.log(logging.WARNING, 'follow-up')
    records = [(record.levelno, record.getMessage()) for record in captured.records]
    self.assertIn((logging.INFO, 'message'), records)
    self.assertIn((logging.WARNING, 'follow-up'), records)

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

    self.assertTrue(any(call.args[0] > 0
                        for call in mocked_sleep.call_args_list))
    self.assertEqual(logs[0][0], logging.WARNING)

  def test_default_set_position_logs_and_waits(self) -> None:
    """Checks fallback behavior for undefined position commands."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.set_position(1.5, None))

    self.assertTrue(any(call.args[0] > 0
                        for call in mocked_sleep.call_args_list))
    self.assertEqual(logs[0][0], logging.WARNING)

  def test_default_get_speed_logs_waits_and_returns_none(self) -> None:
    """Checks fallback behavior for undefined speed acquisition."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.get_speed())

    self.assertTrue(any(call.args[0] > 0
                        for call in mocked_sleep.call_args_list))
    self.assertEqual(logs[0][0], logging.WARNING)

  def test_default_get_position_logs_waits_and_returns_none(self) -> None:
    """Checks fallback behavior for undefined position acquisition."""

    actuator = Actuator()
    logs = list()
    actuator.log = lambda level, msg: logs.append((level, msg))

    with patch.object(actuator_module, 'sleep') as mocked_sleep:
      self.assertIsNone(actuator.get_position())

    self.assertTrue(any(call.args[0] > 0
                        for call in mocked_sleep.call_args_list))
    self.assertEqual(logs[0][0], logging.WARNING)

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
