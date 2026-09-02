# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch

from crappy.blocks.machine import ActuatorInstance, Machine
import crappy.blocks.machine as machine_module

from ..block import BlockTestBase, TestBlock, link


class TrackingActuator:
  """Small Actuator test double for Machine tests."""

  ft232h = False
  instances: list['TrackingActuator'] = list()
  events: list[tuple[str, int]] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor kwargs and initializes call state."""

    self.kwargs = dict(kwargs)
    self.speed_commands = list()
    self.position_commands = list()
    self.opened = False
    self.stopped = False
    self.closed = False
    self.speed_to_return = kwargs.get('speed_to_return')
    self.position_to_return = kwargs.get('position_to_return')
    self.id = len(self.instances)
    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears state shared by fake actuator instances."""

    cls.instances = list()
    cls.events = list()

  def open(self) -> None:
    """Records open calls."""

    self.opened = True

  def set_speed(self, speed: float) -> None:
    """Records speed commands."""

    self.speed_commands.append(speed)

  def set_position(self, position: float, speed: float | None) -> None:
    """Records position commands."""

    self.position_commands.append((position, speed))

  def get_speed(self) -> float | None:
    """Returns the configured speed value."""

    return self.speed_to_return

  def get_position(self) -> float | None:
    """Returns the configured position value."""

    return self.position_to_return

  def stop(self) -> None:
    """Records stop calls."""

    self.stopped = True
    self.events.append(('stop', self.id))

  def close(self) -> None:
    """Records close calls."""

    self.closed = True
    self.events.append(('close', self.id))


class FT232HTrackingActuator(TrackingActuator):
  """Tracking actuator declaring FT232H support."""

  ft232h = True


class TestMachine(BlockTestBase):
  """Unit tests for the Machine Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Resets fake actuator state before each test."""

    TrackingActuator.reset()

  @staticmethod
  def _actuator_patch():
    """Registers fake actuators in Machine's actuator registry."""

    return patch.dict(machine_module.actuator_dict, {
      'TrackingActuator': TrackingActuator,
      'FT232HTrackingActuator': FT232HTrackingActuator,
    })

  @staticmethod
  def _capture_send(block: Machine) -> list[dict[str, Any]]:
    """Captures output data sent by Machine."""

    sent = list()

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    block.send = send
    return sent

  @staticmethod
  def _set_received(block: Machine,
                    data: dict[str, Any]) -> list[bool]:
    """Makes recv_last_data return a deterministic payload."""

    fill_missing_values = list()

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      fill_missing_values.append(fill_missing)
      return dict(data)

    block.recv_last_data = recv_last_data
    return fill_missing_values

  @staticmethod
  def _set_t0(block: Machine) -> None:
    """Sets a deterministic start time on a Machine."""

    block._instance_t0 = Value('d', TestMachine._t0)

  def test_constructor_validation(self) -> None:
    """Checks invalid actuator settings fail early."""

    cases = (
      ([], ValueError),
      ([{'cmd_label': 'cmd'}], ValueError),
      ([{'type': 'Fake_motor'}], NotImplementedError),
      ([{'type': 'UnknownActuator'}], ValueError),
      ([{'type': 'TrackingActuator', 'mode': 'speeed'}], ValueError),
    )

    with self._actuator_patch():
      for actuators, exception in cases:
        with self.subTest(actuators=actuators):
          with self.assertRaises(exception):
            Machine(actuators)

  def test_constructor_splits_machine_settings_from_actuator_kwargs(
      self) -> None:
    """Checks settings normalization and common-key precedence."""

    actuator = {
      'type': 'TrackingActuator',
      'cmd_label': 'cmd',
      'position_label': 'pos',
      'custom': 1,
    }
    common = {'cmd_label': 'common_cmd', 'shared': 2}

    with self._actuator_patch():
      block = Machine([actuator],
                      common=common,
                      time_label='time',
                      spam=True,
                      freq=None)

    self.assertEqual(block._types, ['TrackingActuator'])
    self.assertEqual(block._settings, [{
      'cmd_label': 'common_cmd',
      'position_label': 'pos',
    }])
    self.assertEqual(block._actuators_kw, [{'custom': 1, 'shared': 2}])
    self.assertEqual(block._time_label, 'time')
    self.assertTrue(block._spam)
    self.assertIsNone(block.freq)

  def test_constructor_registers_ft232h_once_when_needed(self) -> None:
    """Checks FT232H registration for compatible actuators."""

    with (self._actuator_patch(),
          patch.object(machine_module.USBServer,
                       'register',
                       return_value=('ft232h',)) as register):
      block = Machine([{'type': 'FT232HTrackingActuator'}],
                      ft232h_ser_num='ABC')

    register.assert_called_once_with('ABC')
    self.assertEqual(block._ft232h_args, ('ft232h',))

  def test_prepare_requires_a_link_before_instantiating_actuators(
      self) -> None:
    """Checks link validation happens before Actuator construction."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}])

      with self.assertRaises(IOError):
        block.prepare()

    self.assertEqual(TrackingActuator.instances, [])

  def test_prepare_instantiates_and_opens_actuators(self) -> None:
    """Checks actuator creation, settings, and open calls."""

    source = TestBlock()
    with (self._actuator_patch(),
          patch.object(machine_module.USBServer,
                       'register',
                       return_value=('ft232h',))):
      block = Machine([
        {'type': 'TrackingActuator', 'custom': 1},
        {'type': 'FT232HTrackingActuator', 'custom': 2},
      ])
      link(source, block)
      block.prepare()

    self.assertEqual(len(block._actuators), 2)
    self.assertEqual(TrackingActuator.instances[0].kwargs, {'custom': 1})
    self.assertEqual(TrackingActuator.instances[1].kwargs, {
      'custom': 2,
      '_ft232h_args': ('ft232h',),
    })
    self.assertTrue(all(actuator.opened
                        for actuator in TrackingActuator.instances))

  def test_loop_sends_speed_commands_from_latest_data(self) -> None:
    """Checks speed-mode command forwarding and spam flag usage."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}], spam=False)
    actuator = TrackingActuator()
    block._actuators = [
      ActuatorInstance(actuator=actuator,
                       mode='speed',
                       cmd_label='drive'),
    ]
    fill_missing_values = self._set_received(block, {'drive': 3.5})

    block.loop()

    self.assertEqual(fill_missing_values, [False])
    self.assertEqual(actuator.speed_commands, [3.5])
    self.assertEqual(actuator.position_commands, [])

  def test_loop_updates_position_speed_before_position_command(self) -> None:
    """Checks position mode with a runtime speed command label."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}], spam=True)
    actuator = TrackingActuator()
    block._actuators = [
      ActuatorInstance(actuator=actuator,
                       speed=1.0,
                       mode='position',
                       cmd_label='target',
                       speed_cmd_label='speed'),
    ]
    fill_missing_values = self._set_received(block, {
      'target': 12,
      'speed': 4,
    })

    block.loop()

    self.assertEqual(fill_missing_values, [True])
    self.assertEqual(block._actuators[0].speed, 4)
    self.assertEqual(actuator.position_commands, [(12, 4)])
    self.assertEqual(actuator.speed_commands, [])

  def test_loop_ignores_missing_command_labels(self) -> None:
    """Checks that unrelated input data does not command actuators."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}])
    actuator = TrackingActuator()
    block._actuators = [
      ActuatorInstance(actuator=actuator,
                       mode='speed',
                       cmd_label='drive'),
    ]
    self._set_received(block, {'other': 3.5})

    block.loop()

    self.assertEqual(actuator.speed_commands, [])
    self.assertEqual(actuator.position_commands, [])

  def test_loop_reads_requested_outputs_and_timestamp(self) -> None:
    """Checks position/speed acquisition and output payload formatting."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}], time_label='time')
    self._set_t0(block)
    actuator_1 = TrackingActuator(position_to_return=2.5,
                                  speed_to_return=7.5)
    actuator_2 = TrackingActuator(position_to_return=None,
                                  speed_to_return=9)
    block._actuators = [
      ActuatorInstance(actuator=actuator_1,
                       position_label='pos',
                       speed_label='speed'),
      ActuatorInstance(actuator=actuator_2,
                       position_label='ignored_pos',
                       speed_label='speed_2'),
    ]
    sent = self._capture_send(block)
    self._set_received(block, {})

    with patch.object(machine_module, 'time', return_value=12.5):
      block.loop()

    self.assertEqual(sent, [{
      'pos': 2.5,
      'speed': 7.5,
      'speed_2': 9,
      'time': 2.5,
    }])

  def test_loop_does_not_send_without_requested_or_available_outputs(
      self) -> None:
    """Checks that empty readouts do not emit downstream data."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}])
    self._set_t0(block)
    actuator = TrackingActuator(position_to_return=None,
                                speed_to_return=None)
    block._actuators = [
      ActuatorInstance(actuator=actuator,
                       position_label='pos',
                       speed_label='speed'),
    ]
    sent = self._capture_send(block)
    self._set_received(block, {})

    block.loop()

    self.assertEqual(sent, [])

  def test_finish_stops_all_actuators_before_closing_them(self) -> None:
    """Checks finish order across several actuators."""

    with self._actuator_patch():
      block = Machine([{'type': 'TrackingActuator'}])
    actuator_1 = TrackingActuator()
    actuator_2 = TrackingActuator()
    block._actuators = [
      ActuatorInstance(actuator=actuator_1),
      ActuatorInstance(actuator=actuator_2),
    ]

    block.finish()

    self.assertEqual(TrackingActuator.events, [
      ('stop', actuator_1.id),
      ('stop', actuator_2.id),
      ('close', actuator_1.id),
      ('close', actuator_2.id),
    ])
    self.assertTrue(actuator_1.stopped)
    self.assertTrue(actuator_1.closed)
    self.assertTrue(actuator_2.stopped)
    self.assertTrue(actuator_2.closed)
