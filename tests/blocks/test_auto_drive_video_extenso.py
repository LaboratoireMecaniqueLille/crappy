# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch

from crappy.blocks.auto_drive_video_extenso import AutoDriveVideoExtenso
import crappy.blocks.auto_drive_video_extenso as auto_drive_module

from ..block import BlockTestBase, TestBlock, link


class TrackingAutoDriveActuator:
  """Small Actuator test double for AutoDriveVideoExtenso tests."""

  ft232h = False
  instances: list['TrackingAutoDriveActuator'] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor kwargs and initializes call state."""

    self.kwargs = dict(kwargs)
    self.speed_commands = list()
    self.opened = False
    self.stopped = False
    self.closed = False
    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears state shared by fake actuator instances."""

    cls.instances = list()

  def open(self) -> None:
    """Records open calls."""

    self.opened = True

  def set_speed(self, speed: float) -> None:
    """Records speed commands."""

    self.speed_commands.append(speed)

  def stop(self) -> None:
    """Records stop calls."""

    self.stopped = True

  def close(self) -> None:
    """Records close calls."""

    self.closed = True


class FT232HTrackingAutoDriveActuator(TrackingAutoDriveActuator):
  """Tracking AutoDrive actuator declaring FT232H support."""

  ft232h = True


class TestAutoDriveVideoExtenso(BlockTestBase):
  """Unit tests for the AutoDriveVideoExtenso Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Resets fake actuator state before each test."""

    TrackingAutoDriveActuator.reset()

  @staticmethod
  def _actuator_patch():
    """Registers fake actuators in AutoDrive's actuator registry."""

    return patch.dict(auto_drive_module.actuator_dict, {
      'TrackingAutoDriveActuator': TrackingAutoDriveActuator,
      'FT232HTrackingAutoDriveActuator': FT232HTrackingAutoDriveActuator,
    })

  @staticmethod
  def _capture_send(block: AutoDriveVideoExtenso) -> list[list[Any]]:
    """Captures output data sent by AutoDriveVideoExtenso."""

    sent = list()

    def send(data: list[Any]) -> None:
      sent.append(list(data))

    block.send = send
    return sent

  @staticmethod
  def _set_received(block: AutoDriveVideoExtenso,
                    data: dict[str, Any]) -> list[bool]:
    """Makes recv_last_data return a deterministic payload."""

    fill_missing_values = list()

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      fill_missing_values.append(fill_missing)
      return dict(data)

    block.recv_last_data = recv_last_data
    return fill_missing_values

  @staticmethod
  def _set_t0(block: AutoDriveVideoExtenso) -> None:
    """Sets a deterministic start time on AutoDriveVideoExtenso."""

    block._instance_t0 = Value('d', TestAutoDriveVideoExtenso._t0)

  def test_constructor_sets_block_options_and_gain_direction(self) -> None:
    """Checks labels, frequency options, and direction sign handling."""

    with self._actuator_patch():
      positive = AutoDriveVideoExtenso(
        {'type': 'TrackingAutoDriveActuator'},
        gain=3,
        direction='X+',
        pixel_range=42,
        max_speed=5,
        freq=None,
        display_freq=True,
        debug=True)
      negative = AutoDriveVideoExtenso(
        {'type': 'TrackingAutoDriveActuator'},
        gain=3,
        direction='y-')

    self.assertEqual(positive.labels, ['t(s)', 'diff(pix)'])
    self.assertEqual(positive._gain, 3)
    self.assertEqual(positive._direction, 'X+')
    self.assertEqual(positive._pixel_range, 42)
    self.assertEqual(positive._max_speed, 5)
    self.assertIsNone(positive.freq)
    self.assertTrue(positive.display_freq)
    self.assertTrue(positive.debug)
    self.assertEqual(negative._gain, -3)
    self.assertEqual(negative._direction, 'y-')

  def test_constructor_validation(self) -> None:
    """Checks invalid AutoDriveVideoExtenso settings fail early."""

    cases = (
      ({}, {}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'direction': 'Z+'}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'pixel_range': 0}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'max_speed': 0}, ValueError),
    )

    with self._actuator_patch():
      for actuator, kwargs, exception in cases:
        with self.subTest(actuator=actuator, kwargs=kwargs):
          with self.assertRaises(exception):
            AutoDriveVideoExtenso(actuator, **kwargs)

  def test_constructor_registers_ft232h_when_needed(self) -> None:
    """Checks FT232H registration for compatible actuators."""

    with (self._actuator_patch(),
          patch.object(auto_drive_module.USBServer,
                       'register',
                       return_value=('ft232h',)) as register):
      block = AutoDriveVideoExtenso(
        {'type': 'FT232HTrackingAutoDriveActuator'},
        ft232h_ser_num='ABC')

    register.assert_called_once_with('ABC')
    self.assertEqual(block._ft232h_args, ('ft232h',))

  def test_prepare_requires_exactly_one_input_link(self) -> None:
    """Checks AutoDriveVideoExtenso input link validation."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})

      with self.assertRaises(IOError):
        block.prepare()

      source_1 = TestBlock()
      source_2 = TestBlock()
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})
      link(source_1, block)
      link(source_2, block)

      with self.assertRaises(IOError):
        block.prepare()

    self.assertEqual(TrackingAutoDriveActuator.instances, [])

  def test_prepare_instantiates_opens_and_stops_actuator(self) -> None:
    """Checks actuator initialization and startup speed command."""

    source = TestBlock()
    with self._actuator_patch():
      block = AutoDriveVideoExtenso({
        'type': 'TrackingAutoDriveActuator',
        'custom': 1,
      })
      link(source, block)
      block.prepare()

    actuator = TrackingAutoDriveActuator.instances[-1]

    self.assertIs(block._device, actuator)
    self.assertEqual(actuator.kwargs, {'custom': 1})
    self.assertTrue(actuator.opened)
    self.assertEqual(actuator.speed_commands, [0])

  def test_prepare_passes_ft232h_args_to_actuator(self) -> None:
    """Checks FT232H constructor argument injection."""

    source = TestBlock()
    with (self._actuator_patch(),
          patch.object(auto_drive_module.USBServer,
                       'register',
                       return_value=('ft232h',))):
      block = AutoDriveVideoExtenso({
        'type': 'FT232HTrackingAutoDriveActuator',
        'custom': 1,
      })
      link(source, block)
      block.prepare()

    self.assertEqual(TrackingAutoDriveActuator.instances[-1].kwargs, {
      'custom': 1,
      '_ft232h_args': ('ft232h',),
    })

  def test_loop_returns_when_no_new_coordinates_are_available(self) -> None:
    """Checks that missing input data does not command the actuator."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})
    actuator = TrackingAutoDriveActuator()
    block._device = actuator
    sent = self._capture_send(block)
    fill_missing_values = self._set_received(block, {})

    block.loop()

    self.assertEqual(fill_missing_values, [False])
    self.assertEqual(actuator.speed_commands, [])
    self.assertEqual(sent, [])

  def test_loop_uses_x_coordinates_and_clamps_speed(self) -> None:
    """Checks X-axis center error, speed clamp, and emitted payload."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso(
        {'type': 'TrackingAutoDriveActuator'},
        gain=10,
        direction='X+',
        pixel_range=100,
        max_speed=50)
    self._set_t0(block)
    actuator = TrackingAutoDriveActuator()
    block._device = actuator
    sent = self._capture_send(block)
    self._set_received(block, {
      'Coord(px)': [(10, 100), (30, 120)],
    })

    with patch.object(auto_drive_module, 'time', return_value=12):
      block.loop()

    self.assertEqual(actuator.speed_commands, [50])
    self.assertEqual(sent, [[2, 60]])

  def test_loop_uses_y_coordinates_and_direction_sign(self) -> None:
    """Checks Y-axis center error and inverted direction sign."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso(
        {'type': 'TrackingAutoDriveActuator'},
        gain=2,
        direction='Y-',
        pixel_range=100,
        max_speed=100)
    self._set_t0(block)
    actuator = TrackingAutoDriveActuator()
    block._device = actuator
    sent = self._capture_send(block)
    self._set_received(block, {
      'Coord(px)': [(10, 50), (30, 70)],
    })

    with patch.object(auto_drive_module, 'time', return_value=13.5):
      block.loop()

    self.assertEqual(actuator.speed_commands, [60])
    self.assertEqual(sent, [[3.5, -30]])

  def test_loop_propagates_missing_coordinate_label(self) -> None:
    """Checks that malformed upstream payloads fail explicitly."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})
    block._device = TrackingAutoDriveActuator()
    self._set_received(block, {'other': []})

    with self.assertRaises(KeyError):
      block.loop()

  def test_finish_stops_and_closes_existing_actuator(self) -> None:
    """Checks actuator cleanup at the end of the test."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})
    actuator = TrackingAutoDriveActuator()
    block._device = actuator

    block.finish()

    self.assertTrue(actuator.stopped)
    self.assertTrue(actuator.closed)

  def test_finish_without_actuator_is_a_noop(self) -> None:
    """Checks finish before prepare remains harmless."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})

    block.finish()

    self.assertEqual(TrackingAutoDriveActuator.instances, [])
