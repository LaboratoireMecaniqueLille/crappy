# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch

from crappy.blocks.auto_drive_video_extenso import AutoDriveVideoExtenso
import crappy.blocks.auto_drive_video_extenso as auto_drive_module

from ..block import BlockTestBase, TestBlock, link


class TrackingAutoDriveActuator:
  """Small Actuator test double for AutoDriveVideoExtenso tests."""

  instances: list['TrackingAutoDriveActuator'] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor kwargs and initializes call state."""

    self.kwargs = dict(kwargs)
    if kwargs.get('constructor_error') is not None:
      raise kwargs['constructor_error']
    self.open_error = kwargs.get('open_error')
    self.speed_error = kwargs.get('speed_error')
    self.stop_error = kwargs.get('stop_error')
    self.close_error = kwargs.get('close_error')
    self.calls: list[str] = list()
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

    self.calls.append('open')
    if self.open_error is not None:
      raise self.open_error
    self.opened = True

  def set_speed(self, speed: float) -> None:
    """Records speed commands."""

    self.speed_commands.append(speed)
    self.calls.append('set_speed')
    if self.speed_error is not None:
      raise self.speed_error

  def stop(self) -> None:
    """Records stop calls."""

    self.calls.append('stop')
    if self.stop_error is not None:
      raise self.stop_error
    self.stopped = True

  def close(self) -> None:
    """Records close calls."""

    self.calls.append('close')
    if self.close_error is not None:
      raise self.close_error
    self.closed = True


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
      ([], {}, TypeError),
      ({'type': 1}, {}, TypeError),
      ({'type': ' '}, {}, ValueError),
      ({'type': 'UnknownActuator'}, {}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'direction': 'Z+'}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'direction': None}, TypeError),
      ({'type': 'TrackingAutoDriveActuator'}, {'pixel_range': 0}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'pixel_range': 1.5}, TypeError),
      ({'type': 'TrackingAutoDriveActuator'}, {'pixel_range': True}, TypeError),
      ({'type': 'TrackingAutoDriveActuator'}, {'max_speed': 0}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'max_speed': 'fast'}, TypeError),
      ({'type': 'TrackingAutoDriveActuator'},
       {'max_speed': float('inf')}, ValueError),
      ({'type': 'TrackingAutoDriveActuator'}, {'gain': None}, TypeError),
      ({'type': 'TrackingAutoDriveActuator'},
       {'gain': float('nan')}, ValueError),
    )

    with self._actuator_patch():
      for actuator, kwargs, exception in cases:
        with self.subTest(actuator=actuator, kwargs=kwargs):
          with self.assertRaises(exception):
            AutoDriveVideoExtenso(actuator, **kwargs)

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
    block._device_opened = True

    block.finish()

    self.assertTrue(actuator.stopped)
    self.assertTrue(actuator.closed)
    block.finish()
    self.assertEqual(actuator.calls, ['stop', 'close'])

  def test_finish_without_actuator_is_a_noop(self) -> None:
    """Checks finish before prepare remains harmless."""

    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator'})

    block.finish()

    self.assertEqual(TrackingAutoDriveActuator.instances, [])

  def test_prepare_does_not_mutate_actuator_options(self) -> None:
    """Checks preparation retains the caller's type and driver kwargs."""

    options = {'type': 'TrackingAutoDriveActuator', 'custom': 1}
    with self._actuator_patch():
      block = AutoDriveVideoExtenso(options)
      link(TestBlock(), block)
      block.prepare()
    self.assertEqual(options,
                     {'type': 'TrackingAutoDriveActuator', 'custom': 1})
    block.finish()

  def test_finish_after_partial_preparation(self) -> None:
    """Checks constructor/open failures and failed initial zero-speed calls."""

    for step in ('constructor_error', 'open_error', 'speed_error'):
      with self.subTest(step=step), self._actuator_patch():
        TrackingAutoDriveActuator.reset()
        error = RuntimeError(f'{step} failed')
        block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator',
                                      step: error})
        link(TestBlock(), block)
        with self.assertRaises(RuntimeError) as caught:
          block.prepare()
        self.assertIs(caught.exception, error)
        block.finish()
        block.finish()

        if step == 'constructor_error':
          self.assertIsNone(block._device)
          self.assertEqual(TrackingAutoDriveActuator.instances, [])
        else:
          device = TrackingAutoDriveActuator.instances[-1]
          expected = (['open', 'close'] if step == 'open_error' else
                      ['open', 'set_speed', 'stop', 'close'])
          self.assertEqual(device.calls, expected)

  def test_finish_groups_stop_and_close_failures(self) -> None:
    """Checks close is attempted after stop fails and both errors survive."""

    stop_error = RuntimeError('stop failed')
    close_error = OSError('close failed')
    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator',
                                    'stop_error': stop_error,
                                    'close_error': close_error})
      link(TestBlock(), block)
      block.prepare()
    device = TrackingAutoDriveActuator.instances[-1]
    device.calls.clear()

    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()

    self.assertEqual(device.calls, ['stop', 'close'])
    self.assertEqual(caught.exception.exceptions, (stop_error, close_error))
    self.assertIn('stop', stop_error.__notes__[0])
    self.assertIn('close', close_error.__notes__[0])
    device.stop_error = device.close_error = None
    block.finish()

  def test_finish_retries_close_without_repeating_stop(self) -> None:
    """Checks successful cleanup steps are not repeated."""

    error = OSError('close failed')
    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator',
                                    'close_error': error})
      link(TestBlock(), block)
      block.prepare()
    device = TrackingAutoDriveActuator.instances[-1]
    device.calls.clear()

    with self.assertRaises(OSError) as caught:
      block.finish()
    self.assertIs(caught.exception, error)
    device.close_error = None
    block.finish()
    block.finish()
    self.assertEqual(device.calls, ['stop', 'close', 'close'])

  def test_finish_preserves_interrupt_with_close_failure_as_cause(self) -> None:
    """Checks interrupted stop still closes and retains its close failure."""

    interrupt = KeyboardInterrupt()
    error = OSError('close failed')
    with self._actuator_patch():
      block = AutoDriveVideoExtenso({'type': 'TrackingAutoDriveActuator',
                                    'stop_error': interrupt,
                                    'close_error': error})
      link(TestBlock(), block)
      block.prepare()
    device = TrackingAutoDriveActuator.instances[-1]
    device.calls.clear()

    with self.assertRaises(KeyboardInterrupt) as caught:
      block.finish()
    self.assertIs(caught.exception, interrupt)
    self.assertEqual(device.calls, ['stop', 'close'])
    self.assertEqual(interrupt.__cause__.exceptions, (error,))
    device.stop_error = device.close_error = None
    block.finish()
