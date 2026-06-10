# coding: utf-8

from multiprocessing import Value
from struct import pack
from typing import Any
from unittest.mock import patch
from crappy._global import OptionalModule
import crappy.blocks.ucontroller as ucontroller_module

from ..block import BlockTestBase, TestBlock, link


class FakeSerialError(Exception):
  """Exception raised by the fake serial bus."""


class FakeMicrocontrollerSerial:
  """Small serial-side emulator for the bundled microcontroller protocol."""

  instances: list['FakeMicrocontrollerSerial'] = list()
  open_error: Exception | None = None

  def __init__(self,
               port: str,
               baudrate: int,
               timeout: float,
               write_timeout: float) -> None:
    """Stores constructor arguments and initializes protocol state."""

    if self.open_error is not None:
      raise self.open_error

    self.port = port
    self.baudrate = baudrate
    self.timeout = timeout
    self.write_timeout = write_timeout

    self.writes = list()
    self.commands = dict()
    self.labels = dict()
    self.received_commands = list()
    self.read_buffer = bytearray()

    self.expected_commands = 0
    self.expected_labels = 0
    self.send_t = False
    self.closed = False
    self.stopped = False
    self.input_resets = 0
    self.output_resets = 0
    self._mode = 'waiting'

    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears state shared between fake serial instances."""

    cls.instances = list()
    cls.open_error = None

  def reset_input_buffer(self) -> None:
    """Records input buffer reset calls."""

    self.input_resets += 1

  def reset_output_buffer(self) -> None:
    """Records output buffer reset calls."""

    self.output_resets += 1

  def write(self, msg: bytes) -> int:
    """Records and parses data written by UController."""

    self.writes.append(msg)
    self._parse_line(msg.decode().strip())
    return len(msg)

  def read(self, size: int = 1) -> bytes:
    """Returns queued bytes one chunk at a time, like pyserial."""

    if not self.read_buffer:
      return b''

    ret = bytes(self.read_buffer[:size])
    del self.read_buffer[:size]
    return ret

  def close(self) -> None:
    """Marks the fake serial bus as closed."""

    self.closed = True

  def queue_label(self,
                  label: str,
                  value: float,
                  timestamp_ms: int = 0) -> None:
    """Queues a packed label frame like send_to_pc would."""

    label_id = self.labels[label]
    if self.send_t:
      self.read_buffer.extend(pack('<ibf', timestamp_ms, label_id, value))
    else:
      self.read_buffer.extend(pack('<bf', label_id, value))

  def queue_raw(self, raw: bytes) -> None:
    """Queues raw bytes to be read by UController."""

    self.read_buffer.extend(raw)

  def _parse_line(self, line: str) -> None:
    """Parses startup rows, command rows, and stop messages."""

    if line == 'stop!':
      self.stopped = True
      return

    if line.startswith('go'):
      self.expected_commands = int(line[2])
      self.expected_labels = int(line[3])
      self._mode = 'commands' if self.expected_commands else 'labels'
      if not self.expected_labels and not self.expected_commands:
        self._mode = 'running'
      return

    if self._mode == 'commands':
      self.commands[int(line[0])] = line[1:]
      if len(self.commands) == self.expected_commands:
        self._mode = 'labels' if self.expected_labels else 'running'
      return

    if self._mode == 'labels':
      self.labels[line[1:]] = int(line[0])
      self.send_t = self.send_t or line[1:] == 't(s)'
      if len(self.labels) == self.expected_labels:
        self._mode = 'running'
      return

    self.received_commands.append((self.commands[int(line[0])],
                                   float(line[1:])))


class TestUController(BlockTestBase):
  """Unit tests for the UController Block-specific behavior."""

  def setUp(self) -> None:
    """Resets the fake serial bus before each test."""

    FakeMicrocontrollerSerial.reset()

  def _serial_patch(self):
    """Patches UController to use the fake serial bus."""

    return patch.multiple(ucontroller_module,
                          Serial=FakeMicrocontrollerSerial,
                          SerialException=FakeSerialError)

  @staticmethod
  def _capture_send(controller) -> list[dict[str, Any]]:
    """Captures output data sent by UController."""

    sent = list()

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    controller.send = send
    return sent

  def test_labels_and_command_labels_normalization(self) -> None:
    """Checks the supported labels and cmd_labels forms."""

    ctrl = ucontroller_module.UController(labels='nr',
                                          cmd_labels='freq',
                                          init_output={'nr': 0})

    self.assertEqual(ctrl._labels, ['nr'])
    self.assertEqual(ctrl._cmd_labels, ['freq'])
    self.assertEqual(ctrl._prev_cmd, {'freq': None})

    ctrl = ucontroller_module.UController(labels=('a', 'b'),
                                          cmd_labels=('x', 'y'),
                                          init_output={'a': 0, 'b': 1},
                                          freq=None)

    self.assertEqual(ctrl._labels, ['a', 'b'])
    self.assertEqual(ctrl._cmd_labels, ['x', 'y'])
    self.assertIsNone(ctrl.freq)

  def test_argument_validation(self) -> None:
    """Checks validation of label counts and initialization values."""

    labels = [f'label_{i}' for i in range(9)]

    with self.assertRaises(ValueError):
      ucontroller_module.UController(
        labels=labels,
        init_output={label: 0 for label in labels},
        t_device=True)

    with self.assertRaises(ValueError):
      ucontroller_module.UController(labels='nr')

    with self.assertRaises(TypeError):
      ucontroller_module.UController(labels='nr',
                                     init_output={'nr': 0},
                                     post_process={'nr': 1})

  def test_prepare_requires_matching_links(self) -> None:
    """Checks that labels/cmd_labels match the Link layout."""

    with self.assertRaises(IOError):
      ucontroller_module.UController(labels='nr',
                                     init_output={'nr': 0}).prepare()

    with self.assertRaises(IOError):
      ucontroller_module.UController(cmd_labels='freq').prepare()

    source = TestBlock()
    ctrl = ucontroller_module.UController()
    link(source, ctrl)

    with self.assertRaises(IOError):
      ctrl.prepare()

    ctrl = ucontroller_module.UController()
    sink = TestBlock()
    link(ctrl, sink)

    with self.assertRaises(IOError):
      ctrl.prepare()

  def test_prepare_sends_startup_tables(self) -> None:
    """Checks the startup protocol expected by the microcontroller template."""

    source = TestBlock()
    ctrl = ucontroller_module.UController(labels='nr',
                                          cmd_labels='freq',
                                          init_output={'nr': 0},
                                          t_device=True,
                                          port='COM1',
                                          baudrate=9600)
    sink = TestBlock()
    link(source, ctrl)
    link(ctrl, sink)

    with self._serial_patch():
      ctrl.prepare()

    bus = FakeMicrocontrollerSerial.instances[-1]

    self.assertEqual((bus.port, bus.baudrate, bus.timeout, bus.write_timeout),
                     ('COM1', 9600, 0, 0))
    self.assertEqual(bus.input_resets, 1)
    self.assertEqual(bus.output_resets, 1)
    self.assertEqual(bus.writes, [
      b'go12\r\n',
      b'1freq\r\n',
      b'1nr\r\n',
      b'0t(s)\r\n',
    ])
    self.assertEqual(bus.commands, {1: 'freq'})
    self.assertEqual(bus.labels, {'nr': 1, 't(s)': 0})
    self.assertTrue(bus.send_t)

  def test_loop_sends_changed_commands(self) -> None:
    """Checks command formatting, de-duplication, and device-side parsing."""

    source = TestBlock()
    ctrl = ucontroller_module.UController(cmd_labels='freq')
    link(source, ctrl)

    with self._serial_patch():
      ctrl.prepare()
      bus = FakeMicrocontrollerSerial.instances[-1]

      source.send({'freq': 1.23456})
      ctrl.loop()
      source.send({'freq': 1.23456})
      ctrl.loop()
      source.send({'freq': 2})
      ctrl.loop()

    self.assertEqual(bus.writes, [
      b'go10\r\n',
      b'1freq\r\n',
      b'11.235\r\n',
      b'12.000\r\n',
    ])
    self.assertEqual(bus.received_commands, [('freq', 1.235), ('freq', 2.0)])

  def test_loop_receives_pc_timestamped_label(self) -> None:
    """Checks parsing a label frame and adding the PC timestamp."""

    ctrl = ucontroller_module.UController(
      labels='nr',
      init_output={'nr': 0},
      post_process={'nr': lambda value: 2 * value})
    ctrl._instance_t0 = Value('d', 10.0)
    link(ctrl, TestBlock())
    sent = self._capture_send(ctrl)

    with self._serial_patch():
      ctrl.prepare()
      bus = FakeMicrocontrollerSerial.instances[-1]
      bus.queue_label('nr', 4.0)

      with patch.object(ucontroller_module, 'time', return_value=12.5):
        ctrl.loop()

    self.assertEqual(len(sent), 1)
    self.assertEqual(set(sent[0]), {'nr', 't(s)'})
    self.assertAlmostEqual(sent[0]['nr'], 8.0)
    self.assertAlmostEqual(sent[0]['t(s)'], 2.5)

  def test_loop_receives_device_timestamped_label(self) -> None:
    """Checks parsing a timestamped frame emitted by the device."""

    ctrl = ucontroller_module.UController(labels=('a', 'b'),
                                          init_output={'a': 0, 'b': 99},
                                          t_device=True)
    link(ctrl, TestBlock())
    sent = self._capture_send(ctrl)

    with self._serial_patch():
      ctrl.prepare()
      bus = FakeMicrocontrollerSerial.instances[-1]
      bus.queue_label('a', 5.5, timestamp_ms=1234)

      ctrl.loop()

    self.assertEqual(len(sent), 1)
    self.assertEqual(set(sent[0]), {'a', 'b', 't(s)'})
    self.assertAlmostEqual(sent[0]['a'], 5.5)
    self.assertEqual(sent[0]['b'], 99)
    self.assertAlmostEqual(sent[0]['t(s)'], 1.234)

  def test_loop_buffers_incomplete_frames(self) -> None:
    """Checks that partial serial frames are kept for the next loop."""

    ctrl = ucontroller_module.UController(labels='nr',
                                          init_output={'nr': 0})
    ctrl._instance_t0 = Value('d', 1.0)
    link(ctrl, TestBlock())
    sent = self._capture_send(ctrl)

    raw = pack('<bf', 1, 7.0)

    with self._serial_patch():
      ctrl.prepare()
      bus = FakeMicrocontrollerSerial.instances[-1]

      bus.queue_raw(raw[:2])
      ctrl.loop()

      self.assertEqual(sent, [])
      self.assertEqual(ctrl._buffer, raw[:2])

      bus.queue_raw(raw[2:])
      with patch.object(ucontroller_module, 'time', return_value=2.0):
        ctrl.loop()

    self.assertEqual(len(sent), 1)
    self.assertAlmostEqual(sent[0]['nr'], 7.0)

  def test_loop_ignores_unknown_label_ids(self) -> None:
    """Checks that frames for unknown labels are discarded."""

    ctrl = ucontroller_module.UController(labels='nr',
                                          init_output={'nr': 0})
    link(ctrl, TestBlock())
    sent = self._capture_send(ctrl)

    with self._serial_patch():
      ctrl.prepare()
      FakeMicrocontrollerSerial.instances[-1].queue_raw(pack('<bf', 9, 3.0))

      ctrl.loop()

    self.assertEqual(sent, [])

  def test_finish_sends_stop_and_closes_serial_bus(self) -> None:
    """Checks that finish tells the device to stop and closes the port."""

    ctrl = ucontroller_module.UController(labels='nr',
                                          init_output={'nr': 0})
    link(ctrl, TestBlock())

    with self._serial_patch():
      ctrl.prepare()
      bus = FakeMicrocontrollerSerial.instances[-1]

      ctrl.finish()

    self.assertTrue(bus.stopped)
    self.assertTrue(bus.closed)
    self.assertEqual(bus.writes[-1], b'stop!\r\n')

  def test_prepare_surfaces_missing_pyserial(self) -> None:
    """Checks that missing pyserial does not break exception handling."""

    class MissingSerialException(Exception):
      """Fallback exception class mirroring the import-time fallback."""

    ctrl = ucontroller_module.UController(labels='nr',
                                          init_output={'nr': 0},
                                          port='missing')
    link(ctrl, TestBlock())

    with patch.multiple(ucontroller_module,
                        Serial=OptionalModule("pyserial"),
                        SerialException=MissingSerialException):
      self.assertIsNot(ucontroller_module.SerialException, Exception)
      with self.assertRaisesRegex(RuntimeError, "Missing module: pyserial"):
        ctrl.prepare()
