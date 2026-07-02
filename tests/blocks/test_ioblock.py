# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch

import numpy as np

import crappy.blocks.ioblock as ioblock_module
from crappy.blocks.ioblock import IOBlock
from crappy.inout.meta_inout import InOut

from ..block import BlockTestBase, TestBlock, link


InOut.classes.pop('IOBlockTestInOut', None)
InOut.classes.pop('IOBlockFT232HInOut', None)


class IOBlockTestInOut(InOut):
  """InOut test double recording every call made by IOBlock."""

  instances: list['IOBlockTestInOut'] = list()

  def __init__(self, **kwargs) -> None:
    super().__init__()

    self.kwargs = kwargs
    self.data = list(kwargs.pop('data', list()))
    self.stream = list(kwargs.pop('stream', list()))
    self.stop_error = kwargs.pop('stop_error', None)
    self.cmd_error = kwargs.pop('cmd_error', None)

    self.opened = False
    self.closed = False
    self.stream_started = False
    self.stream_stopped = False
    self.commands = list()
    self.zero_delays = list()
    self.return_data_calls = 0
    self.return_stream_calls = 0
    self.start_stream_calls = 0
    self.stop_stream_calls = 0

    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears shared state between tests."""

    cls.instances = list()

  def open(self) -> None:
    self.opened = True

  def close(self) -> None:
    self.closed = True

  def make_zero(self, delay: float) -> None:
    self.zero_delays.append(delay)

  def set_cmd(self, *cmd) -> None:
    self.commands.append(cmd)
    if self.cmd_error is not None:
      raise self.cmd_error

  def return_data(self):
    self.return_data_calls += 1
    return self.data.pop(0) if self.data else None

  def start_stream(self) -> None:
    self.start_stream_calls += 1
    self.stream_started = True

  def return_stream(self):
    self.return_stream_calls += 1
    return self.stream.pop(0) if self.stream else None

  def stop_stream(self) -> None:
    self.stop_stream_calls += 1
    self.stream_stopped = True
    if self.stop_error is not None:
      raise self.stop_error


class IOBlockFT232HInOut(IOBlockTestInOut):
  """FT232H-enabled test double."""

  ft232h = True


class TestIOBlock(BlockTestBase):
  """Unit tests for the IOBlock Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Clears fake InOut instances before each test."""

    IOBlockTestInOut.reset()
    IOBlockFT232HInOut.reset()

  def _make_block(self, **kwargs) -> IOBlock:
    """Creates an IOBlock ready for direct loop calls."""

    kwargs.setdefault('freq', None)
    block = IOBlock('IOBlockTestInOut', **kwargs)
    block._instance_t0 = Value('d', self._t0)
    return block

  @staticmethod
  def _capture_send(block: IOBlock) -> list[Any]:
    """Captures values sent by IOBlock."""

    sent = list()

    def send(data) -> None:
      if isinstance(data, dict):
        sent.append(dict(data))
      else:
        sent.append(list(data))

    block.send = send
    return sent

  @staticmethod
  def _set_received(block: IOBlock,
                    data: dict[str, Any]) -> list[bool]:
    """Makes recv_last_data return deterministic data."""

    fill_missing_values = list()

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      fill_missing_values.append(fill_missing)
      return dict(data)

    block.recv_last_data = recv_last_data
    return fill_missing_values

  def test_labels_and_command_labels_normalization(self) -> None:
    """Checks supported labels and cmd_labels forms."""

    block = self._make_block(labels='mem',
                             cmd_labels='target',
                             initial_cmd='init',
                             exit_cmd='exit')

    self.assertEqual(block.labels, ['mem'])
    self.assertEqual(block._cmd_labels, ['target'])
    self.assertEqual(block._initial_cmd, ['init'])
    self.assertEqual(block._exit_cmd, ['exit'])

    block = self._make_block(labels=('t(s)', 'value'),
                             cmd_labels=('a', 'b'),
                             initial_cmd=(1, 2),
                             exit_cmd=(3, 4))

    self.assertEqual(block.labels, ['t(s)', 'value'])
    self.assertEqual(block._cmd_labels, ['a', 'b'])
    self.assertEqual(block._initial_cmd, [1, 2])
    self.assertEqual(block._exit_cmd, [3, 4])

    block = self._make_block(streamer=True)

    self.assertEqual(block.labels, ['t(s)', 'stream'])
    self.assertEqual(block._cmd_labels, [])

  def test_command_value_counts_are_validated(self) -> None:
    """Checks validation for initial and exit commands."""

    with self.assertRaises(ValueError):
      self._make_block(cmd_labels=('a', 'b'), initial_cmd=(1,))

    with self.assertRaises(ValueError):
      self._make_block(cmd_labels=('a', 'b'), exit_cmd=(1,))

    # Without command labels, commands are only stored and never applied.
    block = self._make_block(initial_cmd=(1,), exit_cmd=(2,))

    self.assertEqual(block._initial_cmd, [1])
    self.assertEqual(block._exit_cmd, [2])
    self.assertFalse(block._cmd_labels)

  def test_unknown_and_deprecated_inouts_are_rejected(self) -> None:
    """Checks InOut name validation."""

    with self.assertRaises(ValueError):
      IOBlock('MissingInOut')

    with patch.dict(ioblock_module.deprecated_inouts,
                    {'OldInOut': 'NewInOut'}):
      with self.assertRaises(NotImplementedError):
        IOBlock('OldInOut')

  def test_prepare_requires_links_and_command_labels(self) -> None:
    """Checks prepare-time Link layout validation."""

    with self.assertRaises(IOError):
      self._make_block().prepare()

    source = TestBlock()
    block = self._make_block()
    link(source, block)

    with self.assertRaises(ValueError):
      block.prepare()

  def test_prepare_opens_offsets_and_sends_initial_command(self) -> None:
    """Checks the main prepare side effects."""

    source = TestBlock()
    sink = TestBlock()
    block = self._make_block(labels=('t(s)', 'value'),
                             cmd_labels=('a', 'b'),
                             initial_cmd=(1, 2),
                             make_zero_delay=0.25,
                             extra='kept')
    link(source, block)
    link(block, sink)

    block.prepare()
    device = IOBlockTestInOut.instances[-1]

    self.assertIs(block._device, device)
    self.assertTrue(device.opened)
    self.assertEqual(device.kwargs, {'extra': 'kept'})
    self.assertEqual(device.zero_delays, [0.25])
    self.assertEqual(device.commands, [(1, 2)])
    self.assertTrue(block._read)
    self.assertTrue(block._write)
    self.assertEqual(block._last_cmd, [1, 2])
    self.assertEqual(block._prev_values, {'a': 1, 'b': 2})

  def test_ft232h_inout_gets_registered_connection_arguments(self) -> None:
    """Checks FT232H registration and constructor forwarding."""

    source = TestBlock()

    with patch.object(ioblock_module.USBServer, 'register',
                      return_value=('server', 'args')) as register:
      block = IOBlock('IOBlockFT232HInOut',
                      cmd_labels='cmd',
                      ft232h_ser_num='ABC',
                      freq=None,
                      option=1)
      link(source, block)
      block.prepare()

    device = IOBlockFT232HInOut.instances[-1]

    register.assert_called_once_with('ABC')
    self.assertEqual(device.kwargs, {'option': 1,
                                     '_ft232h_args': ('server', 'args')})

  def test_loop_reads_iterable_data_and_offsets_time(self) -> None:
    """Checks regular acquisition from iterable data."""

    block = self._make_block(labels=('t(s)', 'value'))
    device = IOBlockTestInOut(data=[[12.5, 3.0]])
    sent = self._capture_send(block)
    self._set_received(block, dict())
    block._device = device
    block._read = True

    block.loop()

    self.assertEqual(device.return_data_calls, 1)
    self.assertEqual(sent, [[2.5, 3.0]])

  def test_loop_reads_dict_data_and_offsets_time(self) -> None:
    """Checks regular acquisition from dict data."""

    block = self._make_block()
    device = IOBlockTestInOut(data=[{'t(s)': 13.0, 'value': 5.0}])
    sent = self._capture_send(block)
    self._set_received(block, dict())
    block._device = device
    block._read = True

    block.loop()

    self.assertEqual(device.return_data_calls, 1)
    self.assertEqual(sent, [{'t(s)': 3.0, 'value': 5.0}])

  def test_loop_does_not_send_when_no_data_is_available(self) -> None:
    """Checks that None reads are silently ignored."""

    block = self._make_block(labels=('t(s)', 'value'))
    device = IOBlockTestInOut(data=[None])
    sent = self._capture_send(block)
    self._set_received(block, dict())
    block._device = device
    block._read = True

    block.loop()

    self.assertEqual(device.return_data_calls, 1)
    self.assertEqual(sent, [])

  def test_trigger_label_controls_acquisition(self) -> None:
    """Checks that trigger_label gates reads without being filled."""

    block = self._make_block(trigger_label='trig')
    device = IOBlockTestInOut(data=[{'t(s)': 14.0, 'value': 1.0}])
    sent = self._capture_send(block)
    block._device = device
    block._read = True

    self._set_received(block, {'cmd': 1})
    block.loop()

    self.assertEqual(device.return_data_calls, 0)
    self.assertEqual(sent, [])

    self._set_received(block, {'trig': True})
    block.loop()

    self.assertEqual(device.return_data_calls, 1)
    self.assertEqual(sent, [{'t(s)': 4.0, 'value': 1.0}])

  def test_loop_starts_stream_once_and_sends_stream_data(self) -> None:
    """Checks streamer acquisition lifecycle."""

    stream = [np.array([11.0, 12.0]), np.array([[1.0], [2.0]])]
    block = self._make_block(streamer=True)
    device = IOBlockTestInOut(stream=[stream])
    sent = self._capture_send(block)
    self._set_received(block, dict())
    block._device = device
    block._read = True

    block.loop()
    block.loop()

    self.assertEqual(device.start_stream_calls, 1)
    self.assertEqual(device.return_stream_calls, 2)
    self.assertTrue(block._stream_started)
    self.assertEqual(len(sent), 1)
    np.testing.assert_array_equal(sent[0][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(sent[0][1], np.array([[1.0], [2.0]]))

  def test_loop_writes_complete_commands_in_label_order(self) -> None:
    """Checks command collection, ordering, and previous value filling."""

    block = self._make_block(cmd_labels=('a', 'b'))
    device = IOBlockTestInOut()
    block._device = device
    block._write = True

    self._set_received(block, {'a': 1})
    block.loop()
    self.assertEqual(device.commands, [])

    self._set_received(block, {'b': 2})
    block.loop()
    self.assertEqual(device.commands, [(1, 2)])

    self._set_received(block, {'a': 3, 'other': 9})
    block.loop()
    self.assertEqual(device.commands, [(1, 2), (3, 2)])

  def test_loop_suppresses_duplicate_commands_unless_spamming(self) -> None:
    """Checks duplicate command filtering and spam behavior."""

    block = self._make_block(cmd_labels='cmd')
    device = IOBlockTestInOut()
    block._device = device
    block._write = True

    self._set_received(block, {'cmd': 1})
    block.loop()
    block.loop()

    self.assertEqual(device.commands, [(1,)])

    block = self._make_block(cmd_labels='cmd', spam=True)
    device = IOBlockTestInOut()
    block._device = device
    block._write = True

    self._set_received(block, {'cmd': 1})
    block.loop()
    block.loop()

    self.assertEqual(device.commands, [(1,), (1,)])

  def test_finish_stops_started_stream_sends_exit_and_closes(self) -> None:
    """Checks the normal finish sequence."""

    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,), streamer=True)
    device = IOBlockTestInOut()
    block._device = device
    block._write = True
    block._stream_started = True

    block.finish()

    self.assertEqual(device.stop_stream_calls, 1)
    self.assertTrue(device.stream_stopped)
    self.assertEqual(device.commands, [(0,)])
    self.assertTrue(device.closed)

  def test_finish_does_not_stop_stream_that_never_started(self) -> None:
    """Checks that streamer finish does not call stop_stream unnecessarily."""

    block = self._make_block(streamer=True)
    device = IOBlockTestInOut()
    block._device = device

    block.finish()

    self.assertEqual(device.stop_stream_calls, 0)
    self.assertTrue(device.closed)

  def test_finish_closes_when_stop_stream_raises(self) -> None:
    """Checks cleanup if stopping the stream fails."""

    error = RuntimeError('stop failed')
    block = self._make_block(streamer=True)
    device = IOBlockTestInOut(stop_error=error)
    block._device = device
    block._stream_started = True

    with self.assertRaises(RuntimeError):
      block.finish()

    self.assertTrue(device.closed)

  def test_finish_closes_when_exit_command_raises(self) -> None:
    """Checks cleanup if sending the exit command fails."""

    error = RuntimeError('command failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,))
    device = IOBlockTestInOut(cmd_error=error)
    block._device = device
    block._write = True

    with self.assertRaises(RuntimeError):
      block.finish()

    self.assertEqual(device.commands, [(0,)])
    self.assertTrue(device.closed)
