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


class IOBlockTestInOut(InOut):
  """InOut test double recording every call made by IOBlock."""

  instances: list['IOBlockTestInOut'] = list()

  def __init__(self, **kwargs) -> None:
    super().__init__()

    self.kwargs = kwargs
    self.data = list(kwargs.pop('data', list()))
    self.stream = list(kwargs.pop('stream', list()))
    self.open_error = kwargs.pop('open_error', None)
    self.close_error = kwargs.pop('close_error', None)
    self.start_error = kwargs.pop('start_error', None)
    self.zero_error = kwargs.pop('zero_error', None)
    self.stop_error = kwargs.pop('stop_error', None)
    self.cmd_error = kwargs.pop('cmd_error', None)

    self.calls = list()
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
    self.calls.append('open')
    if self.open_error is not None:
      raise self.open_error
    self.opened = True

  def close(self) -> None:
    self.calls.append('close')
    if self.close_error is not None:
      raise self.close_error
    self.closed = True

  def make_zero(self, delay: float) -> None:
    self.calls.append('make_zero')
    self.zero_delays.append(delay)
    if self.zero_error is not None:
      raise self.zero_error

  def set_cmd(self, *cmd) -> None:
    self.calls.append('set_cmd')
    self.commands.append(cmd)
    if self.cmd_error is not None:
      raise self.cmd_error

  def return_data(self):
    self.return_data_calls += 1
    return self.data.pop(0) if self.data else None

  def start_stream(self) -> None:
    self.calls.append('start_stream')
    self.start_stream_calls += 1
    if self.start_error is not None:
      raise self.start_error
    self.stream_started = True

  def return_stream(self):
    self.return_stream_calls += 1
    return self.stream.pop(0) if self.stream else None

  def stop_stream(self) -> None:
    self.calls.append('stop_stream')
    self.stop_stream_calls += 1
    self.stream_stopped = True
    if self.stop_error is not None:
      raise self.stop_error


class TestIOBlock(BlockTestBase):
  """Unit tests for the IOBlock Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Clears fake InOut instances before each test."""

    IOBlockTestInOut.reset()

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

  def test_unknown_and_unimported_collection_inouts_are_rejected(self) -> None:
    """Checks unknown names and collection drivers that are not registered."""

    with self.assertRaises(ValueError):
      IOBlock('MissingInOut')

    with patch.object(ioblock_module, 'moved_to_collection',
                      ('UnimportedCollectionInOut',)):
      with self.assertRaises(NotImplementedError):
        IOBlock('UnimportedCollectionInOut')

  def test_prepare_requires_links_and_command_labels(self) -> None:
    """Checks prepare-time Link layout validation."""

    block = self._make_block()
    with self.assertRaises(IOError):
      block.prepare()
    block.finish()
    self.assertEqual(IOBlockTestInOut.instances, [])

    source = TestBlock()
    block = self._make_block()
    link(source, block)

    with self.assertRaises(ValueError):
      block.prepare()
    block.finish()
    self.assertEqual(IOBlockTestInOut.instances, [])

  def test_finish_is_safe_after_device_construction_fails(self) -> None:
    """Checks that a failed constructor leaves no device to clean up."""

    error = RuntimeError('construction failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,))
    link(TestBlock(), block)

    with patch.object(IOBlockTestInOut, '__init__', side_effect=error):
      with self.assertRaises(RuntimeError) as caught:
        block.prepare()

    self.assertIs(caught.exception, error)
    block.finish()
    self.assertEqual(IOBlockTestInOut.instances, [])

  def test_failed_open_is_closed_without_sending_exit_command(self) -> None:
    """Checks the public driver boundary after a failed open call."""

    error = RuntimeError('open failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,),
                             open_error=error)
    link(TestBlock(), block)

    with self.assertRaises(RuntimeError) as caught:
      block.prepare()

    self.assertIs(caught.exception, error)
    device = IOBlockTestInOut.instances[-1]
    block.finish()
    block.finish()

    self.assertEqual(device.calls, ['open', 'close'])
    self.assertEqual(device.commands, [])
    self.assertTrue(device.closed)

  def test_finish_after_later_prepare_failure_sends_exit_command(self) -> None:
    """Checks cleanup after open succeeds but offsetting or a command fails."""

    for step in ('offset', 'initial command'):
      with self.subTest(step=step):
        error = RuntimeError(f'{step} failed')
        kwargs = ({'make_zero_delay': 0.25, 'zero_error': error}
                  if step == 'offset'
                  else {'initial_cmd': (1,), 'cmd_error': error})
        block = self._make_block(cmd_labels='cmd', exit_cmd=(0,), **kwargs)
        link(TestBlock(), block)
        link(block, TestBlock())

        with self.assertRaises(RuntimeError) as caught:
          block.prepare()

        self.assertIs(caught.exception, error)
        device = IOBlockTestInOut.instances[-1]
        device.cmd_error = None
        device.calls.clear()
        block.finish()

        self.assertEqual(device.calls, ['set_cmd', 'close'])
        self.assertEqual(device.commands[-1], (0,))
        self.assertTrue(device.closed)

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
    block._device_opened = True
    block._write = True
    block._stream_started = True

    block.finish()

    self.assertEqual(device.stop_stream_calls, 1)
    self.assertTrue(device.stream_stopped)
    self.assertEqual(device.commands, [(0,)])
    self.assertTrue(device.closed)
    self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])

    block.finish()
    self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])

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
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,), streamer=True)
    device = IOBlockTestInOut(stop_error=error)
    block._device = device
    block._device_opened = True
    block._write = True
    block._stream_started = True

    with self.assertRaises(RuntimeError) as caught:
      block.finish()

    self.assertIs(caught.exception, error)
    self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])
    self.assertEqual(device.commands, [(0,)])
    self.assertTrue(device.closed)

  def test_finish_closes_when_exit_command_raises(self) -> None:
    """Checks cleanup if sending the exit command fails."""

    error = RuntimeError('command failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,))
    device = IOBlockTestInOut(cmd_error=error)
    block._device = device
    block._device_opened = True
    block._write = True

    with self.assertRaises(RuntimeError) as caught:
      block.finish()

    self.assertIs(caught.exception, error)
    self.assertEqual(device.commands, [(0,)])
    self.assertTrue(device.closed)

  def test_failed_stream_start_is_closed_without_stopping_stream(self) -> None:
    """Checks that IOBlock only stops streams whose start call succeeded."""

    error = RuntimeError('start failed')
    block = self._make_block(streamer=True, start_error=error)
    link(block, TestBlock())
    block.prepare()
    device = IOBlockTestInOut.instances[-1]

    with self.assertRaises(RuntimeError) as caught:
      block._read_data()

    self.assertIs(caught.exception, error)
    block.finish()

    self.assertEqual(device.calls, ['open', 'start_stream', 'close'])
    self.assertEqual(device.stop_stream_calls, 0)
    self.assertTrue(device.closed)

  def test_finish_reports_every_cleanup_failure_after_all_attempts(
      self) -> None:
    """Checks all cleanup steps run before their noted failures are grouped."""

    stop_error = RuntimeError('stop failed')
    cmd_error = ValueError('exit command failed')
    close_error = OSError('close failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,), streamer=True,
                             stop_error=stop_error, cmd_error=cmd_error,
                             close_error=close_error)
    link(TestBlock(), block)
    block.prepare()
    device = IOBlockTestInOut.instances[-1]
    block._read_data()
    device.calls.clear()

    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()

    self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])
    self.assertEqual(caught.exception.message, 'InOut cleanup failures')
    self.assertEqual(caught.exception.exceptions,
                     (stop_error, cmd_error, close_error))
    for error, operation in zip(
        caught.exception.exceptions,
        ('stop stream', 'set exit command', 'close device')):
      self.assertIsNotNone(error.__traceback__)
      self.assertIn(f'IOBlockTestInOut IOBlock cleanup step: {operation}',
                    error.__notes__)

  def test_finish_retries_failed_close_without_repeating_completed_steps(
      self) -> None:
    """Checks that successful cleanup is not repeated when close is retried."""

    error = OSError('close failed')
    block = self._make_block(cmd_labels='cmd', exit_cmd=(0,), streamer=True,
                             close_error=error)
    link(TestBlock(), block)
    block.prepare()
    device = IOBlockTestInOut.instances[-1]
    block._read_data()
    device.calls.clear()

    with self.assertRaises(OSError) as caught:
      block.finish()

    self.assertIs(caught.exception, error)
    self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])

    device.close_error = None
    block.finish()
    block.finish()

    self.assertEqual(device.calls,
                     ['stop_stream', 'set_cmd', 'close', 'close'])
    self.assertTrue(device.closed)

  def test_finish_completes_cleanup_before_reraising_keyboard_interrupt(
      self) -> None:
    """Checks that interrupts remain recognizable after remaining cleanup."""

    for step in ('stop_error', 'cmd_error', 'close_error'):
      with self.subTest(step=step):
        interrupt = KeyboardInterrupt()
        block = self._make_block(cmd_labels='cmd', exit_cmd=(0,),
                                 streamer=True, **{step: interrupt})
        link(TestBlock(), block)
        block.prepare()
        device = IOBlockTestInOut.instances[-1]
        block._read_data()
        device.calls.clear()

        with self.assertRaises(KeyboardInterrupt) as caught:
          block.finish()

        self.assertIs(caught.exception, interrupt)
        self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])

  def test_finish_preserves_other_failures_when_interrupted(self) -> None:
    """Checks the original interrupt is raised with noted failures as cause."""

    operations = ('stop stream', 'set exit command', 'close device')
    for index, operation in enumerate(operations):
      with self.subTest(operation=operation):
        interrupt = KeyboardInterrupt()
        errors: list[Exception | KeyboardInterrupt] = [
          RuntimeError('stop failed'),
          ValueError('exit command failed'),
          OSError('close failed')]
        errors[index] = interrupt
        block = self._make_block(cmd_labels='cmd', exit_cmd=(0,),
                                 streamer=True, stop_error=errors[0],
                                 cmd_error=errors[1], close_error=errors[2])
        link(TestBlock(), block)
        block.prepare()
        device = IOBlockTestInOut.instances[-1]
        block._read_data()
        device.calls.clear()

        with self.assertRaises(KeyboardInterrupt) as caught:
          block.finish()

        self.assertIs(caught.exception, interrupt)
        self.assertEqual(device.calls, ['stop_stream', 'set_cmd', 'close'])
        group = interrupt.__cause__
        self.assertIsInstance(group, ExceptionGroup)
        self.assertEqual(group.message, 'Other InOut cleanup failures')
        self.assertEqual(group.exceptions,
                         tuple(errors[:index] + errors[index + 1:]))
        for error in group.exceptions:
          self.assertIsNotNone(error.__traceback__)
        for error, step in zip(errors, operations):
          self.assertIn(f'IOBlockTestInOut IOBlock cleanup step: {step}',
                        error.__notes__)
