# coding: utf-8

from multiprocessing import Value
from typing import Any
from unittest.mock import patch

import crappy.blocks.generator as generator_module
from crappy._global import GeneratorStop
from crappy.blocks.generator import Generator
from crappy.blocks.generator_path.meta_path import Path

from ..block import BlockTestBase


Path.classes.pop('GeneratorUnitPath', None)
Path.classes.pop('GeneratorProbePath', None)


class GeneratorUnitPath(Path):
  """Small Generator Path double returning scripted commands."""

  def __init__(self, commands: list[Any]) -> None:
    super().__init__()

    self.commands = list(commands)

  def get_cmd(self, data: dict[str, list]) -> Any:
    """Returns the next scripted command or stops the path."""

    command = self.commands.pop(0)
    if command == 'stop':
      raise StopIteration
    if callable(command):
      return command(data)
    return command


class GeneratorProbePath(Path):
  """Path double recording the class-level state provided at construction."""

  instances: list['GeneratorProbePath'] = list()

  def __init__(self, command: Any = 0) -> None:
    super().__init__()

    self.command = command
    self.start_t0 = self.t0
    self.start_last_cmd = self.last_cmd
    self.instances.append(self)

  @classmethod
  def reset(cls) -> None:
    """Clears shared test state."""

    cls.instances = list()

  def get_cmd(self, data: dict[str, list]) -> Any:
    """Returns the configured command."""

    return self.command


class TestGenerator(BlockTestBase):
  """Unit tests for the Generator Block-specific behavior."""

  _t0 = 10.0

  def setUp(self) -> None:
    """Resets Generator Path class-level state before each test."""

    GeneratorProbePath.reset()
    Path.t0 = 0
    Path.last_cmd = None

  def tearDown(self) -> None:
    """Resets Generator Path state and lets the Block harness clean up."""

    Path.t0 = 0
    Path.last_cmd = None
    super().tearDown()

  def _make_generator(self,
                      path: list[dict[str, Any]] | None = None,
                      **kwargs) -> Generator:
    """Creates a Generator ready for direct loop calls."""

    if path is None:
      path = [{'type': 'GeneratorUnitPath', 'commands': [0]}]

    kwargs.setdefault('freq', None)
    kwargs.setdefault('debug', None)

    generator = Generator(path, **kwargs)
    generator._instance_t0 = Value('d', self._t0)
    return generator

  @staticmethod
  def _capture_send(generator: Generator) -> list[list[Any]]:
    """Captures values sent by the Generator."""

    sent = list()

    def send(data: list[Any]) -> None:
      sent.append(list(data))

    generator.send = send
    return sent

  @staticmethod
  def _set_received_batches(generator: Generator,
                            batches: list[dict[str, list]]
                            ) -> list[None]:
    """Makes recv_all_data return deterministic batches."""

    calls = list()
    batches_iter = iter(batches)

    def recv_all_data() -> dict[str, list]:
      calls.append(None)
      return dict(next(batches_iter))

    generator.recv_all_data = recv_all_data
    return calls

  @staticmethod
  def _set_available(generator: Generator,
                     values: list[bool]) -> list[None]:
    """Makes data_available return deterministic values."""

    calls = list()
    values_iter = iter(values)

    def data_available() -> bool:
      calls.append(None)
      return next(values_iter)

    generator.data_available = data_available
    return calls

  def test_constructor_sets_block_options(self) -> None:
    """Checks Generator-specific initialization."""

    generator = self._make_generator(cmd_label='drive',
                                     path_index_label='phase',
                                     spam=True,
                                     safe_start=True,
                                     end_delay=None,
                                     display_freq=True,
                                     debug=True)

    self.assertEqual(generator.labels, ['t(s)', 'drive', 'phase'])
    self.assertIsNone(generator.freq)
    self.assertTrue(generator.display_freq)
    self.assertTrue(generator.debug)
    self.assertTrue(generator._spam)
    self.assertTrue(generator._safe_start)
    self.assertIsNone(generator._end_delay)

  def test_constructor_validates_path_sequence(self) -> None:
    """Checks invalid Generator path declarations fail early."""

    cases = (
      ([], ValueError),
      ([{'type': 'GeneratorUnitPath', 'commands': [0]}, 'bad'], TypeError),
      ([{}], ValueError),
      ([{'type': 'MissingPath'}], ValueError),
      ([{'type': 'Constant', 'condition': None}], ValueError),
    )

    for path, exception in cases:
      with self.subTest(path=path):
        with self.assertRaises(exception):
          Generator(path, freq=None)

  def test_begin_instantiates_first_path_with_shared_state(self) -> None:
    """Checks first Path setup from the Generator start time."""

    generator = self._make_generator([
      {'type': 'GeneratorProbePath', 'command': 3},
    ])
    GeneratorProbePath.reset()

    generator.begin()

    path = GeneratorProbePath.instances[-1]
    self.assertIs(generator._current_path, path)
    self.assertEqual(generator._path_id, 0)
    self.assertEqual(path.start_t0, self._t0)
    self.assertIsNone(path.start_last_cmd)

  def test_loop_sends_changed_command_with_relative_time_and_path_id(
      self) -> None:
    """Checks the normal command emission payload."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [5]},
    ])
    generator.begin()
    sent = self._capture_send(generator)
    recv_calls = self._set_received_batches(generator, [{'x': [1]}])

    with patch.object(generator_module, 'time', return_value=12.5):
      generator.loop()

    self.assertEqual(recv_calls, [None])
    self.assertEqual(sent, [[2.5, 5, 0]])
    self.assertEqual(generator._last_cmd, 5)
    self.assertEqual(generator._last_id, 0)

  def test_loop_does_not_send_none_commands(self) -> None:
    """Checks that a Path can deliberately skip output for one loop."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [None]},
    ])
    generator.begin()
    sent = self._capture_send(generator)
    self._set_received_batches(generator, [{}])

    with patch.object(generator_module, 'time', return_value=11):
      generator.loop()

    self.assertEqual(sent, [])
    self.assertIsNone(generator._last_cmd)
    self.assertIsNone(generator._last_id)

  def test_duplicate_commands_are_suppressed_unless_spam_is_enabled(
      self) -> None:
    """Checks duplicate filtering and the spam override."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [1, 1]},
    ])
    generator.begin()
    sent = self._capture_send(generator)
    self._set_received_batches(generator, [{}, {}])

    with patch.object(generator_module, 'time', side_effect=(11, 12)):
      generator.loop()
      generator.loop()

    self.assertEqual(sent, [[1.0, 1, 0]])

    spam_generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [1, 1]},
    ], spam=True)
    spam_generator.begin()
    spam_sent = self._capture_send(spam_generator)
    self._set_received_batches(spam_generator, [{}, {}])

    with patch.object(generator_module, 'time', side_effect=(11, 12)):
      spam_generator.loop()
      spam_generator.loop()

    self.assertEqual(spam_sent, [[1.0, 1, 0], [2.0, 1, 0]])

  def test_safe_start_waits_for_first_available_data(self) -> None:
    """Checks safe-start gating before the first output."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [3]},
    ], safe_start=True)
    generator.begin()
    sent = self._capture_send(generator)
    availability_calls = self._set_available(generator, [False, True])
    recv_calls = self._set_received_batches(generator, [{}])

    generator.loop()

    self.assertEqual(sent, [])
    self.assertEqual(recv_calls, [])

    with patch.object(generator_module, 'time', return_value=12):
      generator.loop()

    self.assertEqual(availability_calls, [None, None])
    self.assertEqual(recv_calls, [None])
    self.assertEqual(sent, [[2.0, 3, 0]])

  def test_path_transition_restarts_loop_with_latest_data(self) -> None:
    """Checks recursive transition behavior between Paths."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': ['stop']},
      {'type': 'GeneratorUnitPath',
       'commands': [lambda data: data['phase'][-1]]},
    ])
    generator.begin()
    sent = self._capture_send(generator)
    recv_calls = self._set_received_batches(generator, [
      {'phase': [1]},
      {'phase': [2]},
    ])

    with patch.object(generator_module, 'time', side_effect=(11, 12)):
      generator.loop()

    self.assertEqual(recv_calls, [None, None])
    self.assertEqual(sent, [[2.0, 2, 1]])

  def test_exhaustion_with_end_delay_raises_generator_stop(self) -> None:
    """Checks default terminal behavior when all Paths are exhausted."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': ['stop']},
    ], end_delay=0)
    generator.begin()
    self._set_received_batches(generator, [{}])

    with (patch.object(generator_module, 'time', return_value=11),
          patch.object(generator_module, 'sleep') as sleep_mock):
      with self.assertRaises(GeneratorStop):
        generator.loop()

    sleep_mock.assert_called_once_with(0)

  def test_exhaustion_without_end_delay_stays_idle(self) -> None:
    """Checks non-stopping terminal behavior when end_delay is None."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': ['stop']},
    ], end_delay=None)
    generator.begin()
    sent = self._capture_send(generator)
    recv_calls = self._set_received_batches(generator, [{}])

    with patch.object(generator_module, 'time', return_value=11):
      generator.loop()

    self.assertEqual(sent, [])
    self.assertTrue(generator._ended_no_raise)
    self.assertEqual(recv_calls, [None])

    def fail_on_recv() -> dict[str, list]:
      raise AssertionError('ended Generator should not read inputs')

    generator.recv_all_data = fail_on_recv
    generator.loop()

  def test_repeat_restarts_path_and_keeps_incrementing_ids(self) -> None:
    """Checks repeat mode and monotonically increasing path ids."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [1, 'stop']},
    ], repeat=True)
    generator.begin()
    sent = self._capture_send(generator)
    self._set_received_batches(generator, [{}, {}, {}])

    with patch.object(generator_module, 'time', side_effect=(11, 12, 13)):
      generator.loop()
      generator.loop()

    self.assertEqual(sent, [[1.0, 1, 0], [3.0, 1, 1]])
    self.assertEqual(generator._path_id, 1)

  def test_next_path_receives_previous_time_and_command(self) -> None:
    """Checks Path.t0 and Path.last_cmd handoff during transitions."""

    generator = self._make_generator([
      {'type': 'GeneratorUnitPath', 'commands': [5, 'stop']},
      {'type': 'GeneratorProbePath', 'command': 6},
    ])
    GeneratorProbePath.reset()
    generator.begin()
    sent = self._capture_send(generator)
    self._set_received_batches(generator, [{}, {}, {}])

    with patch.object(generator_module, 'time', side_effect=(12, 13, 14)):
      generator.loop()
      generator.loop()

    path = GeneratorProbePath.instances[-1]
    self.assertEqual(path.start_t0, 13)
    self.assertEqual(path.start_last_cmd, 5)
    self.assertEqual(sent, [[2.0, 5, 0], [4.0, 6, 1]])
