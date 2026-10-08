# coding: utf-8

import logging
from multiprocessing import Event, Value
from typing import Any
from unittest.mock import Mock, call, patch

from crappy.blocks.pause import Pause
import crappy.blocks.pause as pause_module

from ..block import BlockTestBase, TestBlock, link


class TestPause(BlockTestBase):
  """Unit tests for the Pause Block-specific behavior."""

  _t0 = 10.0

  def _make_pause(self,
                  criteria,
                  batches: list[dict[str, list[Any]]] | None = None
                  ) -> tuple[Pause, list[tuple[int, str]], list[None]]:
    """Creates a linked Pause with deterministic data and synchronization."""

    source = TestBlock()
    pause = Pause(criteria)
    link(source, pause)

    pause._pause_event = Event()
    pause._instance_t0 = Value('d', self._t0)

    logs = list()
    recv_calls = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    pause.log = log

    if batches is not None:
      batches_iter = iter(batches)

      def recv_all_data() -> dict[str, list[Any]]:
        recv_calls.append(None)
        return dict(next(batches_iter))

      pause.recv_all_data = recv_all_data

    pause.prepare()
    return pause, logs, recv_calls

  def test_init_sets_block_options_and_normalizes_criteria(self) -> None:
    """Checks Pause-specific initialization."""

    def criterion(_):
      return False

    pause = Pause('value>8', freq=None, display_freq=True, debug=True)

    self.assertFalse(pause.pausable)
    self.assertIsNone(pause.freq)
    self.assertTrue(pause.display_freq)
    self.assertTrue(pause.debug)
    self.assertEqual(pause._raw_crit, ('value>8',))
    self.assertIsNone(pause._criteria)

    self.assertEqual(Pause(criterion)._raw_crit, (criterion,))
    self.assertEqual(Pause(('value>8', criterion))._raw_crit,
                     ('value>8', criterion))

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Pause without input Links fails early."""

    pause = Pause('value>8')

    with self.assertRaises(IOError):
      pause.prepare()

  def test_prepare_converts_criteria_to_callables(self) -> None:
    """Checks that prepare parses all raw criteria."""

    def criterion(_):
      return True

    pause, _, _ = self._make_pause(('value>8', criterion))

    self.assertEqual(len(pause._criteria), 2)
    self.assertTrue(all(callable(crit) for crit in pause._criteria))
    self.assertIs(pause._criteria[1], criterion)
    self.assertEqual(pause._criterion_states, [False, False])

  def test_prepare_resets_criterion_states(self) -> None:
    """Checks that preparing initializes each criterion as not met."""

    pause, _, _ = self._make_pause('value>8', batches=[{'value': [9]}])
    pause.loop()

    pause.prepare()

    self.assertEqual(pause._criterion_states, [False])

  def test_prepare_rejects_wrong_string_syntax(self) -> None:
    """Checks that invalid criterion strings fail during prepare."""

    source = TestBlock()
    pause = Pause('value=8')
    link(source, pause)

    with self.assertRaises(ValueError):
      pause.prepare()

  def test_string_criteria_ignore_spaces_around_operands(self) -> None:
    """Checks documented criterion parsing with optional whitespace."""

    pause, _, _ = self._make_pause((' value < 3 ', ' value > 7 '))
    less_than, greater_than = pause._criteria

    self.assertTrue(less_than({'value': [4, 2]}))
    self.assertFalse(less_than({'value': [3, 4]}))
    self.assertIsNone(less_than({' value': [2]}))

    self.assertTrue(greater_than({'value': [7, 8]}))
    self.assertFalse(greater_than({'value': [6, 7]}))
    self.assertIsNone(greater_than({' value': [8]}))

  def test_string_criteria_return_none_without_values(self) -> None:
    """Checks that absent labels and empty batches are insufficient data."""

    pause, _, _ = self._make_pause(('value<3', 'value>7'))

    for criterion in pause._criteria:
      for data in ({}, {'other': [9]}, {'value': []}):
        with self.subTest(criterion=criterion, data=data):
          self.assertIsNone(criterion(data))

  def test_loop_stays_quiet_without_matching_data(self) -> None:
    """Checks that missing data does not set or clear the pause Event."""

    pause, logs, recv_calls = self._make_pause('value>8', batches=[{}])

    pause.loop()

    self.assertEqual(recv_calls, [None])
    self.assertFalse(pause._pause_event.is_set())
    self.assertIn((logging.DEBUG, 'No data received during this loop'), logs)
    self.assertIn((logging.DEBUG, 'No pausing or un-pausing during this loop'),
                  logs)

  def test_loop_pauses_when_any_criterion_is_met(self) -> None:
    """Checks that matching data sets the shared pause Event."""

    pause, logs, _ = self._make_pause(('value>8', 'other<0'),
                                      batches=[{'value': [1, 9]}])

    pause.loop()

    self.assertTrue(pause._pause_event.is_set())
    self.assertIn((logging.WARNING,
                   'Pause criterion reached, pausing the Blocks!'), logs)

  def test_loop_unpauses_when_criteria_are_no_longer_met(self) -> None:
    """Checks that non-matching data clears an active pause Event."""

    pause, logs, _ = self._make_pause('value>8',
                                      batches=[
                                        {'value': [10]},
                                        {'value': [7]},
                                      ])

    pause.loop()
    self.assertTrue(pause._pause_event.is_set())

    pause.loop()

    self.assertFalse(pause._pause_event.is_set())
    self.assertIn((logging.WARNING,
                   'Pause criterion no longer satisfied, '
                   'un-pausing the Blocks!'), logs)

  def test_loop_retains_state_without_values_for_criterion(self) -> None:
    """Checks that only new samples can change a data criterion's state."""

    for criterion, trigger in (('value>8', 9), ('value<3', 2)):
      batches = [{'value': [trigger]}, {}, {'other': [0]}, {'value': []},
                 {'value': [5]}, {}, {'value': []}]
      pause, _, recv_calls = self._make_pause(criterion, batches=batches)

      for batch, expected in zip(batches,
                                 (True, True, True, True, False, False, False)):
        with self.subTest(criterion=criterion, batch=batch):
          pause.loop()
          self.assertEqual(pause._pause_event.is_set(), expected)

      self.assertEqual(len(recv_calls), len(batches))

  def test_loop_tracks_criteria_independently(self) -> None:
    """Checks pause state when criterion labels arrive independently."""

    batches = [{'value': [9]}, {'other': [-1]}, {'value': [7]}, {},
               {'other': [1]}, {'value': [9]}, {'other': [1]}, {'value': [7]}]
    pause, _, _ = self._make_pause(('value>8', 'other<0'), batches=batches)

    for batch, expected in zip(batches,
                               (True, True, True, True, False, True, True,
                                False)):
      with self.subTest(batch=batch):
        pause.loop()
        self.assertEqual(pause._pause_event.is_set(), expected)

  def test_loop_keeps_current_state_when_no_transition_is_needed(self) -> None:
    """Checks already-paused and already-unpaused steady states."""

    pause, _, _ = self._make_pause('value>8', batches=[{'value': [9]}])
    pause._pause_event.set()

    pause.loop()

    self.assertTrue(pause._pause_event.is_set())

    pause, _, _ = self._make_pause('value>8', batches=[{'value': [8]}])

    pause.loop()

    self.assertFalse(pause._pause_event.is_set())

  def test_time_criterion_can_pause_without_received_data(self) -> None:
    """Checks time-based criteria even when no upstream payload is available."""

    pause, _, _ = self._make_pause('t(s)>5', batches=[{}])

    with patch.object(pause_module, 'time',
                      return_value=self._t0 + 6) as clock:
      pause.loop()

    clock.assert_called_once_with()
    self.assertTrue(pause._pause_event.is_set())

  def test_time_criterion_uses_elapsed_time_threshold(self) -> None:
    """Checks the false case for time-based criteria."""

    pause, _, _ = self._make_pause('t(s)>5', batches=[{}])

    with patch.object(pause_module, 'time',
                      return_value=self._t0 + 5) as clock:
      pause.loop()

    clock.assert_called_once_with()
    self.assertFalse(pause._pause_event.is_set())

  def test_time_criterion_runs_every_loop_after_met_data_criterion(self) -> None:
    """Checks time despite an earlier true criterion and missing data."""

    batches = [{'value': [9]}, {}, {'value': [7]}]
    pause, _, _ = self._make_pause(('value>8', 't(s)>5'), batches=batches)

    with patch.object(pause_module, 'time',
                      side_effect=[self._t0 + 4, self._t0 + 6,
                                   self._t0 + 7]) as clock:
      for index, batch in enumerate(batches, start=1):
        with self.subTest(batch=batch):
          pause.loop()
          self.assertTrue(pause._pause_event.is_set())
          self.assertEqual(clock.call_count, index)

  def test_callable_criteria_can_retain_previous_state(self) -> None:
    """Checks None before and after both Boolean callable results."""

    criterion = Mock(side_effect=[None, True, None, False, None])
    batches = [{}, {'value': [1]}, {}, {'value': [0]}, {}]
    pause, _, _ = self._make_pause(criterion, batches=batches)

    for index, (batch, expected) in enumerate(zip(
        batches, (False, True, True, False, False)), start=1):
      with self.subTest(batch=batch):
        pause.loop()
        self.assertEqual(pause._pause_event.is_set(), expected)
        self.assertEqual(criterion.call_count, index)

    self.assertEqual(criterion.call_args_list,
                     [call(batch) for batch in batches])

  def test_loop_evaluates_all_callables_once_per_loop(self) -> None:
    """Checks all criteria during transitions and both steady states."""

    first = Mock(side_effect=[False, True, True, False, False])
    second = Mock(side_effect=[False, True, False, False, False])
    batches = [{'value': [value]} for value in range(5)]
    pause, _, recv_calls = self._make_pause((first, second), batches=batches)

    for index, (batch, expected) in enumerate(zip(
        batches, (False, True, True, False, False)), start=1):
      with self.subTest(batch=batch):
        pause.loop()
        self.assertEqual(pause._pause_event.is_set(), expected)
        self.assertEqual(first.call_count, index)
        self.assertEqual(second.call_count, index)

    self.assertEqual(recv_calls, [None] * len(batches))
    self.assertEqual(first.call_args_list, [call(batch) for batch in batches])
    self.assertEqual(second.call_args_list, [call(batch) for batch in batches])

  def test_callable_criteria_receive_data_batches(self) -> None:
    """Checks custom criterion callables against received data."""

    seen = list()

    def criterion(data: dict[str, list[Any]]) -> bool:
      seen.append(dict(data))
      return data.get('value') == [1]

    pause, _, _ = self._make_pause(criterion, batches=[{'value': [1]}])

    pause.loop()

    self.assertTrue(pause._pause_event.is_set())
    self.assertEqual(seen, [{'value': [1]}])
