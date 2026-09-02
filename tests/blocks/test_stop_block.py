# coding: utf-8

import logging
from multiprocessing import Event, Value
from typing import Any
from unittest.mock import patch

from crappy.blocks.stop_block import StopBlock
import crappy.blocks.stop_block as stop_block_module

from ..block import BlockTestBase, TestBlock, link


class TestStopBlock(BlockTestBase):
  """Unit tests for the StopBlock-specific behavior."""

  _t0 = 10.0

  def _make_stop_block(self,
                       criteria,
                       batches: list[dict[str, list[Any]]] | None = None,
                       *,
                       linked: bool = True
                       ) -> tuple[StopBlock, list[tuple[int, str]], list[None]]:
    """Creates a StopBlock with deterministic data and synchronization."""

    stop = StopBlock(criteria)

    if linked:
      source = TestBlock()
      link(source, stop)

    stop._stop_event = Event()
    stop._instance_t0 = Value('d', self._t0)

    logs = list()
    recv_calls = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    stop.log = log

    if batches is not None:
      batches_iter = iter(batches)

      def recv_all_data() -> dict[str, list[Any]]:
        recv_calls.append(None)
        return dict(next(batches_iter))

      stop.recv_all_data = recv_all_data

    stop.prepare()
    return stop, logs, recv_calls

  def test_init_sets_block_options_and_normalizes_criteria(self) -> None:
    """Checks StopBlock-specific initialization."""

    def criterion(_):
      return False

    stop = StopBlock('value>8', freq=None, display_freq=True, debug=True)

    self.assertFalse(stop.pausable)
    self.assertIsNone(stop.freq)
    self.assertTrue(stop.display_freq)
    self.assertTrue(stop.debug)
    self.assertEqual(stop._raw_crit, ('value>8',))
    self.assertIsNone(stop._criteria)

    self.assertEqual(StopBlock(criterion)._raw_crit, (criterion,))
    self.assertEqual(StopBlock(('value>8', criterion))._raw_crit,
                     ('value>8', criterion))

  def test_prepare_accepts_time_only_block_without_input_link(self) -> None:
    """Checks that a time-only StopBlock can be prepared without Links."""

    stop, _, _ = self._make_stop_block('t(s)>5',
                                       batches=[{}],
                                       linked=False)

    self.assertEqual(len(stop._criteria), 1)
    self.assertTrue(callable(stop._criteria[0]))

  def test_prepare_converts_criteria_to_callables(self) -> None:
    """Checks that prepare parses all raw criteria."""

    def criterion(_):
      return True

    stop, _, _ = self._make_stop_block(('value>8', criterion))

    self.assertEqual(len(stop._criteria), 2)
    self.assertTrue(all(callable(crit) for crit in stop._criteria))
    self.assertIs(stop._criteria[1], criterion)

  def test_prepare_rejects_wrong_string_syntax(self) -> None:
    """Checks that invalid criterion strings fail during prepare."""

    stop = StopBlock('value=8')

    with self.assertRaises(ValueError):
      stop.prepare()

  def test_string_criteria_ignore_spaces_around_operands(self) -> None:
    """Checks documented criterion parsing with optional whitespace."""

    stop, _, _ = self._make_stop_block((' value < 3 ', ' value > 7 '))
    less_than, greater_than = stop._criteria

    self.assertTrue(less_than({'value': [4, 2]}))
    self.assertFalse(less_than({'value': [3, 4]}))
    self.assertFalse(less_than({' value': [2]}))

    self.assertTrue(greater_than({'value': [7, 8]}))
    self.assertFalse(greater_than({'value': [6, 7]}))
    self.assertFalse(greater_than({' value': [8]}))

  def test_loop_stays_quiet_without_matching_data(self) -> None:
    """Checks that non-matching data does not set the stop Event."""

    stop, logs, recv_calls = self._make_stop_block('value>8',
                                                   batches=[{}])

    stop.loop()

    self.assertEqual(recv_calls, [None])
    self.assertFalse(stop._stop_event.is_set())
    self.assertIn((logging.DEBUG,
                   'No stop criterion reached during this loop'), logs)

  def test_loop_stops_when_any_criterion_is_met(self) -> None:
    """Checks that matching data sets the shared stop Event."""

    stop, logs, _ = self._make_stop_block(('value>8', 'other<0'),
                                          batches=[{'value': [1, 9]}])

    stop.loop()

    self.assertTrue(stop._stop_event.is_set())
    self.assertIn((logging.WARNING,
                   'Stop criterion reached, stopping all the Blocks !'), logs)
    self.assertIn((logging.WARNING,
                   'stop method called, setting the stop event !'), logs)
    self.assertNotIn((logging.DEBUG,
                      'No stop criterion reached during this loop'), logs)

  def test_loop_keeps_stop_event_clear_when_criteria_are_false(self) -> None:
    """Checks that false criteria do not stop the test."""

    stop, _, _ = self._make_stop_block('value>8',
                                       batches=[{'value': [8]}])

    stop.loop()

    self.assertFalse(stop._stop_event.is_set())

  def test_time_criterion_can_stop_without_received_data(self) -> None:
    """Checks time-based criteria even when no upstream payload is available."""

    stop, _, _ = self._make_stop_block('t(s)>5',
                                       batches=[{}],
                                       linked=False)

    with patch.object(stop_block_module, 'time', return_value=self._t0 + 6):
      stop.loop()

    self.assertTrue(stop._stop_event.is_set())

  def test_time_criterion_uses_elapsed_time_threshold(self) -> None:
    """Checks the false case for time-based criteria."""

    stop, _, _ = self._make_stop_block('t(s)>5',
                                       batches=[{}],
                                       linked=False)

    with patch.object(stop_block_module, 'time', return_value=self._t0 + 5):
      stop.loop()

    self.assertFalse(stop._stop_event.is_set())

  def test_callable_criteria_receive_data_batches(self) -> None:
    """Checks custom criterion callables against received data."""

    seen = list()

    def criterion(data: dict[str, list[Any]]) -> bool:
      seen.append(dict(data))
      return data.get('value') == [1]

    stop, _, _ = self._make_stop_block(criterion, batches=[{'value': [1]}])

    stop.loop()

    self.assertTrue(stop._stop_event.is_set())
    self.assertEqual(seen, [{'value': [1]}])
