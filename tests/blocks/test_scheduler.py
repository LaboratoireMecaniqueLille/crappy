# coding: utf-8

from unittest.mock import patch
from typing import Any

import crappy.blocks.scheduler as scheduler_module
from crappy._global import SchedulerStop
from crappy.blocks.scheduler import Scheduler
from crappy.blocks.schedulers import State
from crappy.blocks.schedulers.conditions import Delay
from crappy.blocks.schedulers.outputs import Constant

from ..block import BlockTestBase


class ProbeOutput:
  """Records when a State output is evaluated or reset."""

  def __init__(self, value: dict[str, Any] | None) -> None:
    self.value = value
    self.calls = []
    self.resets = 0

  def __call__(self, dt, data):
    self.calls.append((dt, data.copy()))
    return self.value

  def reset(self):
    self.resets += 1


class ProbeCondition:
  """Records when a State condition is evaluated or reset."""

  def __init__(self, result: bool) -> None:
    self.result = result
    self.calls = []
    self.resets = 0

  def __call__(self, dt, data):
    self.calls.append((dt, data.copy()))
    return self.result

  def reset(self):
    self.resets += 1


class TestScheduler(BlockTestBase):
  """Unit tests for Scheduler initialization and state-machine behavior."""

  def _make_scheduler(self,
                      states: list[State] | None = None,
                      output_labels: list[str] | tuple[str, ...] = ('x',),
                      **kwargs) -> Scheduler:
    """Constructs a quiet Scheduler with one default State."""

    if states is None:
      states = [State('A', [Constant('x', 1)], [])]
    kwargs.setdefault('freq', None)
    kwargs.setdefault('debug', None)
    return Scheduler(states, output_labels, **kwargs)

  @staticmethod
  def _capture_send(scheduler: Scheduler) -> list[dict[str, Any]]:
    """Replaces transport with an in-memory record of sent payloads."""

    sent = []
    scheduler.send = lambda data: sent.append(data.copy())
    return sent

  @staticmethod
  def _set_batches(scheduler: Scheduler,
                   batches: list[dict[str, list[Any]]]) -> None:
    """Makes input reception deterministic across loop calls."""

    batches_iter = iter(batches)
    scheduler.recv_all_data = lambda: next(batches_iter)

  def test_constructor_builds_internal_states_and_options(self) -> None:
    """Start and End are registered beside the supplied States."""

    scheduler = self._make_scheduler(input_labels=['sensor'],
                                     init_values={'x': 2},
                                     last_output={'x': 0},
                                     spam=True, safe_start=True,
                                     state_id_label='phase',
                                     end_delay=None, display_freq=True)
    self.assertEqual(set(scheduler._states), {'Start', 'A', 'End'})
    self.assertEqual(scheduler._current_state.id, 'Start')
    self.assertEqual(scheduler._input_labels, ('sensor',))
    self.assertEqual(scheduler._init_values, {'x': 2})
    self.assertEqual(scheduler._last_output, {'x': 0})
    self.assertTrue(scheduler._spam)
    self.assertTrue(scheduler._safe_start)
    self.assertEqual(scheduler._state_id_label, 'phase')
    self.assertIsNone(scheduler._end_delay)
    self.assertIsNone(scheduler.freq)
    self.assertTrue(scheduler.display_freq)

  def test_constructor_rejects_invalid_state_graphs(self) -> None:
    """State IDs and transition targets must form a valid graph."""

    cases = (([], ValueError),
             ([1], TypeError),
             ([State('A', [], []), State('A', [], [])], ValueError),
             ([State('Start', [], [])], ValueError),
             ([State('End', [], [])], ValueError),
             ([State('A', [], [(lambda _dt, _data: True, 'missing')])],
              IOError))
    for states, error in cases:
      with self.subTest(states=states), self.assertRaises(error):
        self._make_scheduler(states)

  def test_constructor_rejects_invalid_labels_and_options(self) -> None:
    """Bad output, input, cache, policy and delay options fail early."""

    cases = (({'output_labels': []}, ValueError),
             ({'output_labels': ['']}, ValueError),
             ({'output_labels': [1]}, TypeError),
             ({'input_labels': ['']}, ValueError),
             ({'input_labels': [1]}, TypeError),
             ({'init_values': {'unknown': 1}}, ValueError),
             ({'init_values': {'': 1}}, ValueError),
             ({'init_values': 1}, TypeError),
             ({'last_output': {'unknown': 1}}, ValueError),
             ({'last_output': 1}, TypeError),
             ({'spam': 1}, TypeError),
             ({'safe_start': 1}, TypeError),
             ({'state_id_label': 'x'}, ValueError),
             ({'state_id_label': ''}, ValueError),
             ({'state_id_label': 1}, TypeError),
             ({'end_delay': -1}, ValueError),
             ({'end_delay': '1'}, TypeError))
    for kwargs, error in cases:
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        self._make_scheduler(**kwargs)

  def test_begin_early_send_and_safe_start(self) -> None:
    """Complete initial values are sent from Start only without safe-start."""

    scheduler = self._make_scheduler(init_values={'x': 3})
    sent = self._capture_send(scheduler)
    scheduler.begin()
    self.assertEqual(sent, [{'x': 3, 'state': 'Start'}])
    self.assertEqual(scheduler._current_state.id, 'A')

    safe = self._make_scheduler(init_values={'x': 3}, safe_start=True,
                                input_labels=['sensor'])
    safe_sent = self._capture_send(safe)
    safe.begin()
    self.assertEqual(safe_sent, [])
    self.assertEqual(safe._current_state.id, 'A')

  def test_loop_filters_inputs_and_reuses_latest_value(self) -> None:
    """Only selected labels are passed to outputs, with cached last values."""

    received = []

    def output(_dt, data):
      received.append(data.copy())
      return {'x': data['a'][-1]}

    scheduler = self._make_scheduler([State('A', [output], [])],
                                     input_labels=['a'])
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{'a': [1, 2], 'ignored': [8]}, {}])
    scheduler.begin()
    scheduler.loop()
    scheduler.loop()

    self.assertEqual(received, [{'a': [1, 2]}, {'a': [2]}])
    self.assertEqual(sent, [{'x': 2, 'state': 'A'}])

  def test_safe_start_waits_for_all_inputs_but_not_conditions(self) -> None:
    """Missing input labels gate outputs, while transitions still run."""

    output = ProbeOutput({'x': 1})
    scheduler = self._make_scheduler([State('A', [output], [])],
                                     input_labels=['a', 'b'],
                                     safe_start=True)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{'a': [1]}, {'b': [2]}])
    scheduler.begin()
    scheduler.loop()
    self.assertEqual(output.calls, [])
    scheduler.loop()
    self.assertEqual(output.calls[0][1], {'a': [1], 'b': [2]})
    self.assertEqual(sent, [{'x': 1, 'state': 'A'}])

    transition = ProbeCondition(True)
    gated_output = ProbeOutput({'x': 2})
    gated = self._make_scheduler(
      [State('A', [gated_output], [(transition, 'End')])],
      input_labels=['missing'], safe_start=True, end_delay=None)
    self._capture_send(gated)
    self._set_batches(gated, [{}])
    gated.begin()
    gated.loop()
    self.assertEqual(gated._current_state.id, 'End')
    self.assertEqual(gated_output.calls, [])

  def test_transition_priority_skips_output_and_resets_target(self) -> None:
    """Only the first true condition wins; target hooks run once on entry."""

    first, second = ProbeCondition(True), ProbeCondition(True)
    old_output, new_output = ProbeOutput({'x': 1}), ProbeOutput({'x': 2})
    new_condition = ProbeCondition(False)
    states = [State('A', [old_output], [(first, 'B'), (second, 'C')]),
              State('B', [new_output], [(new_condition, 'End')]),
              State('C', [Constant('x', 3)], [])]
    scheduler = self._make_scheduler(states)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}])
    scheduler.begin()
    scheduler.loop()
    self.assertEqual(scheduler._current_state.id, 'B')
    self.assertEqual(len(first.calls), 1)
    self.assertEqual(second.calls, [])
    self.assertEqual(old_output.calls, [])
    self.assertEqual(new_output.resets, 1)
    self.assertEqual(new_condition.resets, 1)
    scheduler.loop()
    self.assertEqual(sent, [{'x': 2, 'state': 'B'}])

  def test_output_cache_waits_for_all_labels_across_states(self) -> None:
    """Partial values survive a transition until all labels are known."""

    transition = ProbeCondition(False)
    states = [State('A', [Constant('x', 1)], [(transition, 'B')]),
              State('B', [Constant('y', 2)], [])]
    scheduler = self._make_scheduler(states, output_labels=('x', 'y'))
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}, {}])
    scheduler.begin()
    scheduler.loop()
    self.assertEqual(sent, [])
    transition.result = True
    scheduler.loop()
    self.assertEqual(sent, [])
    scheduler.loop()
    self.assertEqual(sent, [{'x': 1, 'y': 2, 'state': 'B'}])

  def test_revisiting_state_resets_hooks_and_sends_unchanged_value(self) -> None:
    """Feedback transitions revisit States without losing entry semantics."""

    to_b, to_a = ProbeCondition(False), ProbeCondition(False)
    output_a, output_b = ProbeOutput({'x': 1}), ProbeOutput({'x': 1})
    scheduler = self._make_scheduler([
      State('A', [output_a], [(to_b, 'B')]),
      State('B', [output_b], [(to_a, 'A')])])
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}, {}, {}, {}])
    scheduler.begin()
    scheduler.loop()
    to_b.result = True
    scheduler.loop()
    scheduler.loop()
    to_a.result = True
    scheduler.loop()
    to_b.result = False
    scheduler.loop()

    self.assertEqual([item['state'] for item in sent], ['A', 'B', 'A'])
    self.assertEqual([item['x'] for item in sent], [1, 1, 1])
    self.assertEqual((output_a.resets, output_b.resets), (2, 1))
    self.assertEqual((to_b.resets, to_a.resets), (2, 1))

  def test_output_receives_elapsed_time_from_current_state_entry(self) -> None:
    """The supplied dt is relative to the latest transition timestamp."""

    output = ProbeOutput({'x': 1})
    scheduler = self._make_scheduler([State('A', [output], [])])
    self._capture_send(scheduler)
    self._set_batches(scheduler, [{}])
    scheduler.begin()
    scheduler._last_t_sched = 10
    with patch.object(scheduler_module, 'monotonic', return_value=12.5):
      scheduler.loop()
    self.assertEqual(output.calls[0][0], 2.5)

  def test_output_merge_and_return_validation(self) -> None:
    """Later outputs overwrite earlier ones and invalid returns fail."""

    scheduler = self._make_scheduler([
      State('A', [lambda _dt, _data: {'x': 1},
                  lambda _dt, _data: None,
                  lambda _dt, _data: {'x': 2, 'y': 3}], [])])
    scheduler.begin()
    with patch.object(scheduler, 'log') as log:
      self.assertEqual(scheduler._evaluate_output(0, {}), {'x': 2, 'y': 3})
      scheduler._evaluate_output(0, {})
    self.assertEqual(log.call_count, 1)

    for value, error in ((1, TypeError), ({'': 1}, ValueError),
                         ({1: 1}, ValueError)):
      with self.subTest(value=value):
        bad = self._make_scheduler([
          State('A', [lambda _dt, _data, result=value: result], [])])
        bad.begin()
        with self.assertRaises(error):
          bad._evaluate_output(0, {})

  def test_unexpected_output_label_is_ignored_and_warned_once(self) -> None:
    """Unregistered labels cannot leak to downstream Blocks."""

    scheduler = self._make_scheduler([
      State('A', [lambda _dt, _data: {'x': 1, 'extra': 2}], [])])
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}])
    scheduler.begin()
    with patch.object(scheduler, 'log') as log:
      scheduler.loop()
      scheduler.loop()
    self.assertEqual(sent, [{'x': 1, 'state': 'A'}])
    warnings = [call for call in log.call_args_list if call.args[0] == 30]
    self.assertEqual(len(warnings), 1)

  def test_send_policy_suppresses_duplicates_unless_requested(self) -> None:
    """Changes, State entries and spam independently cause a send."""

    scheduler = self._make_scheduler()
    sent = self._capture_send(scheduler)
    scheduler.begin()
    scheduler._send_values({'x': 1})
    scheduler._send_values({'x': 1})
    scheduler._send_on_state_change = True
    scheduler._send_values({'x': 1})
    scheduler._send_values({'x': 2})
    self.assertEqual(sent, [{'x': 1, 'state': 'A'},
                            {'x': 1, 'state': 'A'},
                            {'x': 2, 'state': 'A'}])

    spam = self._make_scheduler(spam=True)
    spam_sent = self._capture_send(spam)
    spam.begin()
    spam._send_values({'x': 1})
    spam._send_values({'x': 1})
    self.assertEqual(len(spam_sent), 2)

  def test_end_state_sends_last_output_before_stopping(self) -> None:
    """The terminal delay does not prevent a configured final command."""

    scheduler = self._make_scheduler([
      State('A', [Constant('x', 1)], [(lambda _dt, _data: True, 'End')])],
      last_output={'x': 0}, end_delay=0)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}, {}])
    scheduler.begin()
    scheduler.loop()
    self.assertEqual(scheduler._current_state.id, 'End')
    self.assertEqual(sent, [])
    with patch.object(scheduler_module, 'monotonic',
                      return_value=scheduler._last_t_sched + 1):
      scheduler.loop()
      self.assertEqual(sent, [{'x': 0, 'state': 'End'}])
      with self.assertRaises(SchedulerStop):
        scheduler.loop()

  def test_end_condition_boundary_and_indefinite_wait(self) -> None:
    """Stopping requires elapsed time strictly beyond end_delay."""

    scheduler = self._make_scheduler(end_delay=2)
    self.assertFalse(scheduler._end_condition(2, {}))
    with self.assertRaises(SchedulerStop):
      scheduler._end_condition(2.01, {})
    idle = self._make_scheduler(end_delay=None)
    self.assertFalse(idle._end_condition(1e9, {}))

  def test_end_without_last_output_sends_retained_values_then_idles(
      self) -> None:
    """An indefinite End State announces its entry only once."""

    transition = ProbeCondition(False)
    scheduler = self._make_scheduler([
      State('A', [Constant('x', 1)], [(transition, 'End')])],
      end_delay=None)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}, {}, {}])
    scheduler.begin()
    scheduler.loop()
    transition.result = True
    scheduler.loop()
    scheduler.loop()
    scheduler.loop()
    self.assertEqual(sent, [{'x': 1, 'state': 'A'},
                            {'x': 1, 'state': 'End'}])
    self.assertTrue(scheduler._end)

  def test_zero_delay_stops_on_first_end_loop_without_final_output(
      self) -> None:
    """Without a final command, no extra End loop is required."""

    scheduler = self._make_scheduler([
      State('A', [Constant('x', 1)], [(lambda _dt, _data: True, 'End')])],
      end_delay=0)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}])
    scheduler.begin()
    scheduler.loop()
    with patch.object(scheduler_module, 'monotonic',
                      return_value=scheduler._last_t_sched + 1):
      with self.assertRaises(SchedulerStop):
        scheduler.loop()
    self.assertEqual(sent, [])

  def test_last_output_does_not_bypass_missing_output_labels(self) -> None:
    """The End State obeys the same complete-output guard as other States."""

    scheduler = self._make_scheduler([
      State('A', [], [(lambda _dt, _data: True, 'End')])],
      output_labels=('x', 'y'), last_output={'x': 0}, end_delay=None)
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}, {}])
    scheduler.begin()
    scheduler.loop()
    scheduler.loop()
    self.assertEqual(sent, [])
    self.assertEqual(scheduler._send_cache, {'x': 0})

  def test_finish_clears_runtime_caches_for_another_run(self) -> None:
    """A second begin starts from Start with fresh caches."""

    scheduler = self._make_scheduler()
    sent = self._capture_send(scheduler)
    self._set_batches(scheduler, [{}])
    scheduler.begin()
    scheduler.loop()
    self.assertEqual(sent, [{'x': 1, 'state': 'A'}])
    scheduler.finish()
    self.assertEqual(scheduler._current_state.id, 'Start')
    self.assertEqual(scheduler._send_cache, {})
    self.assertEqual(scheduler._receive_cache, {})
    self.assertEqual(scheduler._latest_sent, {})
    scheduler.begin()
    self._set_batches(scheduler, [{}])
    scheduler.loop()
    self.assertEqual(len(sent), 2)
    self.assertEqual(sent[-1], {'x': 1, 'state': 'A'})
