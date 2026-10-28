# coding: utf-8

from multiprocessing import Value
from numbers import Real
from typing import Any
from unittest.mock import patch
from crappy.blocks.pid import PID
import crappy.blocks.pid as pid_module

from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock, link


class TestPID(BlockTestBase):
  """Unit tests for the PID Block-specific behavior."""

  _t0 = 10.0

  def _make_pid(self,
                batches: list[dict[str, Any]],
                **kwargs) -> tuple[PID,
                                   list[list[Any]],
                                   list[bool],
                                   list[tuple[int, str]]]:
    """Creates an instrumented PID for direct method calls."""

    pid = PID(**kwargs)
    pid._instance_t0 = Value('d', self._t0)

    sent = list()
    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      recv_calls.append(fill_missing)
      return dict(next(batches_iter))

    def send(data: list[Any]) -> None:
      sent.append(list(data))

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    pid.recv_last_data = recv_last_data
    pid.send = send
    pid.log = log

    return pid, sent, recv_calls, logs

  @staticmethod
  def _set_t0(pid: PID) -> PID:
    """Sets a deterministic start time on a PID."""

    pid._instance_t0 = Value('d', TestPID._t0)
    return pid

  def _loop(self, pid: PID, now: float = 15.0) -> None:
    """Runs one loop with a deterministic wall-clock timestamp."""

    with patch.object(pid_module, 'time', return_value=now):
      pid.loop()

  def assert_values_almost_equal(self,
                                 values: list[Any],
                                 expected: list[Any]) -> None:
    """Compares output values while allowing tiny float errors."""

    self.assertEqual(len(values), len(expected))

    for value, expected_value in zip(values, expected):
      if isinstance(expected_value, Real):
        self.assertAlmostEqual(value, expected_value)
      else:
        self.assertEqual(value, expected_value)

  def assert_dict_almost_equal(self,
                               values: dict[str, Any],
                               expected: dict[str, Any]) -> None:
    """Compares output dictionaries while allowing tiny float errors."""

    self.assertEqual(set(values), set(expected))

    for label, expected_value in expected.items():
      if isinstance(expected_value, Real):
        self.assertAlmostEqual(values[label], expected_value)
      else:
        self.assertEqual(values[label], expected_value)

  def test_output_labels(self) -> None:
    """Checks the default and custom output labels."""

    self.assertEqual(PID(kp=1).labels, ['t(s)', 'pid'])
    self.assertEqual(PID(kp=1, labels=('time', 'drive')).labels,
                     ['time', 'drive'])
    self.assertEqual(PID(kp=1, send_terms=True).labels,
                     ['t(s)', 'pid', 'p_term', 'i_term', 'd_term'])
    self.assertEqual(
      PID(kp=1, labels=('time', 'drive'), send_terms=True).labels,
      ['time', 'drive', 'p_term', 'i_term', 'd_term'])

  def test_initial_gain_signs_follow_reverse_argument(self):
    """All three gains affect emitted terms with the requested direction."""

    for reverse, sign, raw_sign in ((False, 1, -1), (False, 1, 1),
                                    (True, -1, -1), (True, -1, 1)):
      with self.subTest(reverse=reverse, raw_sign=raw_sign):
        pid, sent, _, _ = self._make_pid(
          kp=raw_sign, ki=2 * raw_sign, kd=3 * raw_sign,
          reverse=reverse, send_terms=True,
          batches=[{'cmd': 10, 'V': 7, 't(s)': 1}, {'V': 6, 't(s)': 3}])
        self._loop(pid)
        self._loop(pid, now=16)
        self.assert_values_almost_equal(sent[0], [5, 3 * sign, 3 * sign, 0, 0])
        self.assert_values_almost_equal(sent[1],
                                       [6, 21.5 * sign, 4 * sign, 16 * sign, 1.5 * sign])

  def test_prepare_requires_input_and_output_links(self) -> None:
    """Checks that prepare fails early when the Block is not linked enough."""

    pid = PID(kp=1)

    with self.assertRaises(IOError):
      pid.prepare()

    source = TestBlock()
    pid = PID(kp=1)
    link(source, pid)

    with self.assertRaises(IOError):
      pid.prepare()

  def test_prepare_accepts_input_and_output_links(self) -> None:
    """Checks that prepare accepts one incoming and one outgoing Link."""

    source = TestBlock()
    pid = PID(kp=1)
    sink = TestBlock()

    link(source, pid)
    link(pid, sink)

    pid.prepare()

  def test_loop_requests_latest_data_without_filling_missing_labels(
      self) -> None:
    """Checks that PID only uses labels received during the current loop."""

    pid, sent, recv_calls, _ = self._make_pid(
      kp=1,
      batches=[{'cmd': 10, 'V': 7, 't(s)': 1}])

    self._loop(pid)

    self.assertEqual(recv_calls, [False])
    self.assert_values_almost_equal(sent[0], [5, 3])

  def test_loop_returns_without_complete_input(self):
    """Incomplete samples cannot initialize history or emit a control output."""

    for data, expected in (({}, 0), ({'cmd': 12}, 4), ({'V': 7}, 0), ({'t(s)': 1}, 0)):
      with self.subTest(data=data):
        pid, sent, _, _ = self._make_pid(
          kp=1, ki=2, kd=3, send_terms=True,
          batches=[data, {'V': 8, 't(s)': 2}])
        self._loop(pid)
        self.assertEqual(sent, [])
        self._loop(pid)
        self.assert_values_almost_equal(sent[0], [5, expected, expected, 0, 0])

  def test_setpoint_update_without_input_is_used_later(self) -> None:
    """Checks that setpoint-only messages update the stored target."""

    pid, sent, _, _ = self._make_pid(
      kp=2,
      batches=[
        {'cmd': 12},
        {'V': 7, 't(s)': 1},
      ])

    self._loop(pid)
    self._loop(pid)

    self.assert_values_almost_equal(sent[0], [5, 10])

  def test_gain_update_without_input_is_used_later(self) -> None:
    """Checks that gain-only messages update the stored gains."""

    pid, sent, _, _ = self._make_pid(
      kp=1,
      batches=[
        {'kp': 2},
        {'cmd': 10, 'V': 7, 't(s)': 1},
      ])

    self._loop(pid)
    self._loop(pid)

    self.assert_values_almost_equal(sent[0], [5, 6])

  def test_first_input_sets_default_setpoint_when_missing(self):
    pid, sent, _, _ = self._make_pid(
      kp=2, batches=[{'V': 7, 't(s)': 1}, {'V': 8, 't(s)': 2}])
    self._loop(pid)
    self._loop(pid, now=16)
    self.assert_values_almost_equal(sent[0], [5, 0])
    self.assert_values_almost_equal(sent[1], [6, -2])

  def test_first_input_does_not_integrate_or_derivate(self) -> None:
    """Checks that the first timestamp initializes the PID history."""

    pid, sent, _, _ = self._make_pid(
      kp=2,
      ki=3,
      kd=4,
      send_terms=True,
      batches=[{'cmd': 10, 'V': 7, 't(s)': 100}, {'V': 6, 't(s)': 101}])

    self._loop(pid)

    self.assert_values_almost_equal(sent[0], [5, 6, 6, 0, 0])
    self._loop(pid, now=16)
    self.assert_values_almost_equal(sent[1], [6, 24, 8, 12, 4])

  def test_next_inputs_calculate_all_pid_terms(self) -> None:
    """Checks P, I and D terms after the first history point exists."""

    pid, sent, _, _ = self._make_pid(
      kp=2,
      ki=3,
      kd=4,
      send_terms=True,
      batches=[
        {'cmd': 10, 'V': 7, 't(s)': 1},
        {'V': 6, 't(s)': 3},
      ])

    self._loop(pid)
    self._loop(pid, now=16)

    self.assert_values_almost_equal(sent[0], [5, 6, 6, 0, 0])
    self.assert_values_almost_equal(sent[1], [6, 34, 8, 24, 2])

  def test_output_is_clamped(self) -> None:
    """Checks lower and upper output clamping."""

    cases = (
      ({'cmd': 10, 'V': 7, 't(s)': 1}, 4),
      ({'cmd': 0, 'V': 1, 't(s)': 1}, -5),
    )

    for data, expected in cases:
      with self.subTest(data=data):
        pid, sent, _, _ = self._make_pid(
          kp=10,
          out_min=4,
          out_max=-5,
          batches=[data])

        self._loop(pid)

        self.assert_values_almost_equal(sent[0], [5, expected])

  def test_integral_term_is_clamped(self) -> None:
    """Checks lower and upper integral-term clamping."""

    pid, sent, _, _ = self._make_pid(
      kp=0,
      ki=2,
      kd=0,
      i_limit=(5, -5),
      send_terms=True,
      batches=[
        {'cmd': 10, 'V': 0, 't(s)': 0},
        {'V': 0, 't(s)': 1},
        {'V': 20, 't(s)': 2},
      ])

    self._loop(pid)
    self._loop(pid, now=16)
    self._loop(pid, now=17)

    self.assert_values_almost_equal(sent[0], [5, 0, 0, 0, 0])
    self.assert_values_almost_equal(sent[1], [6, 5, 0, 5, 0])
    self.assert_values_almost_equal(sent[2], [7, -5, 0, -5, 0])

  def test_derivative_term_is_zero_when_delta_t_is_zero(self) -> None:
    """Checks that repeated input timestamps do not divide by zero."""

    pid, sent, _, _ = self._make_pid(
      kp=0,
      ki=0,
      kd=10,
      send_terms=True,
      batches=[
        {'cmd': 10, 'V': 0, 't(s)': 1},
        {'V': 5, 't(s)': 1},
      ])

    self._loop(pid)
    self._loop(pid, now=16)

    self.assert_values_almost_equal(sent[1], [6, 0, 0, 0, 0])

  def test_runtime_gain_updates_follow_normal_sign_rule(self):
    pid, sent, _, _ = self._make_pid(
      kp=1, ki=1, kd=1, reverse=False, send_terms=True,
      batches=[{'kp': -2, 'ki': -3, 'kd': -4, 'cmd': 10, 'V': 7, 't(s)': 1},
               {'V': 6, 't(s)': 3}])
    self._loop(pid)
    self._loop(pid, now=16)
    self.assert_values_almost_equal(sent[0], [5, 6, 6, 0, 0])
    self.assert_values_almost_equal(sent[1], [6, 34, 8, 24, 2])

  def test_runtime_gain_updates_follow_reverse_sign_rule(self):
    pid, sent, _, _ = self._make_pid(
      kp=1, ki=1, kd=1, reverse=True, send_terms=True,
      batches=[{'kp': 2, 'ki': 3, 'kd': 4, 'cmd': 10, 'V': 7, 't(s)': 1},
               {'V': 6, 't(s)': 3}])
    self._loop(pid)
    self._loop(pid, now=16)
    self.assert_values_almost_equal(sent[0], [5, -6, -6, 0, 0])
    self.assert_values_almost_equal(sent[1], [6, -34, -8, -24, -2])

  def test_custom_input_labels(self) -> None:
    """Checks custom labels for setpoint, input and input time."""

    pid, sent, _, _ = self._make_pid(
      kp=2,
      setpoint_label='target',
      input_label='actual',
      time_label='time',
      batches=[{'target': 5, 'actual': 2, 'time': 4}])

    self._loop(pid)

    self.assert_values_almost_equal(sent[0], [5, 6])

  def test_custom_gain_labels(self) -> None:
    """Checks that custom gain labels replace the defaults."""

    pid, sent, _, _ = self._make_pid(
      kp=1,
      ki=1,
      kd=1,
      kp_label='p_gain',
      ki_label='i_gain',
      kd_label='d_gain',
      send_terms=True,
      batches=[{'kp': 100, 'ki': 100, 'kd': 100,
                'p_gain': 2, 'i_gain': 3, 'd_gain': 4,
                'cmd': 10, 'V': 8, 't(s)': 1}, {'V': 7, 't(s)': 3}])

    self._loop(pid)

    self.assert_values_almost_equal(sent[0], [5, 4, 4, 0, 0])
    self._loop(pid, now=16)
    self.assert_values_almost_equal(sent[1], [6, 26, 6, 18, 2])

  def test_real_send_uses_custom_labels_and_pid_terms(self) -> None:
    """Checks iterable-to-dict conversion with custom labels and terms."""

    pid = self._set_t0(PID(kp=2,
                           labels=('time', 'drive'),
                           send_terms=True))
    sink = TestBlock()
    link(pid, sink)

    def recv_last_data(fill_missing: bool = True) -> dict[str, Any]:
      return {'cmd': 10, 'V': 7, 't(s)': 1}

    pid.recv_last_data = recv_last_data

    self._loop(pid)

    self.assert_dict_almost_equal(sink.recv_data(), {
      'time': 5,
      'drive': 6,
      'p_term': 6,
      'i_term': 0,
      'd_term': 0,
    })
