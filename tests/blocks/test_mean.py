# coding: utf-8

import logging
from multiprocessing import Value
from typing import Any
from unittest.mock import patch
from crappy.blocks.mean import MeanBlock
import crappy.blocks.mean as mean_module

from ..block import BlockTestBase, TestBlock, link


class TestMeanBlock(BlockTestBase):
  """Unit tests for the MeanBlock-specific behavior."""

  _t0 = 10.0

  def _make_mean(self,
                 batches: list[dict[str, Any]], **kwargs
                 ) -> tuple[MeanBlock,
                            list[dict[str, Any]],
                            list[tuple[float | None, float]],
                            list[tuple[int, str]]]:
    """Creates an instrumented MeanBlock for direct method calls."""

    mean = MeanBlock(**kwargs)
    mean._instance_t0 = Value('d', self._t0)

    sent = list()
    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)

    def recv_all_data(delay: float | None = None,
                      poll_delay: float = 0.1) -> dict[str, Any]:
      recv_calls.append((delay, poll_delay))
      return dict(next(batches_iter))

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    mean.recv_all_data = recv_all_data
    mean.send = send
    mean.log = log

    return mean, sent, recv_calls, logs

  def test_delay_must_be_positive(self) -> None:
    """Checks that invalid averaging delays are rejected."""

    for delay in (0, -1):
      with self.subTest(delay=delay):
        with self.assertRaises(ValueError):
          MeanBlock(delay=delay)

  def test_out_labels_normalization(self) -> None:
    """Checks the supported out_labels forms."""

    self.assertIsNone(MeanBlock(delay=1)._out_labels)
    self.assertEqual(MeanBlock(delay=1, out_labels='abc')._out_labels, ['abc'])
    self.assertEqual(MeanBlock(delay=1, out_labels=('a', 'b'))._out_labels,
                     ['a', 'b'])

  def test_prepare_requires_input_and_output_links(self) -> None:
    """Checks that prepare fails early when the Block is not linked enough."""

    mean = MeanBlock(delay=1)

    with self.assertRaises(IOError):
      mean.prepare()

    source = TestBlock()
    mean = MeanBlock(delay=1)
    link(source, mean)

    with self.assertRaises(IOError):
      mean.prepare()

  def test_prepare_accepts_input_and_output_links(self) -> None:
    """Checks that prepare accepts one incoming and one outgoing Link."""

    source = TestBlock()
    mean = MeanBlock(delay=1)
    sink = TestBlock()

    link(source, mean)
    link(mean, sink)

    mean.prepare()

  def test_begin_initializes_last_sent_time(self) -> None:
    """Checks that begin initializes the timestamp counter from t0."""

    mean = MeanBlock(delay=1)
    mean._instance_t0 = Value('d', self._t0)

    mean.begin()

    self.assertEqual(mean._last_sent_t, self._t0)

  def test_loop_averages_numeric_labels(self) -> None:
    """Checks numeric averaging and helper delay arguments."""

    mean, sent, recv_calls, _ = self._make_mean(
      delay=2,
      batches=[{'t(s)': [11, 15], 'a': [1, 3], 'b': [2, 6]}])

    with patch.object(mean_module, 'time', return_value=20):
      mean.loop()

    self.assertEqual(recv_calls, [(2, 0.2)])
    self.assertEqual(sent, [{'a': 2.0, 'b': 4.0, 't(s)': 3.0}])
    self.assertEqual(mean._last_sent_t, 20)

  def test_loop_filters_output_labels(self) -> None:
    """Checks that out_labels limits the averaged labels."""

    mean, sent, _, _ = self._make_mean(
      delay=1,
      out_labels='a',
      batches=[{'t(s)': [12, 16], 'a': [1, 3], 'b': [100, 200]}])

    with patch.object(mean_module, 'time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 't(s)': 4.0}])

  def test_loop_uses_custom_time_label(self) -> None:
    """Checks that a custom time label is removed then re-emitted."""

    mean, sent, _, _ = self._make_mean(
      delay=1,
      time_label='time',
      batches=[{'time': [20, 24], 'a': [1, 3]}])

    with patch.object(mean_module, 'time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 'time': 12.0}])

  def test_loop_generates_time_when_input_has_no_time_label(self) -> None:
    """Checks timestamp generation when incoming data has no time label."""

    mean, sent, _, _ = self._make_mean(delay=1,
                                       batches=[{'a': [1, 3]}])
    mean.begin()

    with patch.object(mean_module, 'time', side_effect=(14, 14)):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 't(s)': 2.0}])
    self.assertEqual(mean._last_sent_t, 14)

  def test_loop_keeps_last_non_numeric_value(self) -> None:
    """Checks fallback behavior for labels that cannot be averaged."""

    mean, sent, _, logs = self._make_mean(
      delay=1,
      batches=[{'t(s)': [10, 12], 'label': ['first', 'last']}])

    with patch.object(mean_module, 'time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'label': 'last', 't(s)': 1.0}])
    warning_logs = [msg for level, msg in logs if level == logging.WARNING]
    self.assertEqual(len(warning_logs), 1)
    self.assertIn("Cannot perform averaging on label label", warning_logs[0])

  def test_loop_does_not_send_without_matching_data(self) -> None:
    """Checks that empty or filtered data does not produce an output."""

    cases = (
      ({}, None),
      ({'t(s)': [10, 12], 'b': [1, 3]}, 'a'),
    )

    for data, out_labels in cases:
      with self.subTest(data=data, out_labels=out_labels):
        kwargs = {'delay': 1, 'batches': [data]}
        if out_labels is not None:
          kwargs['out_labels'] = out_labels
        mean, sent, recv_calls, _ = self._make_mean(**kwargs)
        mean._last_sent_t = 123

        mean.loop()

        self.assertEqual(recv_calls, [(1, 0.1)])
        self.assertEqual(sent, [])
        self.assertEqual(mean._last_sent_t, 123)
