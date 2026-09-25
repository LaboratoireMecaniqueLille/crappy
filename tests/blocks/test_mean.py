# coding: utf-8

import logging
from multiprocessing import Value
from typing import Any
from unittest.mock import patch
from crappy.blocks.mean import MeanBlock

from ..block import BlockTestBase, TestBlock, link


class TestMeanBlock(BlockTestBase):
  """Unit tests for the MeanBlock-specific behavior."""

  _t0 = 10.0

  def _make_mean(self,
                 batches: list[dict[str, Any]], **kwargs
                 ) -> tuple[MeanBlock,
                            list[dict[str, Any]],
                            list[None],
                            list[tuple[int, str]]]:
    """Creates an instrumented MeanBlock for direct method calls."""

    mean = MeanBlock(**kwargs)
    mean._instance_t0 = Value('d', self._t0)
    mean._last_sent_t = self._t0

    sent = list()
    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)

    def recv_all_data() -> dict[str, Any]:
      recv_calls.append(None)
      return dict(next(batches_iter))

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    mean.recv_all_data = recv_all_data
    mean.send = send
    mean.log = log

    return mean, sent, recv_calls, logs

  def test_delay_is_validated(self) -> None:
    """Checks that the averaging delay is finite, numeric, and positive."""

    for delay in (0, -1, float('nan'), float('inf'), float('-inf')):
      with self.subTest(delay=delay):
        with self.assertRaises(ValueError):
          MeanBlock(delay=delay)

    with self.assertRaises(TypeError):
      MeanBlock(delay='1')

    self.assertEqual(MeanBlock(delay=True)._delay, True)

  def test_time_label_is_validated(self) -> None:
    """Checks that the time label is a non-empty string."""

    with self.assertRaises(ValueError):
      MeanBlock(delay=1, time_label='')
    with self.assertRaises(TypeError):
      MeanBlock(delay=1, time_label=1)

  def test_out_labels_normalization(self) -> None:
    """Checks the supported out_labels forms."""

    self.assertIsNone(MeanBlock(delay=1)._out_labels)
    self.assertEqual(MeanBlock(delay=1, out_labels='abc')._out_labels, ['abc'])
    self.assertEqual(MeanBlock(delay=1, out_labels=('a', 'b'))._out_labels,
                     ['a', 'b'])

    for out_labels in ('', (), [], ('a', ''), ('a', 1)):
      with self.subTest(out_labels=out_labels):
        with self.assertRaises(ValueError):
          MeanBlock(delay=1, out_labels=out_labels)

    with self.assertRaises(TypeError):
      MeanBlock(delay=1, out_labels={'a', 'b'})

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
    """Checks that begin initializes the averaging-window counter."""

    mean = MeanBlock(delay=1)
    mean._instance_t0 = Value('d', self._t0)

    with patch('crappy.blocks.mean.time', return_value=20):
      mean.begin()

    self.assertEqual(mean._last_sent_t, 20)

  def test_loop_averages_numeric_labels(self) -> None:
    """Checks numeric averaging and non-blocking data reception."""

    mean, sent, recv_calls, _ = self._make_mean(
      delay=2,
      batches=[{'t(s)': [1, 5], 'a': [1, 3], 'b': [2, 6]}])

    with patch('crappy.blocks.mean.time', return_value=20):
      mean.loop()

    self.assertEqual(recv_calls, [None])
    self.assertEqual(sent, [{'a': 2.0, 'b': 4.0, 't(s)': 3.0}])
    self.assertEqual(mean._last_sent_t, 20)

  def test_loop_filters_output_labels(self) -> None:
    """Checks that out_labels limits the averaged labels."""

    mean, sent, _, _ = self._make_mean(
      delay=1,
      out_labels='a',
      batches=[{'t(s)': [2, 6], 'a': [1, 3], 'b': [100, 200]}])

    with patch('crappy.blocks.mean.time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 't(s)': 4.0}])

  def test_loop_uses_custom_time_label(self) -> None:
    """Checks that a custom time label is removed then re-emitted."""

    mean, sent, _, _ = self._make_mean(
      delay=1,
      time_label='time',
      batches=[{'time': [10, 14], 'a': [1, 3]}])

    with patch('crappy.blocks.mean.time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 'time': 12.0}])

  def test_loop_generates_time_when_input_has_no_time_label(self) -> None:
    """Checks timestamp generation when incoming data has no time label."""

    mean, sent, _, _ = self._make_mean(delay=1,
                                       batches=[{'a': [1, 3]}])

    with patch('crappy.blocks.mean.time', return_value=14):
      mean.loop()

    self.assertEqual(sent, [{'a': 2.0, 't(s)': 2.0}])
    self.assertEqual(mean._last_sent_t, 14)

  def test_loop_keeps_last_non_numeric_value(self) -> None:
    """Checks fallback behavior for labels that cannot be averaged."""

    mean, sent, _, logs = self._make_mean(
      delay=1,
      batches=[{'t(s)': [0, 2], 'label': ['first', 'last']}])

    with patch('crappy.blocks.mean.time', return_value=30):
      mean.loop()

    self.assertEqual(sent, [{'label': 'last', 't(s)': 1.0}])
    warning_logs = [msg for level, msg in logs if level == logging.WARNING]
    self.assertEqual(len(warning_logs), 1)
    self.assertIn("Cannot perform averaging on label label", warning_logs[0])

  def test_loop_does_not_send_without_matching_data(self) -> None:
    """Checks empty and filtered windows are discarded and advance time."""

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

        with patch('crappy.blocks.mean.time', return_value=30):
          mean.loop()

        self.assertEqual(recv_calls, [None])
        self.assertEqual(sent, [])
        self.assertEqual(mean._last_sent_t, 30)
        self.assertEqual(dict(mean._data_buf), {})

  def test_loop_buffers_data_until_delay_is_reached(self) -> None:
    """Checks that non-blocking reads accumulate complete averaging windows."""

    mean, sent, recv_calls, _ = self._make_mean(
      delay=2,
      batches=[
        {'t(s)': [1, 2], 'a': [2, 4]},
        {'t(s)': [3, 4], 'a': [6, 8]},
      ])

    with patch('crappy.blocks.mean.time', side_effect=(11, 12, 12)):
      mean.loop()
      self.assertEqual(sent, [])
      self.assertEqual(dict(mean._data_buf),
                       {'t(s)': [1, 2], 'a': [2, 4]})

      mean.loop()

    self.assertEqual(recv_calls, [None, None])
    self.assertEqual(sent, [{'a': 5.0, 't(s)': 2.5}])
    self.assertEqual(mean._last_sent_t, 12)
    self.assertEqual(dict(mean._data_buf), {})

  def test_loop_recovers_after_an_empty_window(self) -> None:
    """Checks that an empty interval does not prevent later averaging."""

    mean, sent, _, _ = self._make_mean(
      delay=2,
      batches=[{}, {'t(s)': [3], 'a': [5]},
               {'t(s)': [4], 'a': [7]}])

    with patch('crappy.blocks.mean.time', side_effect=(12, 12, 13, 14, 14)):
      mean.loop()
      mean.loop()
      mean.loop()

    self.assertEqual(sent, [{'a': 6.0, 't(s)': 3.5}])
    self.assertEqual(mean._last_sent_t, 14)
