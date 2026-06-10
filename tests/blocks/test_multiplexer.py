# coding: utf-8

import logging
from numbers import Real
from typing import Any
from crappy.blocks.multiplexer import Multiplexer

from ..block import BlockTestBase, TestBlock, link


class TestMultiplexer(BlockTestBase):
  """Unit tests for the Multiplexer Block-specific behavior."""

  def _make_multiplexer(self,
                        batches: list[list[dict[str, Any]]],
                        **kwargs) -> tuple[Multiplexer,
                                           list[dict[str, Any]],
                                           list[None],
                                           list[tuple[int, str]]]:
    """Creates an instrumented Multiplexer for direct method calls."""

    mux = Multiplexer(**kwargs)
    sent = list()
    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)

    def recv_all_data_raw() -> list[dict[str, Any]]:
      recv_calls.append(None)
      return [{key: list(value) for key, value in link_data.items()}
              for link_data in next(batches_iter)]

    def send(data: dict[str, Any]) -> None:
      sent.append(dict(data))

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    mux.recv_all_data_raw = recv_all_data_raw
    mux.send = send
    mux.log = log

    return mux, sent, recv_calls, logs

  def assert_sent_almost_equal(self,
                               sent: list[dict[str, Any]],
                               expected: list[dict[str, Any]]) -> None:
    """Compares sent dictionaries while allowing tiny interpolation errors."""

    self.assertEqual(len(sent), len(expected))

    for got, want in zip(sent, expected):
      self.assertEqual(set(got), set(want))
      for label, value in want.items():
        if isinstance(value, Real):
          self.assertAlmostEqual(got[label], value)
        else:
          self.assertEqual(got[label], value)

  def test_interp_freq_must_be_positive(self) -> None:
    """Checks that invalid interpolation frequencies are rejected."""

    for interp_freq in (0, -1):
      with self.subTest(interp_freq=interp_freq):
        with self.assertRaises(ValueError):
          Multiplexer(interp_freq=interp_freq)

  def test_out_labels_normalization(self) -> None:
    """Checks the supported out_labels forms."""

    self.assertIsNone(Multiplexer()._out_labels)
    self.assertEqual(Multiplexer(out_labels='abc')._out_labels, ['abc'])
    self.assertEqual(Multiplexer(out_labels=('a', 'b'))._out_labels,
                     ['a', 'b'])

  def test_prepare_requires_input_and_output_links(self) -> None:
    """Checks that prepare fails early when the Block is not linked enough."""

    mux = Multiplexer()

    with self.assertRaises(IOError):
      mux.prepare()

    source = TestBlock()
    mux = Multiplexer()
    link(source, mux)

    with self.assertRaises(IOError):
      mux.prepare()

  def test_prepare_accepts_input_and_output_links(self) -> None:
    """Checks that prepare accepts one incoming and one outgoing Link."""

    source = TestBlock()
    mux = Multiplexer()
    sink = TestBlock()

    link(source, mux)
    link(mux, sink)

    mux.prepare()

  def test_loop_ignores_data_without_time_label(self) -> None:
    """Checks that untimestamped data is ignored."""

    mux, sent, recv_calls, logs = self._make_multiplexer(
      batches=[[{'a': [1, 2]}]],
      interp_freq=1)

    mux.loop()

    self.assertEqual(recv_calls, [None])
    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No data in the buffer to process"), logs)

  def test_loop_does_not_send_time_only_data(self) -> None:
    """Checks that timestamps without labels do not produce an output."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[[{'t(s)': [0, 1]}]],
      interp_freq=1)

    mux.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No data in the buffer to process"), logs)

  def test_loop_waits_for_requested_labels(self) -> None:
    """Checks that out_labels gates output until every label is buffered."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[
        [{'t(s)': [0, 1], 'a': [0, 10]}],
        [{'t(s)': [0, 1], 'b': [100, 200]}],
      ],
      out_labels=('a', 'b'),
      interp_freq=1)

    mux.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Not all the requested labels received yet"),
                  logs)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0, 'b': 100},
      {'t(s)': 1, 'a': 10, 'b': 200},
    ])

  def test_loop_waits_for_two_points_per_label(self) -> None:
    """Checks that a single buffered point cannot be interpolated."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[[{'t(s)': [0], 'a': [10]}]],
      interp_freq=1)

    mux.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Not at least 2 values for each label in "
                                  "buffer"), logs)

  def test_loop_waits_for_span_at_least_one_period(self) -> None:
    """Checks that too-close timestamps are buffered but not sent yet."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[[{'t(s)': [0, 0.5], 'a': [0, 1]}]],
      interp_freq=1)

    mux.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "At least one label has values too close "
                                  "together compared to interpolation "
                                  "frequency"), logs)

  def test_loop_waits_for_overlapping_ranges(self) -> None:
    """Checks that non-overlapping label ranges do not produce an output."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[[
        {'t(s)': [0, 1], 'a': [0, 10]},
        {'t(s)': [2, 3], 'b': [20, 30]},
      ]],
      interp_freq=1)

    mux.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Ranges not matching for interpolation"),
                  logs)

  def test_loop_interpolates_single_input_exact_grid_boundaries(self) -> None:
    """Checks that first and last exact grid points are emitted."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [0, 1], 'a': [0, 10], 'b': [100, 200]}]],
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0, 'b': 100},
      {'t(s)': 1, 'a': 10, 'b': 200},
    ])

  def test_loop_interpolates_unsorted_timestamps(self) -> None:
    """Checks that incoming samples are sorted before interpolation."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [2, 0, 1], 'a': [20, 0, 10]}]],
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
      {'t(s)': 2, 'a': 20},
    ])

  def test_loop_filters_output_labels(self) -> None:
    """Checks that out_labels limits the multiplexed labels."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [0, 1], 'a': [0, 10], 'b': [100, 200]}]],
      out_labels='a',
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
    ])

  def test_loop_uses_custom_time_label(self) -> None:
    """Checks that custom time labels are used for input and output."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'time': [0, 1], 'a': [0, 2]}]],
      time_label='time',
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'time': 0, 'a': 0},
      {'time': 1, 'a': 2},
    ])

  def test_loop_handles_negative_and_zero_grid_points(self) -> None:
    """Checks interpolation ranges crossing zero."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [-1, 0], 'a': [-10, 0]}]],
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': -1, 'a': -10},
      {'t(s)': 0, 'a': 0},
    ])

  def test_loop_combines_fast_and_slow_inputs(self) -> None:
    """Checks interpolation with upstream Blocks at different rates."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[
        {'t(s)': [0, 0.25, 0.5, 0.75, 1], 'fast': [0, 2.5, 5, 7.5, 10]},
        {'t(s)': [0, 1], 'slow': [100, 200]},
      ]],
      interp_freq=2)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'fast': 0, 'slow': 100},
      {'t(s)': 0.5, 'fast': 5, 'slow': 150},
      {'t(s)': 1, 'fast': 10, 'slow': 200},
    ])

  def test_loop_downsamples_input_much_faster_than_interp_freq(self) -> None:
    """Checks behavior when incoming data is denser than the output grid."""

    timestamps = [i / 10 for i in range(11)]
    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': timestamps, 'a': [10 * t for t in timestamps]}]],
      interp_freq=2)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 0.5, 'a': 5},
      {'t(s)': 1, 'a': 10},
    ])

  def test_loop_upsamples_input_much_slower_than_interp_freq(self) -> None:
    """Checks behavior when incoming data is sparse versus the output grid."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [0, 2], 'a': [0, 20]}]],
      interp_freq=4)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': i / 4, 'a': 10 * i / 4}
      for i in range(9)
    ])

  def test_loop_is_independent_from_block_freq(self) -> None:
    """Checks that freq does not change direct interpolation behavior."""

    batches = [[{'t(s)': [0, 2], 'a': [0, 20]}]]
    slow_mux, slow_sent, _, _ = self._make_multiplexer(
      batches=batches,
      interp_freq=2,
      freq=0.1)
    fast_mux, fast_sent, _, _ = self._make_multiplexer(
      batches=batches,
      interp_freq=2,
      freq=1000)

    slow_mux.loop()
    fast_mux.loop()

    self.assert_sent_almost_equal(slow_sent, fast_sent)
    self.assert_sent_almost_equal(slow_sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 0.5, 'a': 5},
      {'t(s)': 1, 'a': 10},
      {'t(s)': 1.5, 'a': 15},
      {'t(s)': 2, 'a': 20},
    ])

  def test_loop_buffers_last_point_without_duplicate_outputs(self) -> None:
    """Checks that consecutive loops continue without resending old points."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[
        [{'t(s)': [0, 1], 'a': [0, 10]}],
        [{'t(s)': [2], 'a': [20]}],
      ],
      interp_freq=1)

    mux.loop()
    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
      {'t(s)': 2, 'a': 20},
    ])

  def test_loop_keeps_buffer_when_next_grid_point_is_not_available(self
                                                                   ) -> None:
    """Checks that partial future data is kept until a full period exists."""

    mux, sent, _, logs = self._make_multiplexer(
      batches=[
        [{'t(s)': [0, 1], 'a': [0, 10]}],
        [{'t(s)': [1.25], 'a': [12.5]}],
        [{'t(s)': [2], 'a': [20]}],
      ],
      interp_freq=1)

    mux.loop()
    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
    ])
    self.assertIn((logging.DEBUG, "At least one label has values too close "
                                  "together compared to interpolation "
                                  "frequency"), logs)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
      {'t(s)': 2, 'a': 20},
    ])

  def test_loop_accepts_same_label_from_multiple_inputs(self) -> None:
    """Checks that same-label data from several Links is merged and sorted."""

    mux, sent, _, _ = self._make_multiplexer(
      batches=[[
        {'t(s)': [1, 3], 'a': [10, 30]},
        {'t(s)': [0, 2], 'a': [0, 20]},
      ]],
      interp_freq=1)

    mux.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'a': 0},
      {'t(s)': 1, 'a': 10},
      {'t(s)': 2, 'a': 20},
      {'t(s)': 3, 'a': 30},
    ])

  def test_loop_rejects_inconsistent_label_lengths(self) -> None:
    """Checks that incoming label arrays must align with timestamps."""

    mux, _, _, _ = self._make_multiplexer(
      batches=[[{'t(s)': [0, 1], 'a': [0]}]],
      interp_freq=1)

    with self.assertRaises(ValueError):
      mux.loop()
