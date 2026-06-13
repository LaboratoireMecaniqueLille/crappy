# coding: utf-8

import logging
from numbers import Real
from typing import Any
from crappy.blocks.synchronizer import Synchronizer

from ..block import BlockTestBase, TestBlock, link


class TestSynchronizer(BlockTestBase):
  """Unit tests for the Synchronizer Block-specific behavior."""

  def _make_synchronizer(self,
                         batches: list[list[dict[str, Any]]],
                         **kwargs) -> tuple[Synchronizer,
                                            list[dict[str, Any]],
                                            list[None],
                                            list[tuple[int, str]]]:
    """Creates an instrumented Synchronizer for direct method calls."""

    sync = Synchronizer(**kwargs)
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

    sync.recv_all_data_raw = recv_all_data_raw
    sync.send = send
    sync.log = log

    return sync, sent, recv_calls, logs

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

  def test_labels_to_sync_normalization(self) -> None:
    """Checks the supported labels_to_sync forms."""

    self.assertIsNone(Synchronizer(reference_label='ref')._to_sync)
    self.assertEqual(
      Synchronizer(reference_label='ref', labels_to_sync='abc')._to_sync,
      ['abc'])
    self.assertEqual(
      Synchronizer(reference_label='ref',
                   labels_to_sync=('a', 'b'))._to_sync,
      ['a', 'b'])

  def test_prepare_requires_input_and_output_links(self) -> None:
    """Checks that prepare fails early when the Block is not linked enough."""

    sync = Synchronizer(reference_label='ref')

    with self.assertRaises(IOError):
      sync.prepare()

    source = TestBlock()
    sync = Synchronizer(reference_label='ref')
    link(source, sync)

    with self.assertRaises(IOError):
      sync.prepare()

  def test_prepare_accepts_input_and_output_links(self) -> None:
    """Checks that prepare accepts one incoming and one outgoing Link."""

    source = TestBlock()
    sync = Synchronizer(reference_label='ref')
    sink = TestBlock()

    link(source, sync)
    link(sync, sink)

    sync.prepare()

  def test_loop_ignores_data_without_time_label(self) -> None:
    """Checks that untimestamped data is ignored."""

    sync, sent, recv_calls, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[{'x': [1, 2]}]])

    sync.loop()

    self.assertEqual(recv_calls, [None])
    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No data in the buffer to process"), logs)

  def test_loop_does_not_send_time_only_data(self) -> None:
    """Checks that timestamps without labels do not produce an output."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      batches=[[{'t(s)': [0, 1]}]])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No data in the buffer to process"), logs)

  def test_loop_waits_for_reference_label(self) -> None:
    """Checks that labels cannot be synchronized without reference data."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[{'t(s)': [0, 1], 'x': [0, 10]}]])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No value for the reference label found in "
                                  "the buffer"), logs)

  def test_loop_waits_for_requested_labels(self) -> None:
    """Checks that labels_to_sync gates output until each label is buffered."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync=('x',),
      batches=[
        [{'t(s)': [0, 1], 'ref': [100, 101]}],
        [{'t(s)': [0, 1], 'x': [0, 10]}],
      ])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Not all the requested labels received yet"),
                  logs)

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 0},
      {'t(s)': 1, 'ref': 101, 'x': 10},
    ])

  def test_loop_waits_for_two_points_per_label(self) -> None:
    """Checks that a single buffered point cannot be interpolated."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 1], 'ref': [100, 101]},
        {'t(s)': [0], 'x': [0]},
      ]])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Not at least 2 values for each label in "
                                  "buffer"), logs)

  def test_loop_waits_for_overlapping_ranges(self) -> None:
    """Checks that non-overlapping label ranges do not produce an output."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 1], 'ref': [100, 101]},
        {'t(s)': [2, 3], 'x': [20, 30]},
      ]])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "Ranges not matching for interpolation"),
                  logs)

  def test_loop_waits_for_reference_point_in_overlap(self) -> None:
    """Checks that an overlap range without reference timestamp is not sent."""

    sync, sent, _, logs = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 10], 'ref': [100, 110]},
        {'t(s)': [1, 2], 'x': [10, 20]},
      ]])

    sync.loop()

    self.assertEqual(sent, [])
    self.assertIn((logging.DEBUG, "No value of the target label found between "
                                  "the minimum and maximum possible "
                                  "interpolation times"), logs)

  def test_loop_interpolates_single_input_exact_boundaries(self) -> None:
    """Checks that first and last exact reference points are emitted."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      batches=[[{
        't(s)': [0, 1],
        'ref': [100, 101],
        'x': [0, 10],
        'y': [20, 30],
      }]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 0, 'y': 20},
      {'t(s)': 1, 'ref': 101, 'x': 10, 'y': 30},
    ])

  def test_loop_interpolates_on_reference_timestamps(self) -> None:
    """Checks that other labels are interpolated on the reference time base."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 0.25, 0.75, 1], 'ref': [100, 101, 102, 103]},
        {'t(s)': [0, 1], 'x': [0, 10]},
      ]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 0},
      {'t(s)': 0.25, 'ref': 101, 'x': 2.5},
      {'t(s)': 0.75, 'ref': 102, 'x': 7.5},
      {'t(s)': 1, 'ref': 103, 'x': 10},
    ])

  def test_loop_interpolates_unsorted_timestamps(self) -> None:
    """Checks that incoming samples are sorted before interpolation."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [2, 0, 1], 'ref': [20, 0, 10]},
        {'t(s)': [0, 2], 'x': [100, 200]},
      ]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 0, 'x': 100},
      {'t(s)': 1, 'ref': 10, 'x': 150},
      {'t(s)': 2, 'ref': 20, 'x': 200},
    ])

  def test_loop_filters_synchronized_labels(self) -> None:
    """Checks that labels_to_sync limits the interpolated labels."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[{
        't(s)': [0, 1],
        'ref': [100, 101],
        'x': [0, 10],
        'y': [20, 30],
      }]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 0},
      {'t(s)': 1, 'ref': 101, 'x': 10},
    ])

  def test_loop_uses_custom_time_label(self) -> None:
    """Checks that custom time labels are used for input and output."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      time_label='time',
      batches=[[
        {'time': [0, 1], 'ref': [100, 101]},
        {'time': [0, 1], 'x': [0, 10]},
      ]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'time': 0, 'ref': 100, 'x': 0},
      {'time': 1, 'ref': 101, 'x': 10},
    ])

  def test_loop_handles_reference_point_at_zero(self) -> None:
    """Checks that zero-valued reference timestamps are valid output times."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 1], 'ref': [100, 101]},
        {'t(s)': [0, 0.5], 'x': [10, 15]},
      ]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 10},
    ])

  def test_loop_preserves_reference_values_at_interpolation_times(self) -> None:
    """Checks that reference values are aligned with the sent timestamps."""

    sync, sent, recv_calls, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [0, 1, 2], 'ref': [100, 101, 102]},
        {'t(s)': [1, 2], 'x': [10, 20]},
      ]])

    sync.loop()

    self.assertEqual(recv_calls, [None])
    self.assert_sent_almost_equal(sent, [
      {'ref': 101, 'x': 10, 't(s)': 1},
      {'ref': 102, 'x': 20, 't(s)': 2},
    ])

  def test_loop_buffers_without_duplicate_outputs(self) -> None:
    """Checks that consecutive loops continue without resending old points."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[
        [
          {'t(s)': [0, 1], 'ref': [100, 101]},
          {'t(s)': [0, 1], 'x': [0, 10]},
        ],
        [
          {'t(s)': [2, 3], 'ref': [102, 103]},
          {'t(s)': [2, 3], 'x': [20, 30]},
        ],
      ])

    sync.loop()
    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 100, 'x': 0},
      {'t(s)': 1, 'ref': 101, 'x': 10},
      {'t(s)': 2, 'ref': 102, 'x': 20},
      {'t(s)': 3, 'ref': 103, 'x': 30},
    ])

  def test_loop_accepts_same_label_from_multiple_inputs(self) -> None:
    """Checks that same-label data from several Links is merged and sorted."""

    sync, sent, _, _ = self._make_synchronizer(
      reference_label='ref',
      labels_to_sync='x',
      batches=[[
        {'t(s)': [1, 3], 'ref': [10, 30]},
        {'t(s)': [0, 2], 'ref': [0, 20]},
        {'t(s)': [0, 3], 'x': [100, 130]},
      ]])

    sync.loop()

    self.assert_sent_almost_equal(sent, [
      {'t(s)': 0, 'ref': 0, 'x': 100},
      {'t(s)': 1, 'ref': 10, 'x': 110},
      {'t(s)': 2, 'ref': 20, 'x': 120},
      {'t(s)': 3, 'ref': 30, 'x': 130},
    ])

  def test_loop_rejects_inconsistent_label_lengths(self) -> None:
    """Checks that incoming label arrays must align with timestamps."""

    sync, _, _, _ = self._make_synchronizer(
      reference_label='ref',
      batches=[[{'t(s)': [0, 1], 'ref': [0]}]])

    with self.assertRaises(ValueError):
      sync.loop()
