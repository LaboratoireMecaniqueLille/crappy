# coding: utf-8

from unittest import TestCase

import numpy as np

from crappy.modifier import Demux


class TestDemux(TestCase):
  """Unit tests for the Demux Modifier."""

  def test_single_label_is_wrapped_in_tuple(self) -> None:
    """Checks constructor normalization for a single label."""

    modifier = Demux('value')

    self.assertEqual(modifier._labels, ('value',))

  def test_first_column_values_are_extracted_by_default(self) -> None:
    """Checks demuxing the first value of each stream column."""

    data = {
      't(s)': np.array([10, 11]),
      'stream': np.array([[1, 2], [3, 4]]),
      'other': 5,
    }

    returned = Demux(('a', 'b'))(data)

    self.assertIs(returned, data)
    self.assertEqual(returned, {'t(s)': 10.0, 'other': 5,
                                'a': 1.0, 'b': 2.0})
    self.assertNotIn('stream', returned)

  def test_mean_column_values_are_extracted(self) -> None:
    """Checks demuxing the average value of each stream column."""

    data = {
      't(s)': np.array([10, 12]),
      'stream': np.array([[1, 2], [3, 6]]),
    }

    returned = Demux(('a', 'b'), mean=True)(data)

    self.assertEqual(returned, {'t(s)': 11.0, 'a': 2.0, 'b': 4.0})

  def test_first_row_values_are_extracted_when_transposed(self) -> None:
    """Checks demuxing the first value of each stream row."""

    data = {
      't(s)': np.array([10, 11, 12]),
      'stream': np.array([[1, 2, 3], [4, 5, 6]]),
    }

    returned = Demux(('a', 'b'), transpose=True)(data)

    self.assertEqual(returned, {'t(s)': 10.0, 'a': 1.0, 'b': 4.0})

  def test_mean_row_values_are_extracted_when_transposed(self) -> None:
    """Checks demuxing the average value of each stream row."""

    data = {
      't(s)': np.array([10, 11, 12]),
      'stream': np.array([[1, 2, 3], [4, 5, 6]]),
    }

    returned = Demux(('a', 'b'), mean=True, transpose=True)(data)

    self.assertEqual(returned, {'t(s)': 11.0, 'a': 2.0, 'b': 5.0})

  def test_custom_stream_and_time_labels_are_supported(self) -> None:
    """Checks demuxing when stream and time labels are customized."""

    data = {
      'time': np.array([4]),
      'raw': np.array([[7]]),
    }

    returned = Demux('value',
                     stream_label='raw',
                     time_label='time')(data)

    self.assertEqual(returned, {'time': 4.0, 'value': 7.0})

  def test_empty_stream_is_returned_unchanged(self) -> None:
    """Checks that empty stream arrays cannot be demuxed."""

    data = {
      't(s)': np.array([]),
      'stream': np.empty((0, 2)),
    }

    self.assertIs(Demux(('a', 'b'))(data), data)
    self.assertIn('stream', data)

  def test_one_sample_time_array_is_supported(self) -> None:
    """Checks one-point timestamp arrays in non-mean mode."""

    data = {
      't(s)': np.array([42]),
      'stream': np.array([[3]]),
    }

    returned = Demux('value')(data)

    self.assertEqual(returned, {'t(s)': 42.0, 'value': 3.0})

  def test_one_sample_two_dimensional_time_array_is_supported(self) -> None:
    """Checks one-point 2D timestamp arrays in non-mean mode."""

    data = {
      't(s)': np.array([[42]]),
      'stream': np.array([[3]]),
    }

    returned = Demux('value')(data)

    self.assertEqual(returned, {'t(s)': 42.0, 'value': 3.0})
