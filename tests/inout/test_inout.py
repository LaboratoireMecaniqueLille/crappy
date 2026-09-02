# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np

from crappy._global import DefinitionError
from crappy.inout.meta_inout import InOut
import crappy.inout.meta_inout.inout as inout_module


class SequencedInOutForTest(InOut):
  """InOut test double returning predefined data and stream values."""

  def __init__(self, data=None, stream=None) -> None:
    super().__init__()
    self.data = list(data or ())
    self.stream = stream

  def get_data(self):
    if not self.data:
      return
    return self.data.pop(0)

  def get_stream(self):
    return self.stream


InOut.classes.pop('SequencedInOutForTest', None)


class TestInOut(TestCase):
  """Unit tests for the InOut base class."""

  def tearDown(self) -> None:
    """Removes dynamic classes registered during tests."""

    for name in ('TestRegisteredInOut', 'TestDuplicateInOut'):
      InOut.classes.pop(name, None)

  def test_subclasses_are_registered_by_name(self) -> None:
    """Checks that new InOut subclasses are added to the class registry."""

    InOut.classes.pop('TestRegisteredInOut', None)

    cls = type('TestRegisteredInOut', (InOut,), {})

    self.assertIs(InOut.classes['TestRegisteredInOut'], cls)

  def test_duplicate_subclass_names_are_rejected(self) -> None:
    """Checks that two InOuts cannot share a class name."""

    InOut.classes.pop('TestDuplicateInOut', None)
    type('TestDuplicateInOut', (InOut,), {})

    with self.assertRaises(DefinitionError):
      type('TestDuplicateInOut', (InOut,), {})

  def test_default_attributes_and_ft232h_flag(self) -> None:
    """Checks base instance initialization and class defaults."""

    inout = InOut('unused', option='ignored')

    self.assertFalse(inout.ft232h)
    self.assertEqual(inout._compensations, [])
    self.assertEqual(inout._compensations_dict, {})
    self.assertIsNone(inout._logger)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and lazy initialization."""

    inout = InOut()
    logger_name = f"{current_process().name}.InOut"

    with self.assertLogs(logger_name, level='INFO') as captured:
      inout.log(logging.INFO, "message")

    self.assertIs(inout._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_open_and_close_are_noops_by_default(self) -> None:
    """Checks that optional lifecycle methods can be left undefined."""

    inout = InOut()

    self.assertIsNone(inout.open())
    self.assertIsNone(inout.close())

  def test_default_methods_log_and_wait_where_needed(self) -> None:
    """Checks fallback behavior for undefined acquisition and command
    methods."""

    inout = InOut()
    logs = list()
    inout.log = lambda level, msg: logs.append((level, msg))

    with patch.object(inout_module, 'sleep') as mocked_sleep:
      self.assertIsNone(inout.get_data())
      self.assertIsNone(inout.set_cmd(1, 2))
      self.assertIsNone(inout.get_stream())
      self.assertIsNone(inout.start_stream())
      self.assertIsNone(inout.stop_stream())

    self.assertEqual(mocked_sleep.call_args_list,
                     [call(1), call(1), call(1)])
    self.assertTrue(all(level == logging.WARNING for level, _ in logs))
    self.assertIn('get_data', logs[0][1])
    self.assertIn('set_cmd', logs[1][1])
    self.assertIn('get_stream', logs[2][1])
    self.assertIn('start_stream', logs[3][1])
    self.assertIn('stop_stream', logs[4][1])

  def test_make_zero_stores_offsets_for_iterable_data(self) -> None:
    """Checks zeroing with timestamp-first iterable samples."""

    inout = SequencedInOutForTest([[10, 1, 3], [11, 3, 5], [12, 4, 6]])

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations, [-2, -4])
    self.assertEqual(inout._compensations_dict, {})
    self.assertEqual(inout.return_data(), [12, 2, 2])

  def test_make_zero_replaces_previous_iterable_offsets(self) -> None:
    """Checks that repeated zeroing does not accumulate stale offsets."""

    inout = SequencedInOutForTest([[0, 2], [1, 4]])

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    inout.data = [[2, 10], [3, 14]]
    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations, [-12])

  def test_make_zero_stores_offsets_for_dict_data(self) -> None:
    """Checks zeroing with dict samples and time label removal."""

    inout = SequencedInOutForTest([
      {'t(s)': 10, 'a': 1, 'b': 3},
      {'t(s)': 11, 'a': 3, 'b': 5},
      {'t(s)': 12, 'a': 4, 'b': 6},
    ])

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations, [])
    self.assertEqual(inout._compensations_dict, {'a': -2, 'b': -4})
    self.assertEqual(inout.return_data(), {'t(s)': 12, 'a': 2, 'b': 2})

  def test_make_zero_replaces_previous_dict_offsets(self) -> None:
    """Checks that removed dict labels do not keep stale offsets."""

    inout = SequencedInOutForTest([{'t(s)': 0, 'a': 1, 'b': 2}])

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.2]):
      inout.make_zero(0.2)

    inout.data = [{'t(s)': 1, 'a': 5}]
    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations_dict, {'a': -5})

  def test_make_zero_warns_when_no_data_was_acquired(self) -> None:
    """Checks zeroing abort when get_data only returns None."""

    inout = SequencedInOutForTest([None, None])
    logs = list()
    inout.log = lambda level, msg: logs.append((level, msg))

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations, [])
    self.assertEqual(inout._compensations_dict, {})
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn('No data acquired', logs[0][1])

  def test_make_zero_clears_offsets_when_values_are_not_numeric(self) -> None:
    """Checks zeroing abort when iterable values cannot be averaged."""

    inout = SequencedInOutForTest([[0, 'bad'], [1, 'worse']])
    logs = list()
    inout.log = lambda level, msg: logs.append((level, msg))

    with patch.object(inout_module, 'time',
                      side_effect=[0, 0, 0.1, 0.2]):
      inout.make_zero(0.2)

    self.assertEqual(inout._compensations, [])
    self.assertEqual(logs[0][0], logging.WARNING)
    self.assertIn('Cannot calculate the offset', logs[0][1])

  def test_return_data_without_offsets_returns_iterable_data_as_list(self):
    """Checks return_data without zeroing for iterable samples."""

    inout = SequencedInOutForTest([(1, 2, 3)])

    self.assertEqual(inout.return_data(), [1, 2, 3])

  def test_return_data_without_offsets_returns_dict_data_as_is(self) -> None:
    """Checks return_data without zeroing for dict samples."""

    data = {'t(s)': 1, 'a': 2}
    inout = SequencedInOutForTest([data])

    self.assertIs(inout.return_data(), data)

  def test_return_data_returns_none_when_no_data_is_available(self) -> None:
    """Checks return_data when get_data returns None."""

    inout = SequencedInOutForTest()

    self.assertIsNone(inout.return_data())

  def test_return_data_rejects_dict_without_time_label(self) -> None:
    """Checks that dict data must contain the time label."""

    inout = SequencedInOutForTest([{'a': 1}])

    with self.assertRaises(ValueError):
      inout.return_data()

  def test_return_data_rejects_missing_dict_offsets(self) -> None:
    """Checks that all dict labels must have an offset when zeroed."""

    inout = SequencedInOutForTest([{'t(s)': 0, 'a': 1, 'b': 2}])
    inout._compensations_dict = {'a': -1}

    with self.assertRaises(ValueError):
      inout.return_data()

  def test_return_data_rejects_iterable_offset_count_mismatch(self) -> None:
    """Checks iterable offset count validation."""

    inout = SequencedInOutForTest([[0, 1, 2]])
    inout._compensations = [-1]

    with self.assertRaises(ValueError):
      inout.return_data()

  def test_return_stream_without_offsets_returns_iterable_stream_as_list(self):
    """Checks return_stream without zeroing for iterable streams."""

    t = np.array([0, 1])
    values = np.array([[1, 2], [3, 4]])
    inout = SequencedInOutForTest(stream=(t, values))

    stream = inout.return_stream()

    self.assertIsInstance(stream, list)
    self.assertIs(stream[0], t)
    self.assertIs(stream[1], values)

  def test_return_stream_without_offsets_returns_dict_stream_as_is(self):
    """Checks return_stream without zeroing for dict streams."""

    stream = {'t(s)': np.array([0, 1]), 'a': np.array([[1], [2]])}
    inout = SequencedInOutForTest(stream=stream)

    self.assertIs(inout.return_stream(), stream)

  def test_return_stream_returns_none_when_no_stream_is_available(
      self) -> None:
    """Checks return_stream when get_stream returns None."""

    inout = SequencedInOutForTest()

    self.assertIsNone(inout.return_stream())

  def test_return_stream_offsets_iterable_stream_with_list_offsets(self):
    """Checks stream compensation from iterable make_zero offsets."""

    t = np.array([0, 1])
    values = np.array([[5, 7], [8, 10]])
    inout = SequencedInOutForTest(stream=[t, values])
    inout._compensations = [-1, -2]

    stream = inout.return_stream()

    np.testing.assert_array_equal(stream[0], t)
    np.testing.assert_array_equal(stream[1], np.array([[4, 5], [7, 8]]))

  def test_return_stream_offsets_iterable_stream_with_dict_offsets(self):
    """Checks stream compensation from dict make_zero offsets."""

    t = np.array([0, 1])
    values = np.array([[5, 7], [8, 10]])
    inout = SequencedInOutForTest(stream=[t, values])
    inout._compensations_dict = {'a': -1, 'b': -2}

    stream = inout.return_stream()

    np.testing.assert_array_equal(stream[0], t)
    np.testing.assert_array_equal(stream[1], np.array([[4, 5], [7, 8]]))

  def test_return_stream_offsets_all_arrays_in_dict_stream(self) -> None:
    """Checks that every non-time array in a dict stream is compensated."""

    t = np.array([0, 1])
    stream = {
      't(s)': t,
      'a': np.array([[5, 7], [8, 10]]),
      'b': np.array([[15, 17], [18, 20]]),
    }
    inout = SequencedInOutForTest(stream=stream)
    inout._compensations = [-1, -2]

    returned = inout.return_stream()

    np.testing.assert_array_equal(returned['t(s)'], t)
    np.testing.assert_array_equal(returned['a'],
                                  np.array([[4, 5], [7, 8]]))
    np.testing.assert_array_equal(returned['b'],
                                  np.array([[14, 15], [17, 18]]))

  def test_return_stream_rejects_dict_without_time_label(self) -> None:
    """Checks that dict streams must contain the time label."""

    inout = SequencedInOutForTest(stream={'a': np.array([[1], [2]])})

    with self.assertRaises(ValueError):
      inout.return_stream()

  def test_return_stream_rejects_offset_shape_mismatch(self) -> None:
    """Checks stream offset shape validation."""

    inout = SequencedInOutForTest(stream=[
      np.array([0, 1]),
      np.array([[1, 2], [3, 4]]),
    ])
    inout._compensations = [-1]

    with self.assertRaises(ValueError):
      inout.return_stream()
