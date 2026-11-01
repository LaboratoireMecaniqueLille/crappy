# coding: utf-8

import gc
from itertools import count
import tracemalloc
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from crappy.inout.fake_inout import FakeInOut
import crappy.inout.fake_inout as fake_inout_module


class TestFakeInOut(TestCase):
  """Unit tests for the FakeInOut object."""

  def memory_change(self, operation) -> int:
    """Observe retained Python memory without requiring a buffer representation."""

    if not tracemalloc.is_tracing():
      tracemalloc.start()
      self.addCleanup(tracemalloc.stop)
    gc.collect()
    before = tracemalloc.get_traced_memory()[0]
    operation()
    gc.collect()
    return tracemalloc.get_traced_memory()[0] - before

  def test_open_initializes_buffer_and_close_deletes_it(self) -> None:
    """Closing releases allocated memory and the object can be opened again."""

    inout = FakeInOut()
    inout.open()
    self.addCleanup(inout.close)
    with patch.object(fake_inout_module, 'virtual_memory',
                      new=lambda: SimpleNamespace(percent=25)):
      self.assertGreater(self.memory_change(lambda: inout.set_cmd(30)), 1024)
      self.assertLess(self.memory_change(inout.close), -1024)
      inout.open()
      inout.set_cmd(0)
      self.assertEqual(inout.get_data()[1], 25)

  def test_set_cmd_rejects_non_numeric_command(self) -> None:
    """Checks command type validation."""

    inout = FakeInOut()
    inout.open()

    with self.assertRaises(TypeError):
      inout.set_cmd('50')

  def test_set_cmd_rejects_commands_outside_percentage_range(self) -> None:
    """Checks command range validation."""

    inout = FakeInOut()
    inout.open()

    with self.assertRaises(ValueError):
      inout.set_cmd(-1)
    with self.assertRaises(ValueError):
      inout.set_cmd(101)

  def test_set_cmd_appends_chunk_when_target_is_above_memory(self) -> None:
    """A demand above current usage allocates retained memory."""

    inout = FakeInOut()
    inout.open()
    self.addCleanup(inout.close)
    with patch.object(fake_inout_module, 'virtual_memory',
                      new=lambda: SimpleNamespace(percent=25)):
      self.assertGreater(self.memory_change(lambda: inout.set_cmd(30)), 1024)

  def test_set_cmd_deletes_chunk_when_target_is_below_memory(self) -> None:
    """A lower demand releases memory and tolerates an already empty device."""

    inout = FakeInOut()
    inout.open()
    self.addCleanup(inout.close)
    with patch.object(fake_inout_module, 'virtual_memory',
                      new=lambda: SimpleNamespace(percent=50)):
      self.assertGreater(self.memory_change(lambda: inout.set_cmd(60)), 1024)
      self.assertLess(self.memory_change(lambda: inout.set_cmd(40)), -1024)
      self.assertLess(abs(self.memory_change(lambda: inout.set_cmd(40))), 1024)

  def test_set_cmd_keeps_buffer_when_target_matches_memory(self) -> None:
    """Matching the reported usage leaves retained memory unchanged."""

    inout = FakeInOut()
    inout.open()
    self.addCleanup(inout.close)
    with patch.object(fake_inout_module, 'virtual_memory',
                      new=lambda: SimpleNamespace(percent=50)):
      self.assertGreater(self.memory_change(lambda: inout.set_cmd(60)), 1024)
      self.assertLess(abs(self.memory_change(lambda: inout.set_cmd(50))), 1024)

  def test_get_data_returns_timestamp_and_memory_percent(self) -> None:
    """Checks regular data acquisition."""

    inout = FakeInOut()

    with patch.object(fake_inout_module, 'time', return_value=12.5):
      with patch.object(fake_inout_module,
                        'virtual_memory',
                        return_value=SimpleNamespace(percent=37.5)):
        self.assertEqual(inout.get_data(), [12.5, 37.5])

  def test_start_and_stop_stream_are_noops(self) -> None:
    """Checks stream lifecycle methods only silence base warnings."""

    inout = FakeInOut()

    self.assertIsNone(inout.start_stream())
    self.assertIsNone(inout.stop_stream())

  def test_stream_preserves_sample_order_and_alignment(self) -> None:
    """Streaming preserves sample order and time/value alignment in a batch."""

    inout = FakeInOut()
    samples = count()

    def get_data():
      value = float(next(samples))
      return [value, value + 100]

    with patch.object(inout, 'get_data', new=get_data):
      timestamps, values = inout.get_stream()

    self.assertEqual(timestamps.ndim, 1)
    self.assertEqual(values.shape, (len(timestamps), 1))
    self.assertGreater(len(timestamps), 0)
    np.testing.assert_array_equal(timestamps, np.arange(len(timestamps)))
    np.testing.assert_array_equal(values[:, 0], timestamps + 100)
