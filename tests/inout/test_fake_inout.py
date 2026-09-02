# coding: utf-8

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from crappy.inout.fake_inout import FakeInOut
import crappy.inout.fake_inout as fake_inout_module


class TestFakeInOut(TestCase):
  """Unit tests for the FakeInOut object."""

  def test_open_initializes_buffer_and_close_deletes_it(self) -> None:
    """Checks the fake memory buffer lifecycle."""

    inout = FakeInOut()

    self.assertIsNone(inout._buf)

    inout.open()
    self.assertEqual(inout._buf, [])

    inout.close()
    self.assertFalse(hasattr(inout, '_buf'))

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
    """Checks that memory usage is increased through the internal buffer."""

    inout = FakeInOut()
    inout.open()

    with patch.object(fake_inout_module,
                      'virtual_memory',
                      return_value=SimpleNamespace(percent=25)) as mocked_mem:
      inout.set_cmd(30)

    mocked_mem.assert_called_once_with()
    self.assertEqual(len(inout._buf), 1)
    self.assertEqual(len(inout._buf[0]), 1024 * 1024)

  def test_set_cmd_deletes_chunk_when_target_is_below_memory(self) -> None:
    """Checks that memory usage is reduced through the internal buffer."""

    inout = FakeInOut()
    inout.open()
    chunk = object()
    inout._buf.append(chunk)

    with patch.object(fake_inout_module,
                      'virtual_memory',
                      return_value=SimpleNamespace(percent=50)) as mocked_mem:
      inout.set_cmd(40)
      inout.set_cmd(40)

    self.assertEqual(mocked_mem.call_count, 2)
    self.assertEqual(inout._buf, [])

  def test_set_cmd_keeps_buffer_when_target_matches_memory(self) -> None:
    """Checks that equal command and memory percent is a no-op."""

    inout = FakeInOut()
    inout.open()
    chunk = object()
    inout._buf.append(chunk)

    with patch.object(fake_inout_module,
                      'virtual_memory',
                      return_value=SimpleNamespace(percent=50)) as mocked_mem:
      inout.set_cmd(50.0)

    mocked_mem.assert_called_once_with()
    self.assertEqual(inout._buf, [chunk])

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

  def test_get_stream_batches_ten_get_data_calls(self) -> None:
    """Checks streamer output shape and contents."""

    inout = FakeInOut()
    samples = [[float(i), float(i + 100)] for i in range(10)]

    with patch.object(inout, 'get_data', side_effect=samples) as mocked_get:
      t, values = inout.get_stream()

    self.assertEqual(mocked_get.call_count, 10)
    np.testing.assert_array_equal(t, np.arange(10, dtype=float))
    np.testing.assert_array_equal(values,
                                  np.arange(100, 110, dtype=float)[:, None])
