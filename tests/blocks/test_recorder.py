# coding: utf-8

import csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from crappy.blocks.recorder import Recorder

from ..block import BlockTestBase, TestBlock, link


class TestRecorder(BlockTestBase):
  """Unit tests for the Recorder Block-specific behavior."""

  @staticmethod
  def _read_csv(path: Path) -> list[list[str]]:
    """Reads a Recorder output file using the CSV parser."""

    with open(path, newline='') as file:
      return list(csv.reader(file))

  def _make_recorder(self,
                      path: Path,
                      batches: list[dict[str, list[Any]]],
                      **kwargs) -> tuple[Recorder,
                                         list[None],
                                         list[tuple[int, str]]]:
    """Creates an instrumented Recorder for direct method calls."""

    recorder = Recorder(path, **kwargs)
    recorder._last_write_t = 10.0

    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)

    def recv_all_data() -> dict[str, list[Any]]:
      recv_calls.append(None)
      return {key: list(values) for key, values in next(batches_iter).items()}

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    recorder.recv_all_data = recv_all_data
    recorder.log = log

    return recorder, recv_calls, logs

  def test_file_name_is_validated(self) -> None:
    """Checks that the output path identifies a file."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'

      self.assertEqual(Recorder(path)._path, path)
      self.assertEqual(Recorder(str(path))._path, path)

      for invalid in ('', ' ', '.', Path('.')):
        with self.subTest(invalid=invalid):
          with self.assertRaises(ValueError):
            Recorder(invalid)

      with self.assertRaises(TypeError):
        Recorder(1)

  def test_delay_is_validated(self) -> None:
    """Checks that the write delay is finite, numeric, and positive."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'

      for delay in (0, -1, float('nan'), float('inf'), float('-inf')):
        with self.subTest(delay=delay):
          with self.assertRaises(ValueError):
            Recorder(path, delay=delay)

      with self.assertRaises(TypeError):
        Recorder(path, delay='1')

      self.assertEqual(Recorder(path, delay=True)._delay, True)

  def test_labels_are_normalized_and_validated(self) -> None:
    """Checks the supported labels forms and rejects invalid values."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder = Recorder(path, labels=('a', 'b'))

      self.assertIsNone(Recorder(path)._requested_labels)
      self.assertEqual(Recorder(path)._recorder_labels, [])
      self.assertEqual(Recorder(path, labels='abc')._recorder_labels, ['abc'])
      self.assertEqual(recorder._recorder_labels, ['a', 'b'])

      for labels in ('', (), [], ('a', ''), ('a', 1)):
        with self.subTest(labels=labels):
          with self.assertRaises(ValueError):
            Recorder(path, labels=labels)

      with self.assertRaises(TypeError):
        Recorder(path, labels={'a', 'b'})

  def test_prepare_requires_one_input_link(self) -> None:
    """Checks that prepare fails early when the Block is not linked right."""

    with TemporaryDirectory() as folder:
      recorder = Recorder(Path(folder) / 'data.csv')

      with self.assertRaises(IOError):
        recorder.prepare()

      source_1 = TestBlock()
      source_2 = TestBlock()
      recorder = Recorder(Path(folder) / 'data.csv')

      link(source_1, recorder)
      link(source_2, recorder)

      with self.assertRaises(IOError):
        recorder.prepare()

      source = TestBlock()
      recorder = Recorder(Path(folder) / 'data.csv')
      sink = TestBlock()
      link(source, recorder)
      link(recorder, sink)

      with self.assertRaises(IOError):
        recorder.prepare()

  def test_prepare_accepts_one_input_link(self) -> None:
    """Checks that prepare accepts a single incoming Link."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      recorder = Recorder(Path(folder) / 'data.csv')
      link(source, recorder)

      recorder.prepare()

      self.assertTrue(recorder._path.exists())
      self.assertEqual(recorder._path.read_bytes(), b'')

  def test_prepare_creates_parent_folder(self) -> None:
    """Checks that prepare creates missing parent folders."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      path = Path(folder) / 'nested' / 'data.csv'
      recorder = Recorder(path)
      link(source, recorder)

      recorder.prepare()

      self.assertTrue(path.parent.is_dir())
      self.assertEqual(recorder._path, path)

  def test_prepare_renames_existing_file(self) -> None:
    """Checks that existing files are not overwritten."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      path = Path(folder) / 'data.csv'
      path.write_text('existing\n')
      (Path(folder) / 'data_00001.csv').write_text('existing\n')

      recorder = Recorder(path)
      link(source, recorder)

      recorder.prepare()

      self.assertEqual(recorder._path, Path(folder) / 'data_00002.csv')
      self.assertEqual(path.read_text(), 'existing\n')

  def test_prepare_rejects_existing_directory(self) -> None:
    """Checks that an existing directory cannot be used as an output file."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      path = Path(folder) / 'output'
      path.mkdir()
      recorder = Recorder(path)
      link(source, recorder)

      with self.assertRaises(IsADirectoryError):
        recorder.prepare()

  def test_begin_initializes_write_timer(self) -> None:
    """Checks that begin starts a new write interval."""

    with TemporaryDirectory() as folder:
      recorder = Recorder(Path(folder) / 'data.csv')

      with patch('crappy.blocks.recorder.monotonic', return_value=20):
        recorder.begin()

      self.assertEqual(recorder._last_write_t, 20)

  def test_loop_returns_when_no_data_is_available(self) -> None:
    """Checks that an empty read does not initialize or modify the file."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, recv_calls, _ = self._make_recorder(path, batches=[{}])

      recorder.loop()

      self.assertEqual(recv_calls, [None])
      self.assertFalse(path.exists())
      self.assertFalse(recorder._file_initialized)
      self.assertEqual(recorder._last_write_t, 10)

  def test_loop_buffers_data_until_delay_is_reached(self) -> None:
    """Checks non-blocking reads accumulate a complete write interval."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, recv_calls, _ = self._make_recorder(
        path,
        delay=2,
        batches=[
          {'a': [1, 2], 'b': [10, 20]},
          {'a': [3], 'b': [30]},
        ])

      with patch('crappy.blocks.recorder.monotonic',
                 side_effect=(11, 12, 12)):
        recorder.loop()
        self.assertFalse(path.exists())
        self.assertEqual(dict(recorder._data_buf),
                         {'a': [1, 2], 'b': [10, 20]})

        recorder.loop()

      self.assertEqual(recv_calls, [None, None])
      self.assertEqual(recorder._last_write_t, 12)
      self.assertEqual(dict(recorder._data_buf), {})
      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['1', '10'],
        ['2', '20'],
        ['3', '30'],
      ])

  def test_loop_respects_requested_labels_and_order(self) -> None:
    """Checks that only requested labels are saved in the requested order."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=1,
        labels=('b', 'a'),
        batches=[{'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}])

      with patch('crappy.blocks.recorder.monotonic', return_value=11):
        recorder.loop()

      self.assertEqual(self._read_csv(path), [
        ['b', 'a'],
        ['3', '1'],
        ['4', '2'],
      ])

  def test_inferred_labels_ignore_later_extra_labels(self) -> None:
    """Checks that the first write window fixes the inferred label set."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, logs = self._make_recorder(
        path,
        delay=1,
        batches=[
          {'a': [1], 'b': [10]},
          {'a': [2], 'b': [20], 'c': [30]},
        ])

      with patch('crappy.blocks.recorder.monotonic',
                 side_effect=(11, 11, 12, 12)):
        recorder.loop()
        recorder.loop()

      self.assertEqual(recorder._recorder_labels, ['a', 'b'])
      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['1', '10'],
        ['2', '20'],
      ])
      warnings = [msg for level, msg in logs if level == logging.WARNING]
      self.assertEqual(len(warnings), 1)
      self.assertIn('c', warnings[0])

  def test_write_rejects_missing_requested_label(self) -> None:
    """Checks that every requested column must be available before writing."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=1,
        labels=('a', 'b'),
        batches=[{'a': [1]}])

      with (patch('crappy.blocks.recorder.monotonic', return_value=11),
            self.assertRaises(IOError)):
        recorder.loop()

      self.assertFalse(path.exists())

  def test_write_rejects_columns_with_different_lengths(self) -> None:
    """Checks that inconsistent columns do not produce partial data rows."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=1,
        labels=('a', 'b'),
        batches=[{'a': [1, 2], 'b': [10]}])

      with (patch('crappy.blocks.recorder.monotonic', return_value=11),
            self.assertRaises(IOError)):
        recorder.loop()

      self.assertEqual(self._read_csv(path), [['a', 'b']])

  def test_finish_flushes_buffer_before_delay(self) -> None:
    """Checks that shutdown writes a complete pending buffer."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=5,
        labels=('a', 'b'),
        batches=[{'a': [1], 'b': [2]}])

      with patch('crappy.blocks.recorder.monotonic', return_value=11):
        recorder.loop()
      self.assertFalse(path.exists())

      recorder.finish()

      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['1', '2'],
      ])

  def test_none_is_written_as_an_empty_field(self) -> None:
    """Checks the documented CSV representation of None values."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=1,
        labels=('a', 'b'),
        batches=[{'a': [None], 'b': ['']}])

      with patch('crappy.blocks.recorder.monotonic', return_value=11):
        recorder.loop()

      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['', ''],
      ])

  def test_loop_writes_valid_csv(self) -> None:
    """Checks that labels and values are escaped as valid CSV fields."""

    labels = ['time,s', 'text"label']
    value = 'hello, "world"\nnext'

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        delay=1,
        labels=labels,
        batches=[{
          'time,s': [1],
          'text"label': [value],
        }])

      with patch('crappy.blocks.recorder.monotonic', return_value=11):
        recorder.loop()

      self.assertEqual(self._read_csv(path), [
        labels,
        ['1', value],
      ])
