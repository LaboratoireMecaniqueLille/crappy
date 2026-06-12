# coding: utf-8

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
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
                     available: list[bool] | None = None,
                     **kwargs) -> tuple[Recorder,
                                        list[float | None],
                                        list[tuple[int, str]]]:
    """Creates an instrumented Recorder for direct method calls."""

    recorder = Recorder(path, **kwargs)

    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)
    available_iter = iter([True] if available is None else available)

    def data_available() -> bool:
      return next(available_iter)

    def recv_all_data(delay: float | None = None) -> dict[str, list[Any]]:
      recv_calls.append(delay)
      return {key: list(values) for key, values in next(batches_iter).items()}

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    recorder.data_available = data_available
    recorder.recv_all_data = recv_all_data
    recorder.log = log

    return recorder, recv_calls, logs

  def test_labels_normalization(self) -> None:
    """Checks the supported labels forms."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'

      self.assertIsNone(Recorder(path)._labels)
      self.assertEqual(Recorder(path, labels='abc')._labels, ['abc'])
      self.assertEqual(Recorder(path, labels=('a', 'b'))._labels, ['a', 'b'])

  def test_prepare_requires_one_input_link(self) -> None:
    """Checks that prepare fails early when the Block is not linked right."""

    with TemporaryDirectory() as folder:
      recorder = Recorder(Path(folder) / 'data.csv')

      with self.assertRaises(ValueError):
        recorder.prepare()

      source_1 = TestBlock()
      source_2 = TestBlock()
      recorder = Recorder(Path(folder) / 'data.csv')

      link(source_1, recorder)
      link(source_2, recorder)

      with self.assertRaises(ValueError):
        recorder.prepare()

  def test_prepare_accepts_one_input_link(self) -> None:
    """Checks that prepare accepts a single incoming Link."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      recorder = Recorder(Path(folder) / 'data.csv')
      link(source, recorder)

      recorder.prepare()

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

  def test_loop_waits_for_first_data(self) -> None:
    """Checks that no file is created before the first data is available."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, recv_calls, _ = self._make_recorder(
        path,
        batches=[],
        available=[False],
        labels=('a', 'b'))

      recorder.loop()

      self.assertFalse(path.exists())
      self.assertFalse(recorder._file_initialized)
      self.assertEqual(recv_calls, [])

  def test_loop_initializes_labels_from_first_batch(self) -> None:
    """Checks that labels are inferred from the first received data."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, recv_calls, _ = self._make_recorder(
        path,
        delay=0.5,
        batches=[{'a': [1, 2], 'b': ['x', 'y']}])

      recorder.loop()

      self.assertEqual(recorder._labels, ['a', 'b'])
      self.assertEqual(recv_calls, [0.5])
      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['1', 'x'],
        ['2', 'y'],
      ])

  def test_loop_respects_requested_labels_and_order(self) -> None:
    """Checks that only requested labels are saved in the requested order."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        labels=('b', 'a'),
        batches=[{'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}])

      recorder.loop()

      self.assertEqual(self._read_csv(path), [
        ['b', 'a'],
        ['3', '1'],
        ['4', '2'],
      ])

  def test_loop_appends_subsequent_batches(self) -> None:
    """Checks that later data is appended without rewriting the header."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, recv_calls, _ = self._make_recorder(
        path,
        delay=1.5,
        labels=('a', 'b'),
        batches=[
          {'a': [1, 2], 'b': [10, 20]},
          {'a': [3], 'b': [30]},
        ])

      recorder.loop()
      recorder.loop()

      self.assertEqual(recv_calls, [1.5, 1.5])
      self.assertEqual(self._read_csv(path), [
        ['a', 'b'],
        ['1', '10'],
        ['2', '20'],
        ['3', '30'],
      ])

  def test_loop_writes_valid_csv(self) -> None:
    """Checks that labels and values are escaped as valid CSV fields."""

    labels = ['time,s', 'text"label']
    value = 'hello, "world"\nnext'

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.csv'
      recorder, _, _ = self._make_recorder(
        path,
        labels=labels,
        batches=[{
          'time,s': [1],
          'text"label': [value],
        }])

      recorder.loop()

      self.assertEqual(self._read_csv(path), [
        labels,
        ['1', value],
      ])
