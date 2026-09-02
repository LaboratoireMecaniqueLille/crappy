# coding: utf-8

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest

import numpy as np

from crappy._global import OptionalModule
from crappy.blocks.hdf_recorder import HDFRecorder
import crappy.blocks.hdf_recorder as hdf_module

from ..block import BlockTestBase, TestBlock, link


@unittest.skipIf(isinstance(hdf_module.tables, OptionalModule),
                 'PyTables is not available')
class TestHDFRecorder(BlockTestBase):
  """Unit tests for the HDFRecorder Block-specific behavior."""

  @staticmethod
  def _read_node(path: Path, node: str = 'table') -> np.ndarray:
    """Reads an HDF5 node into memory."""

    with hdf_module.tables.open_file(str(path), 'r') as hfile:
      return hfile.get_node('/', node).read()

  @staticmethod
  def _read_metadata(path: Path, name: str) -> np.ndarray:
    """Reads a metadata array from an HDF5 file."""

    with hdf_module.tables.open_file(str(path), 'r') as hfile:
      return hfile.get_node('/', name).read()

  def _make_recorder(self,
                     path: Path,
                     batches: list[dict[str, list[Any]]],
                     available: list[bool] | None = None,
                     **kwargs) -> tuple[HDFRecorder,
                                        list[None],
                                        list[tuple[int, str]]]:
    """Creates a linked HDFRecorder with deterministic input data."""

    source = TestBlock()
    recorder = HDFRecorder(path, **kwargs)
    link(source, recorder)

    recv_calls = list()
    logs = list()
    batches_iter = iter(batches)
    available_iter = iter([True] if available is None else available)

    def data_available() -> bool:
      return next(available_iter)

    def recv_all_data() -> dict[str, list[Any]]:
      recv_calls.append(None)
      return {key: list(values) for key, values in next(batches_iter).items()}

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    recorder.data_available = data_available
    recorder.recv_all_data = recv_all_data
    recorder.log = log
    recorder.prepare()

    return recorder, recv_calls, logs

  def test_init_sets_block_options_and_normalizes_atom(self) -> None:
    """Checks HDFRecorder-specific initialization."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'
      recorder = HDFRecorder(path,
                             atom='float64',
                             label='samples',
                             flush_period=None,
                             freq=200,
                             display_freq=True,
                             debug=True)

      self.assertEqual(recorder.freq, 200)
      self.assertTrue(recorder.display_freq)
      self.assertTrue(recorder.debug)
      self.assertEqual(recorder._path, path)
      self.assertEqual(recorder._label, 'samples')
      self.assertIsNone(recorder._flush_period)
      self.assertEqual(recorder._atom.dtype, np.dtype('float64'))

      atom = hdf_module.tables.Float32Atom()
      self.assertIs(HDFRecorder(path, atom=atom)._atom, atom)

  def test_init_rejects_invalid_row_and_flush_settings(self) -> None:
    """Checks constructor validation for row estimates and flushing."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'

      for expected_rows in (0, -1, 1.5, '10', None):
        with self.subTest(expected_rows=expected_rows):
          with self.assertRaises(ValueError):
            HDFRecorder(path, expected_rows=expected_rows, atom='float64')

      for flush_period in (0, -1, 1.5, '3'):
        with self.subTest(flush_period=flush_period):
          with self.assertRaises(ValueError):
            HDFRecorder(path, atom='float64', flush_period=flush_period)

      self.assertIsNone(HDFRecorder(path, atom='float64')._flush_period)
      self.assertEqual(HDFRecorder(path, atom='float64',
                                   flush_period=2)._flush_period, 2)

  def test_prepare_requires_one_input_link(self) -> None:
    """Checks that prepare fails early when the Block is not linked right."""

    with TemporaryDirectory() as folder:
      recorder = HDFRecorder(Path(folder) / 'data.h5', atom='float64')

      with self.assertRaises(ValueError):
        recorder.prepare()

      source_1 = TestBlock()
      source_2 = TestBlock()
      recorder = HDFRecorder(Path(folder) / 'data.h5', atom='float64')
      link(source_1, recorder)
      link(source_2, recorder)

      with self.assertRaises(ValueError):
        recorder.prepare()

  def test_prepare_creates_parent_folder_and_writes_metadata(self) -> None:
    """Checks that prepare creates the file and stores metadata arrays."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'nested' / 'data.h5'
      source = TestBlock()
      recorder = HDFRecorder(path,
                             atom='float64',
                             metadata={
                               'sample_rate': np.array([1000]),
                               'channels': np.array([1, 2]),
                             })
      link(source, recorder)

      recorder.prepare()
      recorder.finish()

      self.assertTrue(path.parent.is_dir())
      self.assertTrue(path.exists())
      np.testing.assert_array_equal(self._read_metadata(path, 'sample_rate'),
                                    np.array([1000]))
      np.testing.assert_array_equal(self._read_metadata(path, 'channels'),
                                    np.array([1, 2]))

  def test_prepare_renames_existing_file(self) -> None:
    """Checks that existing files are not overwritten."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'
      path.write_text('existing\n')
      (Path(folder) / 'data_00001.h5').write_text('existing\n')

      source = TestBlock()
      recorder = HDFRecorder(path, atom='float64')
      link(source, recorder)

      recorder.prepare()
      recorder.finish()

      self.assertEqual(recorder._path, Path(folder) / 'data_00002.h5')
      self.assertEqual(path.read_text(), 'existing\n')
      self.assertTrue(recorder._path.exists())

  def test_loop_waits_for_first_stream_data(self) -> None:
    """Checks that no array is created until data is available."""

    with TemporaryDirectory() as folder:
      recorder, recv_calls, _ = self._make_recorder(
        Path(folder) / 'data.h5',
        batches=[],
        available=[False],
        atom='float64')

      try:
        recorder.loop()

        self.assertFalse(recorder._array_initialized)
        self.assertEqual(recv_calls, [])
      finally:
        recorder.finish()

  def test_loop_initializes_array_and_writes_first_batch(self) -> None:
    """Checks that the first received stream creates and fills the EArray."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'
      recorder, recv_calls, logs = self._make_recorder(
        path,
        batches=[{
          'stream': [np.array([[1.0, 2.0], [3.0, 4.0]])],
        }],
        atom='float64')

      try:
        recorder.loop()

        self.assertTrue(recorder._array_initialized)
        self.assertEqual(recorder._array.shape, (2, 2))
        self.assertEqual(recv_calls, [None])
        self.assertIn((logging.INFO, 'Initializing the arrays in the HDF5 file'),
                      logs)
      finally:
        recorder.finish()

      np.testing.assert_array_equal(self._read_node(path),
                                    np.array([[1.0, 2.0], [3.0, 4.0]]))

  def test_loop_appends_subsequent_batches(self) -> None:
    """Checks that later received stream chunks are appended."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'
      recorder, recv_calls, _ = self._make_recorder(
        path,
        batches=[
          {'stream': [np.array([[1.0, 2.0], [3.0, 4.0]])]},
          {'stream': [np.array([[5.0, 6.0]])]},
        ],
        atom='float64')

      try:
        recorder.loop()
        self.assertEqual(recv_calls, [None])

        recorder.loop()
        self.assertEqual(recv_calls, [None, None])
      finally:
        recorder.finish()

      np.testing.assert_array_equal(
        self._read_node(path),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

  def test_loop_ignores_missing_label_after_initialization(self) -> None:
    """Checks that later batches without the target label are ignored."""

    with TemporaryDirectory() as folder:
      path = Path(folder) / 'data.h5'
      recorder, _, _ = self._make_recorder(
        path,
        batches=[
          {'stream': [np.array([[1, 2]])]},
          {'other': [np.array([[3, 4]])]},
        ])

      try:
        recorder.loop()
        recorder.loop()
      finally:
        recorder.finish()

      np.testing.assert_array_equal(self._read_node(path),
                                    np.array([[1, 2]], dtype=np.int16))

  def test_loop_raises_when_first_batch_has_no_target_label(self) -> None:
    """Checks the explicit error when the first data lacks the target label."""

    with TemporaryDirectory() as folder:
      recorder, _, _ = self._make_recorder(
        Path(folder) / 'data.h5',
        batches=[{'other': [np.array([[1, 2]])]}])

      try:
        with self.assertRaises(KeyError):
          recorder.loop()
        self.assertFalse(recorder._array_initialized)
      finally:
        recorder.finish()

  def test_flush_period_resets_counter_at_requested_interval(self) -> None:
    """Checks periodic flushing across initial and subsequent loops."""

    with TemporaryDirectory() as folder:
      recorder, _, _ = self._make_recorder(
        Path(folder) / 'data.h5',
        batches=[
          {'stream': [np.array([[1, 2]])]},
          {'stream': [np.array([[3, 4]])]},
        ],
        flush_period=2)

      try:
        recorder.loop()
        self.assertEqual(recorder._flush_count, 1)

        recorder.loop()
        self.assertEqual(recorder._flush_count, 0)
      finally:
        recorder.finish()

      recorder, _, _ = self._make_recorder(
        Path(folder) / 'data_2.h5',
        batches=[{'stream': [np.array([[1, 2]])]}],
        flush_period=1)

      try:
        recorder.loop()
        self.assertEqual(recorder._flush_count, 0)
      finally:
        recorder.finish()

  def test_finish_is_idempotent(self) -> None:
    """Checks that finish can be called repeatedly."""

    with TemporaryDirectory() as folder:
      source = TestBlock()
      recorder = HDFRecorder(Path(folder) / 'data.h5', atom='float64')
      link(source, recorder)

      recorder.prepare()
      recorder.finish()
      recorder.finish()

      self.assertIsNone(recorder._hfile)
