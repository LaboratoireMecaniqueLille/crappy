# coding: utf-8

import csv
from math import isfinite
from pathlib import Path
import unittest

import numpy as np

try:
  import tables
except (ImportError, ModuleNotFoundError):
  tables = None

from .integration_test_base import IntegrationTestBase


class TestNumericIntegration(IntegrationTestBase):
  """End-to-end tests for Blocks processing regular numeric data."""

  def _read_csv(self,
                output_dir: Path,
                filename: str,
                fieldnames: list[str]) -> list[dict[str, str]]:
    """Reads and minimally validates a CSV artifact."""

    csv_path = output_dir / filename
    self.assertTrue(csv_path.is_file())

    with csv_path.open(newline='', encoding='utf-8') as csv_file:
      reader = csv.DictReader(csv_file)
      self.assertEqual(reader.fieldnames, fieldnames)
      rows = list(reader)

    self.assertGreaterEqual(len(rows), 2)
    return rows

  def test_generator_recorder_pipeline(self) -> None:
    """Runs a real script and validates its generated CSV artifact."""

    with self.run_scenario('generator_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'signal.csv',
                            ['t(s)', 'signal'])

      times = [float(row['t(s)']) for row in rows]
      signals = [float(row['signal']) for row in rows]

      self.assertTrue(all(map(isfinite, times)))
      self.assertTrue(all(map(isfinite, signals)))
      self.assertTrue(all(time_value >= 0 for time_value in times))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(first < second for first, second
                          in zip(signals, signals[1:])))

      # The source ramp is signal = 1 + 2 * t.
      for time_value, signal in zip(times, signals):
        with self.subTest(time=time_value, signal=signal):
          self.assertAlmostEqual(signal, 1 + 2 * time_value, delta=0.02)

  def test_multiplexer_recorder_pipeline(self) -> None:
    """Checks two asynchronous signals are multiplexed and recorded."""

    with self.run_scenario('multiplexer_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'multiplexed.csv',
                            ['t(s)', 'first', 'second'])

      times = [float(row['t(s)']) for row in rows]
      first = [float(row['first']) for row in rows]
      second = [float(row['second']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *first, *second))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(previous < current for previous, current
                          in zip(times, times[1:])))
      for time_value, first_value, second_value in zip(times, first, second):
        with self.subTest(time=time_value):
          self.assertAlmostEqual(first_value,
                                 1 + 2 * time_value,
                                 delta=0.03)
          self.assertAlmostEqual(second_value,
                                 5 - 3 * time_value,
                                 delta=0.03)

  def test_synchronizer_recorder_pipeline(self) -> None:
    """Checks a signal is synchronized to a reference and recorded."""

    with self.run_scenario('synchronizer_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'synchronized.csv',
                            ['t(s)', 'reference', 'signal'])

      times = [float(row['t(s)']) for row in rows]
      references = [float(row['reference']) for row in rows]
      signals = [float(row['signal']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *references, *signals))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(previous < current for previous, current
                          in zip(times, times[1:])))
      for time_value, reference, signal in zip(times, references, signals):
        with self.subTest(time=time_value):
          self.assertAlmostEqual(reference,
                                 10 + 3 * time_value,
                                 delta=0.03)
          self.assertAlmostEqual(signal,
                                 -2 + 4 * time_value,
                                 delta=0.03)

  def test_pid_fake_machine_recorder_pipeline(self) -> None:
    """Checks PID feedback drives a FakeMachine and records its response."""

    with self.run_scenario('pid_fake_machine_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'pid_machine.csv',
                            ['t(s)', 'x(mm)', 'F(N)'])

      times = [float(row['t(s)']) for row in rows]
      positions = [float(row['x(mm)']) for row in rows]
      forces = [float(row['F(N)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *positions, *forces))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(position >= 0 for position in positions))
      self.assertGreater(max(positions), positions[0])
      self.assertGreater(max(forces), forces[0])

  def test_machine_recorder_pipeline(self) -> None:
    """Checks a Generator drives a FakeDCMotor through Machine."""

    with self.run_scenario('machine_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'motor.csv',
                            ['t(s)', 'motor_speed', 'motor_position'])

      times = [float(row['t(s)']) for row in rows]
      speeds = [float(row['motor_speed']) for row in rows]
      positions = [float(row['motor_position']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *speeds, *positions))))
      self.assertEqual(times, sorted(times))
      self.assertGreater(max(speeds), speeds[0])
      self.assertGreater(max(positions), positions[0])

  def test_ioblock_recorder_pipeline(self) -> None:
    """Checks regular FakeInOut acquisitions are recorded as CSV."""

    with self.run_scenario('ioblock_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'memory.csv',
                            ['t(s)', 'memory'])

      times = [float(row['t(s)']) for row in rows]
      memory = [float(row['memory']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *memory))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(time_value >= 0 for time_value in times))
      self.assertTrue(all(0 <= value <= 100 for value in memory))

  @unittest.skipIf(tables is None, 'PyTables is not available')
  def test_stream_hdf_recorder_pipeline(self) -> None:
    """Checks streamed FakeInOut acquisitions are recorded as HDF5."""

    with self.run_scenario('stream_hdf_recorder') as output_dir:
      hdf_path = output_dir / 'stream.h5'
      self.assertTrue(hdf_path.is_file())

      with tables.open_file(str(hdf_path), 'r') as hdf_file:
        stream = hdf_file.get_node('/', 'table').read()

      self.assertEqual(stream.ndim, 2)
      self.assertEqual(stream.shape[1], 1)
      self.assertGreaterEqual(stream.shape[0], 20)
      self.assertTrue(np.isfinite(stream).all())
      self.assertTrue(((stream >= 0) & (stream <= 100)).all())
