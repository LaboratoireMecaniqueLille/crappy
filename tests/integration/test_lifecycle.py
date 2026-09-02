# coding: utf-8

import csv
import json
from math import isfinite

from .integration_test_base import IntegrationTestBase


class TestLifecycleIntegration(IntegrationTestBase):
  """End-to-end tests for control and terminal Blocks."""

  def test_auto_drive_recorder_pipeline(self) -> None:
    """Checks coordinate input drives AutoDrive and records its offset."""

    with self.run_scenario('auto_drive_recorder') as output_dir:
      csv_path = output_dir / 'auto_drive.csv'
      self.assertTrue(csv_path.is_file())

      with csv_path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        self.assertEqual(reader.fieldnames, ['t(s)', 'diff(pix)'])
        rows = list(reader)

      self.assertGreaterEqual(len(rows), 2)
      times = [float(row['t(s)']) for row in rows]
      differences = [float(row['diff(pix)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *differences))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(-15 <= difference <= -10
                          for difference in differences))
      self.assertGreater(differences[-1], differences[0])

  def test_pause_stop_probe_pipeline(self) -> None:
    """Checks scripted pause/resume and stop transitions through a probe."""

    with self.run_scenario('pause_stop_probe') as output_dir:
      probe_path = output_dir / 'pause_probe.json'
      self.assertTrue(probe_path.is_file())

      with probe_path.open(encoding='utf-8') as file:
        probe = json.load(file)

      self.assertTrue(probe['finish_called'])
      loop_times = probe['loop_times']
      self.assertGreaterEqual(len(loop_times), 4)
      self.assertTrue(any(time_value < 0.2 for time_value in loop_times))
      self.assertTrue(any(time_value > 0.5 for time_value in loop_times))
      self.assertGreater(max(second - first for first, second
                             in zip(loop_times, loop_times[1:])),
                         0.15)

  def test_generator_sink_lifecycle(self) -> None:
    """Checks Generator and Sink prepare, run, and stop cleanly."""

    with self.run_scenario('generator_sink'):
      pass

  def test_generator_link_reader_lifecycle(self) -> None:
    """Checks Generator and LinkReader prepare, run, and stop cleanly."""

    with self.run_scenario('generator_link_reader'):
      pass
