# coding: utf-8

import csv
import json
from multiprocessing import get_all_start_methods
from math import isfinite

from tests.support.integration import IntegrationTestBase


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

  def test_pause_stop_probe_pipeline(self):
    """Observe progress, settled pause, resumed progress and normal finish."""

    with self.run_scenario('pause_stop_probe') as output_dir:
      probe = json.loads((output_dir / 'pause_probe.json').read_text())
      controls = json.loads((output_dir / 'pause_controls.json').read_text())
    self.assertTrue(probe['finish_called'])
    self.assertTrue(controls['pause_observed'])
    self.assertGreaterEqual(controls['before_pause'], 2)
    self.assertEqual(controls['paused_start'], controls['paused_end'])
    self.assertGreater(controls['resumed'], controls['paused_end'])
    self.assertGreaterEqual(len(probe['loop_times']), 4)
    self.assertTrue(all(map(isfinite, probe['loop_times'])))
    self.assertEqual(probe['loop_times'], sorted(probe['loop_times']))

  def test_generator_sink_lifecycle(self) -> None:
    """Checks Generator and Sink prepare, run, and stop cleanly."""

    with self.run_scenario('generator_sink'):
      pass

  def test_generator_link_reader_lifecycle(self) -> None:
    """Checks Generator and LinkReader prepare, run, and stop cleanly."""

    with self.run_scenario('generator_link_reader'):
      pass

  def test_repeating_generator_under_available_start_methods(self):
    """A repeating Path is usable after real worker startup in each context."""

    for method in get_all_start_methods():
      with self.subTest(method=method):
        with self.run_scenario('repeating_generator', start_method=method) as output_dir:
          rows = json.loads((output_dir / 'repeat.json').read_text())
        phases = [row[2] for row in rows]
        self.assertTrue({0, 1, 2}.issubset(phases))
        self.assertEqual(phases, sorted(phases))
        self.assertTrue(all(row[1] == 3 for row in rows))
