# coding: utf-8

from ast import literal_eval
import csv
import json
from math import isfinite
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import unittest

import numpy as np

try:
  import cv2  # noqa: F401
except (ImportError, ModuleNotFoundError):
  cv2 = None

try:
  import skimage  # noqa: F401
except (ImportError, ModuleNotFoundError):
  skimage = None

from .integration_test_base import IntegrationTestBase
from .scenarios.vision_blocks import generate_vision_test_image


class TestVisionBlocksIntegration(IntegrationTestBase):
  """End-to-end tests for the independent VisionBlock pipeline."""

  def _read_image_recording(
      self,
      output_dir: Path,
      folder_name: str) -> tuple[list[Path], list[dict[str, str]]]:
    """Returns validated NPY paths and corresponding metadata rows."""

    image_dir = output_dir / folder_name
    self.assertTrue(image_dir.is_dir())

    image_paths = sorted(image_dir.glob('*.npy'))
    self.assertGreaterEqual(len(image_paths), 2)

    metadata_path = image_dir / 'metadata.csv'
    self.assertTrue(metadata_path.is_file())
    with metadata_path.open(newline='', encoding='utf-8') as csv_file:
      reader = csv.DictReader(csv_file)
      self.assertIsNotNone(reader.fieldnames)
      self.assertTrue({'t(s)', 'ImageUniqueID'}.issubset(reader.fieldnames))
      rows = list(reader)

    self.assertEqual(len(rows), len(image_paths))
    return image_paths, rows

  def _read_numeric_csv(
      self,
      output_dir: Path,
      filename: str,
      fieldnames: list[str]) -> list[dict[str, str]]:
    """Reads a nonempty result CSV and validates its exact header."""

    csv_path = output_dir / filename
    self.assertTrue(csv_path.is_file())
    with csv_path.open(newline='', encoding='utf-8') as csv_file:
      reader = csv.DictReader(csv_file)
      self.assertEqual(reader.fieldnames, fieldnames)
      rows = list(reader)

    self.assertGreaterEqual(len(rows), 2)
    return rows

  @staticmethod
  def _read_json(path: Path) -> dict:
    """Reads one JSON artifact produced by a scenario Block."""

    with path.open(encoding='utf-8') as file:
      return json.load(file)

  def _assert_shared_memory_unlinked(self, memory_name: str) -> None:
    """Checks a source-owned shared-memory segment no longer exists."""

    try:
      leaked_memory = SharedMemory(name=memory_name, create=False)
    except FileNotFoundError:
      return

    leaked_memory.close()
    self.fail("The source-owned shared-memory segment was not unlinked")

  def test_camera_recorder_fanout_pipeline(self) -> None:
    """Checks public image fan-out, NPY saving, periods, and notifications."""

    with self.run_scenario('vision_camera_recorder_fanout') as output_dir:
      expected_image = generate_vision_test_image(0, 0)
      recordings = {
        'fast': self._read_image_recording(output_dir,
                                           'vision_fast_images'),
        'sparse': self._read_image_recording(output_dir,
                                             'vision_sparse_images'),
      }

      recording_ids = dict()
      for name, (image_paths, rows) in recordings.items():
        with self.subTest(recorder=name):
          for image_path in image_paths:
            image = np.load(image_path)
            self.assertEqual(image.shape, (48, 64))
            self.assertEqual(image.dtype, np.dtype('uint8'))
            np.testing.assert_array_equal(image, expected_image)

          timestamps = [float(row['t(s)']) for row in rows]
          image_ids = [int(row['ImageUniqueID']) for row in rows]
          filename_ids = [int(path.stem.split('_', 1)[0])
                          for path in image_paths]

          self.assertTrue(all(map(isfinite, timestamps)))
          self.assertTrue(all(timestamp >= 0 for timestamp in timestamps))
          self.assertEqual(timestamps, sorted(timestamps))
          self.assertTrue(all(first < second for first, second
                              in zip(image_ids, image_ids[1:])))
          self.assertEqual(filename_ids, image_ids)
          recording_ids[name] = image_ids

      sparse_ids = recording_ids['sparse']
      self.assertTrue(all(second - first >= 3 for first, second
                          in zip(sparse_ids, sparse_ids[1:])))

      notification_path = output_dir / 'saved_notifications.csv'
      self.assertTrue(notification_path.is_file())
      with notification_path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        self.assertEqual(reader.fieldnames, ['t(s)', 'img_index'])
        notifications = list(reader)

      self.assertGreaterEqual(len(notifications), 1)
      notification_ids = [int(row['img_index']) for row in notifications]
      self.assertTrue(all(image_id in sparse_ids
                          for image_id in notification_ids))
      self.assertTrue(all(first < second for first, second
                          in zip(notification_ids, notification_ids[1:])))

  def test_broken_required_configuration_exits_promptly(self) -> None:
    """Checks source failure unblocks a consumer waiting for configuration."""

    self.assert_scenario_fails(
      'vision_broken_required_config',
      expected_error='Deliberate required configuration source failure',
      timeout=10)

  def test_required_configuration_and_shared_memory_fanout(self) -> None:
    """Checks required config Pipes, coherent frames, and buffer unlinking."""

    with self.run_scenario('vision_required_config_fanout') as output_dir:
      source = self._read_json(output_dir / 'config_source.json')
      probes = {
        'token-a': self._read_json(output_dir / 'config_probe_a.json'),
        'token-b': self._read_json(output_dir / 'config_probe_b.json'),
      }

      self.assertCountEqual(source['answered_tokens'], probes)
      self.assertEqual(source['shape'], [12, 16])
      self.assertEqual(source['dtype'], 'uint16')
      self.assertIsInstance(source['memory_name'], str)
      self.assertTrue(source['memory_name'])

      for token, probe in probes.items():
        with self.subTest(probe=token):
          self.assertEqual(len(probe['configs']), 1)
          config, = probe['configs'].values()
          self.assertEqual(config, [token, [12, 16], 'uint16'])

          observations = probe['observations']
          self.assertGreaterEqual(len(observations), 1)
          transport_ids = [entry['transport_id'] for entry in observations]
          image_ids = [entry['image_id'] for entry in observations]
          self.assertTrue(all(first < second for first, second
                              in zip(transport_ids, transport_ids[1:])))
          self.assertTrue(all(first < second for first, second
                              in zip(image_ids, image_ids[1:])))

          for entry in observations:
            sequence = entry['sequence']
            self.assertEqual(entry['image_id'], sequence)
            self.assertEqual(entry['shape'], [12, 16])
            self.assertEqual(entry['dtype'], 'uint16')
            self.assertEqual(entry['first_pixel'], sequence)
            self.assertEqual(entry['checksum'], sequence * 12 * 16)
            self.assertTrue(isfinite(entry['timestamp']))
            self.assertGreaterEqual(entry['timestamp'], 0)

      self._assert_shared_memory_unlinked(source['memory_name'])

  @unittest.skipIf(cv2 is None, 'OpenCV is not available')
  def test_dicve_processor_pipeline(self) -> None:
    """Checks the separated DICVE pipeline and optional config response."""

    with self.run_scenario('vision_dicve_recorder') as output_dir:
      rows = self._read_numeric_csv(
        output_dir,
        'vision_dicve.csv',
        ['t(s)', 'Eyy(%)', 'Exx(%)'])

      times = [float(row['t(s)']) for row in rows]
      eyy = [float(row['Eyy(%)']) for row in rows]
      exx = [float(row['Exx(%)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *eyy, *exx))))
      self.assertTrue(all(time_value >= 0 for time_value in times))
      self.assertEqual(times, sorted(times))

  @unittest.skipIf(cv2 is None, 'OpenCV is not available')
  def test_dis_correl_processor_pipeline(self) -> None:
    """Checks the separated DISCorrel pipeline and optional config response."""

    with self.run_scenario('vision_dis_correl_recorder') as output_dir:
      rows = self._read_numeric_csv(
        output_dir,
        'vision_dis_correl.csv',
        ['t(s)', 'Exx(%)', 'Eyy(%)'])

      times = [float(row['t(s)']) for row in rows]
      exx = [float(row['Exx(%)']) for row in rows]
      eyy = [float(row['Eyy(%)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *exx, *eyy))))
      self.assertTrue(all(time_value >= 0 for time_value in times))
      self.assertEqual(times, sorted(times))

  @unittest.skipIf(cv2 is None or skimage is None,
                   'OpenCV and scikit-image are required')
  def test_video_extenso_processor_pipeline(self) -> None:
    """Checks required spot config and real tracker process lifecycle."""

    with self.run_scenario('vision_video_extenso_recorder',
                           timeout=25) as output_dir:
      source = self._read_json(output_dir / 'video_extenso_source.json')
      self.assertEqual(source['requesters'], ['Video extenso processor'])
      self.assertEqual(source['shape'], [96, 96])
      self.assertEqual(source['dtype'], 'uint8')

      rows = self._read_numeric_csv(
        output_dir,
        'vision_video_extenso.csv',
        ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)'])

      times = [float(row['t(s)']) for row in rows]
      eyy = [float(row['Eyy(%)']) for row in rows]
      exx = [float(row['Exx(%)']) for row in rows]
      coordinates = [literal_eval(row['Coord(px)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *eyy, *exx))))
      self.assertTrue(all(time_value >= 0 for time_value in times))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(abs(value) < 2 for value in (*eyy, *exx)))

      for centers in coordinates:
        self.assertEqual(len(centers), 2)
        for center in centers:
          self.assertEqual(len(center), 2)
          self.assertTrue(all(isfinite(value) for value in center))
          self.assertTrue(all(0 <= value < 96 for value in center))

      self._assert_shared_memory_unlinked(source['memory_name'])
