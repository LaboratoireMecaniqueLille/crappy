# coding: utf-8

import csv
from math import isfinite
from pathlib import Path
import unittest

import numpy as np

try:
  import cv2  # noqa: F401
except (ImportError, ModuleNotFoundError):
  cv2 = None

from .integration_test_base import IntegrationTestBase
from .scenarios.vision import generate_test_image


class TestVisionIntegration(IntegrationTestBase):
  """End-to-end tests for generated camera images and correlation Blocks."""

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

  def test_camera_image_saver_pipeline(self) -> None:
    """Checks generated Camera frames and metadata are saved as NPY."""

    with self.run_scenario('camera_image_saver') as output_dir:
      image_dir = output_dir / 'images'
      image_paths = sorted(image_dir.glob('*.npy'))
      self.assertGreaterEqual(len(image_paths), 2)

      expected = generate_test_image(0, 0)
      for image_path in image_paths:
        with self.subTest(image=image_path.name):
          image = np.load(image_path)
          self.assertEqual(image.dtype, np.dtype('uint8'))
          np.testing.assert_array_equal(image, expected)

      with (image_dir / 'metadata.csv').open(newline='',
                                              encoding='utf-8') as csv_file:
        metadata = list(csv.DictReader(csv_file))

      self.assertEqual(len(metadata), len(image_paths))
      self.assertTrue(all(float(row['t(s)']) >= 0 for row in metadata))

  @unittest.skipIf(cv2 is None, 'OpenCV is not available')
  def test_dicve_recorder_pipeline(self) -> None:
    """Checks generated strain is processed by DICVE and recorded."""

    with self.run_scenario('dicve_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'dicve.csv',
                            ['t(s)', 'Eyy(%)', 'Exx(%)'])

      times = [float(row['t(s)']) for row in rows]
      eyy = [float(row['Eyy(%)']) for row in rows]
      exx = [float(row['Exx(%)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *eyy, *exx))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(time_value >= 0 for time_value in times))

  @unittest.skipIf(cv2 is None, 'OpenCV is not available')
  def test_dis_correl_recorder_pipeline(self) -> None:
    """Checks generated strain is processed by DISCorrel and recorded."""

    with self.run_scenario('dis_correl_recorder') as output_dir:
      rows = self._read_csv(output_dir,
                            'dis_correl.csv',
                            ['t(s)', 'Exx(%)', 'Eyy(%)'])

      times = [float(row['t(s)']) for row in rows]
      exx = [float(row['Exx(%)']) for row in rows]
      eyy = [float(row['Eyy(%)']) for row in rows]

      self.assertTrue(all(map(isfinite, (*times, *exx, *eyy))))
      self.assertEqual(times, sorted(times))
      self.assertTrue(all(time_value >= 0 for time_value in times))
