# coding: utf-8

import unittest
import cv2
import numpy as np
from dataclasses import dataclass

from crappy.tool.camera_config.config_tools import SpotsDetector


@dataclass
class Prop:
  """"""

  bbox: tuple[int, int, int, int]


class TestSpotsDetector(unittest.TestCase):
  """"""

  def test_01_overlap_bbox(self) -> None:
    """"""

    prop_ref = Prop((100, 100, 200, 200))
    for x in range(300):
      with self.subTest(x=x):
        prop_test = Prop((x, x, 50 + x, 50 + x))
        if x <= 50 or x >= 200:
          self.assertFalse(SpotsDetector._overlap_bbox(prop_test, prop_ref))
        else:
          self.assertTrue(SpotsDetector._overlap_bbox(prop_test, prop_ref))

        prop_test = Prop((x, 300 - x - 50, 50 + x, 300 - x))
        if x <= 50 or x >= 200:
          self.assertFalse(SpotsDetector._overlap_bbox(prop_test, prop_ref))
        else:
          self.assertTrue(SpotsDetector._overlap_bbox(prop_test, prop_ref))

  def test_02_detect_spot(self) -> None:
    """"""

    # Instantiate spot detector
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=150)

    # Prepare image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (100, 100), 50, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that the spot is correctly detected
    self.assertLessEqual(abs(50 - detector.spots.spot_1.x_start), 3)
    self.assertLessEqual(abs(50 - detector.spots.spot_1.y_start), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.x_end), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.y_end), 3)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.x_centroid), 1.5)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.y_centroid), 1.5)

    # Check that no other spot is detected
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

  def test_03_multiple_spots(self) -> None:
    """"""

    # Instantiate spot detector
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=150)

    # Prepare image for detection
    img = np.full((400, 400), 255, dtype=np.uint8)
    img = cv2.circle(img, (100, 200), 50, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (300, 200), 49, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (200, 100), 48, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (200, 300), 47, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that the first spot is correctly detected
    self.assertLessEqual(abs(50 - detector.spots.spot_1.x_start), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.y_start), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.x_end), 3)
    self.assertLessEqual(abs(250 - detector.spots.spot_1.y_end), 3)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.x_centroid), 1.5)
    self.assertLessEqual(abs(200.0 - detector.spots.spot_1.y_centroid), 1.5)

    # Check that the second spot is correctly detected
    self.assertLessEqual(abs(251 - detector.spots.spot_2.x_start), 3)
    self.assertLessEqual(abs(151 - detector.spots.spot_2.y_start), 3)
    self.assertLessEqual(abs(349 - detector.spots.spot_2.x_end), 3)
    self.assertLessEqual(abs(249 - detector.spots.spot_2.y_end), 3)
    self.assertLessEqual(abs(300.0 - detector.spots.spot_2.x_centroid), 1.5)
    self.assertLessEqual(abs(200.0 - detector.spots.spot_2.y_centroid), 1.5)

    # Check that the third spot is correctly detected
    self.assertLessEqual(abs(152 - detector.spots.spot_3.x_start), 3)
    self.assertLessEqual(abs(52 - detector.spots.spot_3.y_start), 3)
    self.assertLessEqual(abs(248 - detector.spots.spot_3.x_end), 3)
    self.assertLessEqual(abs(148 - detector.spots.spot_3.y_end), 3)
    self.assertLessEqual(abs(200.0 - detector.spots.spot_3.x_centroid), 1.5)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_3.y_centroid), 1.5)

    # Check that the fourth spot is correctly detected
    self.assertLessEqual(abs(153 - detector.spots.spot_4.x_start), 3)
    self.assertLessEqual(abs(253 - detector.spots.spot_4.y_start), 3)
    self.assertLessEqual(abs(247 - detector.spots.spot_4.x_end), 3)
    self.assertLessEqual(abs(347 - detector.spots.spot_4.y_end), 3)
    self.assertLessEqual(abs(200.0 - detector.spots.spot_4.x_centroid), 1.5)
    self.assertLessEqual(abs(300.0 - detector.spots.spot_4.y_centroid), 1.5)

  def test_04_ugly_spot(self) -> None:
    """"""

    # Instantiate spot detector
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=150)

    # Prepare image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (65, 100), 20, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (100, 100), 20, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (135, 100), 20, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that no spot is detected
    self.assertIsNone(detector.spots.spot_1)
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

  def test_05_white_spot(self) -> None:
    """"""

    # Instantiate spot detector
    detector = SpotsDetector(white_spots=True,
                             num_spots=None,
                             min_area=150)

    # Prepare wrong image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (100, 100), 50, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that no spot is detected
    self.assertIsNone(detector.spots.spot_1)
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

    # Prepare correct image for detection
    img = np.full((200, 200), 0, dtype=np.uint8)
    img = cv2.circle(img, (100, 100), 50, (255.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that the spot is correctly detected
    self.assertLessEqual(abs(50 - detector.spots.spot_1.x_start), 3)
    self.assertLessEqual(abs(50 - detector.spots.spot_1.y_start), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.x_end), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.y_end), 3)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.x_centroid), 1.5)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.y_centroid), 1.5)

    # Check that no other spot is detected
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

  def test_06_num_spots(self) -> None:
    """"""

    props = (((100, 200), 50),
             ((300, 200), 49),
             ((200, 100), 48),
             ((200, 300), 47))

    for i in range(5):
      with self.subTest(num_spots=i if i else None):

        # Instantiate spot detector
        detector = SpotsDetector(white_spots=False,
                                 num_spots=i if i else None,
                                 min_area=150)

        # Prepare image for detection
        img = np.full((400, 400), 255, dtype=np.uint8)

        for j, (center, size) in enumerate(props):
          with self.subTest(effective_spots=j + 1):
            # Create one more spot
            img = cv2.circle(img, center, size, (0.0,), -1).astype(np.uint8)

            # Detect spots
            detector.spots.reset()
            detector.detect_spots(img, 0, 0)

            # If no specification, detect as many spots as there are
            if not i:
              self.assertEqual(len(detector.spots), j + 1)
            # If more spots than specified, detect only the main ones
            elif i <= j + 1:
              self.assertEqual(len(detector.spots), i)
            # If not enough, don't detect anything
            else:
              self.assertEqual(len(detector.spots), 0)

  def test_07_min_area(self) -> None:
    """"""

    # Instantiate spot detector with min_area=150
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=150)

    # Prepare image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (100, 100), 50, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that the spot is correctly detected
    self.assertLessEqual(abs(50 - detector.spots.spot_1.x_start), 3)
    self.assertLessEqual(abs(50 - detector.spots.spot_1.y_start), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.x_end), 3)
    self.assertLessEqual(abs(150 - detector.spots.spot_1.y_end), 3)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.x_centroid), 1.5)
    self.assertLessEqual(abs(100.0 - detector.spots.spot_1.y_centroid), 1.5)

    # Check that no other spot is detected
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

    # Instantiate spot detector with min_area=10000
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=10000)

    # Prepare image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (100, 100), 50, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Check that no spot is detected
    self.assertIsNone(detector.spots.spot_1)
    self.assertIsNone(detector.spots.spot_2)
    self.assertIsNone(detector.spots.spot_3)
    self.assertIsNone(detector.spots.spot_4)

  def test_08_overlap(self) -> None:
    """"""

    # Instantiate spot detector with min_area=150
    detector = SpotsDetector(white_spots=False,
                             num_spots=None,
                             min_area=150)

    # Prepare image for detection
    img = np.full((200, 200), 255, dtype=np.uint8)
    img = cv2.circle(img, (50, 50), 69, (0.0,), -1).astype(np.uint8)
    img = cv2.circle(img, (150, 150), 68, (0.0,), -1).astype(np.uint8)

    # Detect spots
    detector.detect_spots(img, 0, 0)

    # Only one spot should be detected because the bboxes overlap
    self.assertEqual(len(detector.spots), 1)
