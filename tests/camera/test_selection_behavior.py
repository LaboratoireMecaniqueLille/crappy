# coding: utf-8

"""Selection semantics can be exercised without constructing a Tk window."""

import unittest

import numpy as np

from crappy.tool.camera_config.config_tools import Box, SpotsBoxes, Zoom
from crappy.tool.camera_config.display_state import DisplayGeometry
from crappy.tool.camera_config.selection_behavior import (
  BoxSelectionBehavior, DISCorrelBehavior, DICVEBehavior, VideoExtensoBehavior)


class _Host(BoxSelectionBehavior):
  def __init__(self) -> None:
    super().__init__()
    self._img = np.zeros((100, 100), dtype=np.uint8)
    self._original_img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    self._zoom_values = Zoom()
    self._display_geometry = DisplayGeometry(width=100, height=100,
                                              image_width=100,
                                              image_height=100)

  def _is_on_image(self, x: int, y: int) -> bool:
    return 0 <= x < 100 and 0 <= y < 100

  def log(self, level: int, message: str) -> None:
    pass


class _DIS(DISCorrelBehavior, _Host):
  def __init__(self) -> None:
    super().__init__()
    self._correl_box = Box(10, 30, 10, 30)
    self._draw_correl_box = True


class _DIC(DICVEBehavior, _Host):
  def __init__(self) -> None:
    super().__init__()
    self._create_local_settings()
    self._patch_size.value = 10


class _Detector:
  def __init__(self) -> None:
    self.spots = SpotsBoxes()
    self.thresh = 47
    self.crop = None
    self.origin = None

  def detect_spots(self, crop: np.ndarray, y: int, x: int) -> None:
    self.crop = crop.copy()
    self.origin = (y, x)
    self.spots.spot_1 = Box(x, x + 3, y, y + 3)


class _VE(VideoExtensoBehavior, _Host):
  def __init__(self) -> None:
    super().__init__()
    self._detector = _Detector()
    self._spots = self._detector.spots


class TestSelectionBehavior(unittest.TestCase):
  def test_selection_coordinates_follow_zoom(self) -> None:
    config = _DIS()
    config._zoom_values = Zoom(.25, .75, .25, .75)
    config._start_box_at(20, 40)
    self.assertEqual((config._select_box.x_start,
                      config._select_box.y_start), (35, 45))

  def test_dis_selection_cancel_commit_and_image_invalidation(self) -> None:
    config = _DIS()
    original = config.box
    config._start_box_at(20, 20)
    self.assertFalse(config._draw_correl_box)
    config._extend_box_to(20, 40)
    config._complete_box_selection()
    self.assertEqual(original.sorted(), (10, 30, 10, 30))
    self.assertTrue(config._draw_correl_box)
    self.assertTrue(config._select_box.no_points())

    config._start_box_at(40, 50)
    config._extend_box_to(80, 90)
    config._complete_box_selection()
    self.assertIs(config.get_config()[0], original)
    self.assertEqual(original.sorted(), (40, 80, 50, 90))

    config._start_box_at(60, 60)
    config._start_box_at(-1, -1)
    self.assertTrue(config._draw_correl_box)
    self.assertTrue(config._select_box.no_points())

    config._img = np.zeros((60, 60), dtype=np.uint8)
    config._draw_overlay()
    self.assertTrue(original.no_points())
    self.assertIsNotNone(config._validate_close())

  def test_dic_patch_size_drag_release_and_close(self) -> None:
    config = _DIC()
    config._start_box_at(10, 10)
    config._extend_box_to(20, 20)
    self.assertTrue(config._spots.empty())
    config._extend_box_to(80, 80)
    self.assertEqual(config._spots.spot_1.sorted(), (10, 20, 40, 50))
    config._complete_box_selection()
    self.assertTrue(config._select_box.no_points())
    self.assertIsNone(config._validate_close())
    config._on_valid_close()
    self.assertIsNotNone(config._spots.x_l0)
    self.assertIs(config.get_config()[0], config._spots)

    config._img = np.zeros((50, 50), dtype=np.uint8)
    config._draw_overlay()
    self.assertTrue(config._spots.empty())

  def test_video_crop_action_and_validation(self) -> None:
    config = _VE()
    self.assertIsNotNone(config._validate_close())
    action, = config._extra_actions()
    self.assertEqual((action.id, action.label), ("save_l0", "Save L0"))
    action.callback()
    self.assertIsNone(config._spots.x_l0)

    config._start_box_at(20, 30)
    config._extend_box_to(60, 70)
    config._complete_box_selection()
    self.assertTrue(config._select_box.no_points())
    np.testing.assert_array_equal(config._detector.crop,
                                  config._original_img[30:70, 20:60])
    self.assertEqual(config._detector.origin, (30, 20))
    self.assertIsNone(config._validate_close())
    action.callback()
    self.assertEqual(config.get_config(), (config._spots, 47))
    self.assertIsNotNone(config._spots.x_l0)

    config._img = np.zeros((25, 25), dtype=np.uint8)
    config._draw_overlay()
    self.assertTrue(config._spots.empty())
    self.assertEqual(config.get_config()[1], 47)


if __name__ == '__main__':
  unittest.main()
