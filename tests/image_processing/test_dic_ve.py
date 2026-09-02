# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import numpy as np

import crappy.tool.image_processing.dic_ve as dic_ve_module
from crappy.tool.camera_config import Box, SpotsBoxes
from crappy.tool.image_processing.dic_ve import DICVETool


class TestDICVETool(TestCase):
  """Unit tests for the DICVE image-processing tool."""

  @staticmethod
  def _patches(*patches: tuple[int, int, int, int]) -> SpotsBoxes:
    """Creates SpotsBoxes from (y, x, h, w) patch declarations."""

    spots = SpotsBoxes()
    spots.set_spots(list(patches))
    spots.save_length()
    return spots

  def test_constructor_validates_method_and_border(self) -> None:
    """Checks early validation of DICVE settings."""

    patches = self._patches((10, 10, 10, 10))

    with self.assertRaises(ValueError):
      DICVETool(patches, method='Bad method')

    with self.assertRaises(ValueError):
      DICVETool(patches, method='Pixel precision', border=-0.1)

    with self.assertRaises(ValueError):
      DICVETool(patches, method='Pixel precision', border=1.1)

  def test_set_img0_checks_patch_bounds_in_safe_mode(self) -> None:
    """Checks safe-mode patch boundary validation."""

    patches = self._patches((10, 10, 10, 10))
    tool = DICVETool(patches, method='Pixel precision', safe=True)

    tool.set_img0(np.zeros((30, 30), dtype=np.uint8))

    patches = self._patches((25, 10, 10, 10))
    tool = DICVETool(patches, method='Pixel precision', safe=True)

    with self.assertRaises(RuntimeError):
      tool.set_img0(np.zeros((30, 30), dtype=np.uint8))

  def test_calculate_displacement_requires_reference_image(self) -> None:
    """Checks setup-order validation."""

    tool = DICVETool(self._patches((10, 10, 10, 10)),
                     method='Pixel precision')

    with self.assertRaises(ValueError):
      tool.calculate_displacement(np.zeros((30, 30), dtype=np.uint8))

  def test_calculate_displacement_without_following(self) -> None:
    """Checks strain and displacement output with fixed patches."""

    patches = self._patches((10, 10, 10, 10), (10, 30, 10, 10))
    tool = DICVETool(patches, method='Pixel precision', follow=False)
    tool.set_img0(np.zeros((50, 60), dtype=np.uint8))

    with patch.object(tool,
                      '_calc_pixel_precision',
                      side_effect=([2.0, 0.0], [6.0, 0.0])):
      centers, eyy, exx, disps = tool.calculate_displacement(
        np.zeros((50, 60), dtype=np.uint8))

    self.assertEqual(centers, [(15.0, 15.0), (15.0, 35.0)])
    self.assertEqual(eyy, 0)
    self.assertEqual(exx, 20.0)
    self.assertEqual(disps, [(0.0, 2.0), (0.0, 6.0)])
    self.assertEqual(patches.spot_1.x_start, 10)
    self.assertEqual(patches.spot_2.x_start, 30)

  def test_calculate_displacement_with_following_updates_patch(self) -> None:
    """Checks patch following and cumulative offset handling."""

    patches = self._patches((20, 10, 10, 10))
    tool = DICVETool(patches, method='Pixel precision', follow=True)
    tool.set_img0(np.zeros((50, 60), dtype=np.uint8))

    with patch.object(tool, '_calc_pixel_precision',
                      return_value=[2.2, -1.6]):
      centers, eyy, exx, disps = tool.calculate_displacement(
        np.zeros((50, 60), dtype=np.uint8))

    self.assertEqual((patches.spot_1.x_start, patches.spot_1.x_end), (12, 22))
    self.assertEqual((patches.spot_1.y_start, patches.spot_1.y_end), (18, 28))
    self.assertEqual(centers, [(23.0, 17.0)])
    self.assertEqual((eyy, exx), (0, 0))
    self.assertEqual(disps, [(-1.6, 2.2)])
    self.assertEqual(tool._offsets[0], (-2, 2))

  def test_calc_disflow_averages_trimmed_patch(self) -> None:
    """Checks DISFlow displacement extraction from a patch."""

    patches = self._patches((10, 10, 10, 10))
    tool = DICVETool(patches,
                     method='Disflow',
                     border=0,
                     safe=False)
    tool.set_img0(np.zeros((30, 30), dtype=np.uint8))
    flow = np.zeros((10, 10, 2), dtype=np.float32)
    flow[:, :, 0] = 4
    flow[:, :, 1] = 5

    class DummyDIS:
      def calc(self, *_):
        return flow

    tool._dis = DummyDIS()

    self.assertEqual(tool._calc_disflow(patches.spot_1,
                                        np.zeros((30, 30), dtype=np.uint8),
                                        (0, 0)),
                     [4.0, 5.0])

  def test_parabola_fit_handles_degenerate_inputs(self) -> None:
    """Checks robust parabola refinement fallback cases."""

    self.assertEqual(DICVETool._parabola_fit(np.array([1.0, 2.0, 1.0])), 0.0)
    self.assertEqual(DICVETool._parabola_fit(np.array([1.0, 1.0, 1.0])), 0.0)
    self.assertEqual(DICVETool._parabola_fit(np.array([1.0, 2.0])), 0.0)

  def test_calc_parabola_handles_edge_peak_without_refinement(self) -> None:
    """Checks edge maxima fall back to pixel precision on that axis."""

    patches = self._patches((0, 0, 5, 5))
    tool = DICVETool(patches, method='Parabola', safe=False)
    tool.set_img0(np.zeros((8, 8), dtype=np.uint8))
    cross_correl = np.ones((5, 5), dtype=np.float32)

    with patch.object(tool,
                      '_cross_correlation',
                      return_value=(cross_correl, 0, 2)):
      self.assertEqual(tool._calc_parabola(patches.spot_1,
                                           np.zeros((8, 8), dtype=np.uint8),
                                           (0, 0)),
                       [2.5, 0.5])

  def test_lucas_kanade_uses_status(self) -> None:
    """Checks Lucas-Kanade success and failure handling."""

    patches = self._patches((10, 20, 10, 10))
    tool = DICVETool(patches, method='Lucas Kanade')
    tool.set_img0(np.zeros((40, 50), dtype=np.uint8))
    img = np.zeros((40, 50), dtype=np.uint8)

    with patch.object(dic_ve_module.cv2,
                      'calcOpticalFlowPyrLK',
                      return_value=(np.array([[[7.0, 8.0]]], dtype=np.float32),
                                    np.array([[1]], dtype=np.uint8),
                                    None)):
      self.assertEqual(tool._calc_lucas_kanade(patches.spot_1, img, (0, 0)),
                       [2.0, 3.0])

    with patch.object(dic_ve_module.cv2,
                      'calcOpticalFlowPyrLK',
                      return_value=(np.array([[[7.0, 8.0]]], dtype=np.float32),
                                    np.array([[0]], dtype=np.uint8),
                                    None)):
      with self.assertRaises(RuntimeError):
        tool._calc_lucas_kanade(patches.spot_1, img, (0, 0))

  def test_get_patch_applies_offsets(self) -> None:
    """Checks patch extraction with the tracked offset convention."""

    arr = np.arange(100).reshape(10, 10)
    patch = Box(x_start=4, x_end=8, y_start=3, y_end=7)

    np.testing.assert_array_equal(DICVETool._get_patch(arr, patch, (1, 2)),
                                  arr[2:6, 2:6])
