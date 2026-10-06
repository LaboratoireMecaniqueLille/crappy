# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import numpy as np

import crappy.tool.image_processing.dis_correl as dis_correl_module
from crappy.tool.camera_config import Box
from crappy.tool.image_processing.dis_correl import DISCorrelTool


class DummyDIS:
  """Small stand-in for OpenCV's DIS optical-flow object."""

  def __init__(self) -> None:
    self.params = list()
    self.calls = list()
    self.flow_x = 2.0
    self.flow_y = 3.0

  def setVariationalRefinementAlpha(self, value) -> None:
    self.params.append(('alpha', value))

  def setVariationalRefinementDelta(self, value) -> None:
    self.params.append(('delta', value))

  def setVariationalRefinementGamma(self, value) -> None:
    self.params.append(('gamma', value))

  def setFinestScale(self, value) -> None:
    self.params.append(('finest_scale', value))

  def setVariationalRefinementIterations(self, value) -> None:
    self.params.append(('iterations', value))

  def setGradientDescentIterations(self, value) -> None:
    self.params.append(('gradient_iterations', value))

  def setPatchSize(self, value) -> None:
    self.params.append(('patch_size', value))

  def setPatchStride(self, value) -> None:
    self.params.append(('patch_stride', value))

  def calc(self,
           ref: np.ndarray,
           img: np.ndarray,
           flow: np.ndarray | None) -> np.ndarray:
    """Records inputs and returns a deterministic full-frame flow."""

    self.calls.append((ref.shape, img.shape,
                       None if flow is None else flow.shape))
    out = np.zeros((ref.shape[0], ref.shape[1], 2), dtype=np.float32)
    out[:, :, 0] = self.flow_x
    out[:, :, 1] = self.flow_y
    return out


class TestDISCorrelTool(TestCase):
  """Unit tests for the DISCorrel image-processing tool."""

  @staticmethod
  def _box() -> Box:
    """Returns a deterministic ROI box."""

    return Box(x_start=2, x_end=6, y_start=1, y_end=4)

  @staticmethod
  def _make_tool(**kwargs) -> tuple[DISCorrelTool, DummyDIS]:
    """Instantiates DISCorrelTool with a fake DIS object."""

    box = kwargs.pop('box', TestDISCorrelTool._box())
    defaults = {
      'fields': ['x', 'y', 'exx', 'eyy'],
      'alpha': 3,
      'delta': 1,
      'gamma': 0,
      'finest_scale': 1,
      'init': True,
      'iterations': 1,
      'gradient_iterations': 10,
      'patch_size': 8,
      'patch_stride': 3,
      'border': None,
      'follow': False,
    }
    defaults.update(kwargs)
    dummy = DummyDIS()
    with patch.object(dis_correl_module.cv2,
                      'DISOpticalFlow_create',
                      return_value=dummy):
      tool = DISCorrelTool(box, **defaults)
    return tool, dummy

  def test_constructor_applies_dis_parameters(self) -> None:
    """Checks that DIS parameters are forwarded to OpenCV."""

    _, dummy = self._make_tool(alpha=1,
                               delta=2,
                               gamma=3,
                               finest_scale=4,
                               iterations=5,
                               gradient_iterations=6,
                               patch_size=7,
                               patch_stride=8)

    self.assertEqual(dummy.params, [
      ('alpha', 1),
      ('delta', 2),
      ('gamma', 3),
      ('finest_scale', 4),
      ('iterations', 5),
      ('gradient_iterations', 6),
      ('patch_size', 7),
      ('patch_stride', 8),
    ])

  def test_set_img0_and_set_box_initialize_state(self) -> None:
    """Checks reference image and ROI field initialization."""

    user_field = np.ones((3, 4, 2), dtype=np.float32)
    tool, _ = self._make_tool(fields=['x', user_field])

    tool.set_img0(np.zeros((12, 12), dtype=np.uint8))
    tool.set_box()

    self.assertEqual(tool._dis_flow.shape, (12, 12, 2))
    self.assertEqual(len(tool._base), 2)
    self.assertEqual(tool._base[0].shape, (3, 4, 2))
    self.assertEqual(tool._base[1].shape, (3, 4, 2))
    self.assertEqual(tool._norm2, [12.0, 24.0])

  def test_get_data_requires_reference_and_box(self) -> None:
    """Checks setup-order validation."""

    tool, _ = self._make_tool(fields=['x'])
    image = np.zeros((12, 12), dtype=np.uint8)

    with self.assertRaises(ValueError):
      tool.get_data(image)

    tool.set_img0(image)

    with self.assertRaises(ValueError):
      tool.get_data(image)

  def test_get_data_projects_flow_on_fields(self) -> None:
    """Checks projection of the calculated flow on the requested fields."""

    tool, dummy = self._make_tool(fields=['x', 'y'])
    img0 = np.zeros((12, 12), dtype=np.uint8)
    img = np.zeros((12, 12), dtype=np.uint8)
    tool.set_img0(img0)
    tool.set_box()

    ret = tool.get_data(img)

    self.assertEqual(ret, [2.0, 3.0])
    self.assertEqual(dummy.calls, [((12, 12), (12, 12), (12, 12, 2))])

  def test_get_data_can_append_residuals(self) -> None:
    """Checks cropped residual inputs and averaging."""

    box = Box(x_start=4, x_end=10, y_start=3, y_end=9)
    tool, _ = self._make_tool(box=box, fields=['x'], border=(3, 2))
    image = np.zeros((20, 20), dtype=np.uint8)
    tool.set_img0(image)
    tool.set_box()

    with patch.object(dis_correl_module,
                      'get_res',
                      return_value=np.array([[1, -2], [3, -4]])) as get_res:
      self.assertEqual(tool.get_data(image, residuals=True),
                       [2.0, 2.5])
    ref_crop, img_crop, flow = get_res.call_args.args
    self.assertEqual(ref_crop.shape, (10, 12))
    self.assertEqual(img_crop.shape, (10, 12))
    self.assertEqual(flow.shape, (10, 12, 2))

  def test_crop_to_box_uses_correlation_coordinates(self) -> None:
    """Checks the internal ROI crop helper."""

    tool, _ = self._make_tool(fields=['x'])
    tool.set_img0(np.zeros((12, 12), dtype=np.uint8))
    arr = np.arange(144).reshape(12, 12)

    np.testing.assert_array_equal(tool._crop_to_box(arr), arr[1:4, 2:6])

  def test_set_img0_applies_asymmetric_borders(self) -> None:
    """Checks crop geometry for separate horizontal and vertical borders."""

    box = Box(x_start=4, x_end=10, y_start=3, y_end=9)
    tool, _ = self._make_tool(box=box, fields=['x'], border=(3, 2))

    tool.set_img0(np.zeros((20, 20), dtype=np.uint8))

    self.assertEqual(tool._correl_box, (1, 13, 1, 11))
    self.assertEqual(tool._dis_flow.shape, (10, 12, 2))

  def test_set_img0_clips_borders_to_image_bounds(self) -> None:
    """Checks that borders do not extend beyond the reference image."""

    box = Box(x_start=0, x_end=6, y_start=0, y_end=6)
    tool, _ = self._make_tool(box=box, fields=['x'], border=6)

    tool.set_img0(np.zeros((20, 20), dtype=np.uint8))

    self.assertEqual(tool._correl_box, (0, 12, 0, 12))
    self.assertEqual(tool._dis_flow.shape, (12, 12, 2))

  def test_set_img0_rejects_too_small_correlation_crop(self) -> None:
    """Checks that invalid DIS crop dimensions fail before processing."""

    tool, _ = self._make_tool(fields=['x'], border=0)

    with self.assertRaisesRegex(ValueError,
                                r"correlation area \(4x3\) is too small"):
      tool.set_img0(np.zeros((20, 20), dtype=np.uint8))

  def test_get_data_uses_bordered_crops(self) -> None:
    """Checks that DIS receives the configured sub-images and flow."""

    box = Box(x_start=4, x_end=10, y_start=3, y_end=9)
    tool, dummy = self._make_tool(box=box,
                                  fields=['x', 'y'],
                                  border=(3, 2))
    image = np.zeros((20, 20), dtype=np.uint8)
    tool.set_img0(image)
    tool.set_box()

    self.assertEqual(tool.get_data(image), [2.0, 3.0])
    self.assertEqual(dummy.calls, [((10, 12), (10, 12), (10, 12, 2))])

  def test_follow_updates_offset_and_reused_flow(self) -> None:
    """Checks cumulative output and initialization after crop movement."""

    box = Box(x_start=10, x_end=14, y_start=10, y_end=14)
    tool, _ = self._make_tool(box=box,
                              fields=['x', 'y'],
                              border=8,
                              follow=True)
    image = np.zeros((32, 32), dtype=np.uint8)
    tool.set_img0(image)
    tool.set_box()

    self.assertEqual(tool.get_data(image), [2.0, 3.0])
    self.assertEqual(tool.offset, (2, 3))
    np.testing.assert_array_equal(tool._dis_flow,
                                  np.zeros((20, 20, 2), dtype=np.float32))

    self.assertEqual(tool.get_data(image), [4.0, 6.0])
    self.assertEqual(tool.offset, (4, 6))

  def test_follow_keeps_correlation_crop_inside_image(self) -> None:
    """Checks that large measured translations are clamped to image bounds."""

    box = Box(x_start=10, x_end=14, y_start=10, y_end=14)
    tool, dummy = self._make_tool(box=box,
                                  fields=['x', 'y'],
                                  border=8,
                                  follow=True)
    dummy.flow_x = 100
    dummy.flow_y = -100
    image = np.zeros((24, 24), dtype=np.uint8)
    tool.set_img0(image)
    tool.set_box()

    self.assertEqual(tool.get_data(image), [100.0, -100.0])
    self.assertEqual(tool.offset, (2, -2))
