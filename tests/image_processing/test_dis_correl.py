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

    dummy = DummyDIS()
    with patch.object(dis_correl_module.cv2,
                      'DISOpticalFlow_create',
                      return_value=dummy):
      tool = DISCorrelTool(TestDISCorrelTool._box(), **kwargs)
    return tool, dummy

  def test_constructor_validates_fields(self) -> None:
    """Checks accepted and rejected field declarations."""

    with self.assertRaises(TypeError):
      self._make_tool(fields=['x', object()])

    with self.assertRaises(ValueError):
      self._make_tool(fields=['missing'])

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

    tool.set_img0(np.zeros((8, 10), dtype=np.uint8))
    tool.set_box()

    self.assertEqual(tool._dis_flow.shape, (8, 10, 2))
    self.assertEqual(len(tool._base), 2)
    self.assertEqual(tool._base[0].shape, (3, 4, 2))
    self.assertEqual(tool._base[1].shape, (3, 4, 2))
    self.assertEqual(tool._norm2, [12.0, 24.0])

  def test_get_data_requires_reference_and_box(self) -> None:
    """Checks setup-order validation."""

    tool, _ = self._make_tool(fields=['x'])

    with self.assertRaises(ValueError):
      tool.get_data(np.zeros((8, 10), dtype=np.uint8))

    tool.set_img0(np.zeros((8, 10), dtype=np.uint8))

    with self.assertRaises(ValueError):
      tool.get_data(np.zeros((8, 10), dtype=np.uint8))

  def test_get_data_projects_flow_on_fields(self) -> None:
    """Checks projection of the calculated flow on the requested fields."""

    tool, dummy = self._make_tool(fields=['x', 'y'])
    img0 = np.zeros((8, 10), dtype=np.uint8)
    img = np.zeros((8, 10), dtype=np.uint8)
    tool.set_img0(img0)
    tool.set_box()

    ret = tool.get_data(img)

    self.assertEqual(ret, [2.0, 3.0])
    self.assertEqual(dummy.calls, [((8, 10), (8, 10), (8, 10, 2))])

  def test_get_data_can_append_residuals(self) -> None:
    """Checks residual forwarding and averaging."""

    tool, _ = self._make_tool(fields=['x'])
    tool.set_img0(np.zeros((8, 10), dtype=np.uint8))
    tool.set_box()

    with patch.object(dis_correl_module,
                      'get_res',
                      return_value=np.array([[1, -2], [3, -4]])):
      self.assertEqual(tool.get_data(np.zeros((8, 10), dtype=np.uint8),
                                     residuals=True),
                       [2.0, 2.5])

  def test_crop_uses_configured_box(self) -> None:
    """Checks the internal ROI crop helper."""

    tool, _ = self._make_tool(fields=['x'])
    arr = np.arange(80).reshape(8, 10)

    np.testing.assert_array_equal(tool._crop(arr), arr[1:4, 2:6])
