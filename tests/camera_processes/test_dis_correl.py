# coding: utf-8

from unittest.mock import patch

import numpy as np

import crappy.blocks.camera_processes.dis_correl as dis_correl_module
from crappy.blocks.camera_processes.dis_correl import DISCorrelProcess
from crappy.tool.camera_config import Box, SpotsBoxes

from tests.camera_process.camera_process_test_base import (CameraProcessTestBase,
                                                           TestLink)


class DummyDISCorrelTool:
  """Small stand-in for DISCorrelTool."""

  instances: list["DummyDISCorrelTool"] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor arguments and exposes deterministic outputs."""

    self.kwargs = kwargs
    self.box = kwargs['box']
    self.set_box_called = False
    self.img0 = None
    self.calls = list()
    self.return_value = [10.0, 20.0, 30.0]

    type(self).instances.append(self)

  def set_box(self) -> None:
    """Records set_box calls."""

    self.set_box_called = True

  def set_img0(self, img: np.ndarray) -> None:
    """Records the reference image set by the process."""

    self.img0 = np.copy(img)

  def get_data(self, img: np.ndarray, residual: bool) -> list[float]:
    """Records processed frames and returns deterministic data."""

    self.calls.append((np.copy(img), residual))
    return self.return_value


class TestDISCorrelProcess(CameraProcessTestBase):
  """Unit tests for the DISCorrel CameraProcess wrapper."""

  @staticmethod
  def _box() -> Box:
    """Returns a deterministic ROI box."""

    return Box(x_start=2, x_end=6, y_start=1, y_end=4)

  def setUp(self) -> None:
    """Resets the fake tool registry."""

    DummyDISCorrelTool.instances.clear()

  def test_init_forwards_arguments_and_sets_box(self) -> None:
    """Checks DISCorrelTool instantiation and box preparation."""

    box = self._box()
    fields = ['x', 'y']
    process = DISCorrelProcess(patch=box,
                               fields=fields,
                               alpha=1,
                               delta=2,
                               gamma=3,
                               finest_scale=4,
                               init=False,
                               iterations=5,
                               gradient_iterations=6,
                               patch_size=7,
                               patch_stride=8,
                               residual=True)

    with patch.object(dis_correl_module, 'DISCorrelTool',
                      DummyDISCorrelTool):
      process.init()

    tool = DummyDISCorrelTool.instances[0]
    self.assertIs(process._dis_correl, tool)
    self.assertTrue(tool.set_box_called)
    self.assertEqual(tool.kwargs, {
      'box': box,
      'fields': fields,
      'alpha': 1,
      'delta': 2,
      'gamma': 3,
      'finest_scale': 4,
      'init': False,
      'iterations': 5,
      'gradient_iterations': 6,
      'patch_size': 7,
      'patch_stride': 8,
    })

  def test_loop_sets_reference_then_sends_data_and_overlay(self) -> None:
    """Checks first-frame setup and data/overlay forwarding."""

    box = self._box()
    process = DISCorrelProcess(patch=box,
                               fields=['x', 'y', 'res'],
                               residual=True)
    self._process = process
    self.set_test_logger(process)

    link = TestLink()
    process._outputs = [link]
    process._labels = ['t(s)', 'meta', 'x(pix)', 'y(pix)', 'res']
    sent_overlays = list()
    process.send_to_draw = sent_overlays.append

    with patch.object(dis_correl_module, 'DISCorrelTool',
                      DummyDISCorrelTool):
      process.init()

    img0 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    process.img = img0
    process.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}

    process.loop()

    tool = DummyDISCorrelTool.instances[0]
    self.assertTrue(process._img0_set)
    np.testing.assert_array_equal(tool.img0, img0)
    self.assertFalse(link.sent.is_set())
    self.assertEqual(sent_overlays, list())

    img = img0 + 10
    metadata = {'ImageUniqueID': 2, 't(s)': 0.2}
    process.img = img
    process.metadata = metadata

    process.loop()

    self.assertTrue(link.sent.is_set())
    self.assertEqual(link.sent_values[-1], {
      't(s)': 0.2,
      'meta': metadata,
      'x(pix)': 10.0,
      'y(pix)': 20.0,
      'res': 30.0,
    })
    self.assertEqual(len(tool.calls), 1)
    np.testing.assert_array_equal(tool.calls[0][0], img)
    self.assertTrue(tool.calls[0][1])
    self.assertEqual(len(sent_overlays), 1)
    self.assertIsInstance(sent_overlays[0], SpotsBoxes)
    self.assertIs(sent_overlays[0].spot_1, box)
