# coding: utf-8

from unittest.mock import patch

import numpy as np

import crappy.blocks.camera_processes.dic_ve as dic_ve_module
from crappy.blocks.camera_processes.dic_ve import DICVEProcess
from crappy.tool.camera_config import SpotsBoxes

from tests.camera_process.camera_process_test_base import (CameraProcessTestBase,
                                                           TestLink)


class DummyDICVETool:
  """Small stand-in for the DICVE image-processing tool."""

  instances: list["DummyDICVETool"] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor arguments and exposes deterministic outputs."""

    self.kwargs = kwargs
    self.patches = kwargs['patches']
    self.img0 = None
    self.images = list()
    self.return_value = ([(1.0, 2.0)], 3.0, 4.0, [(5.0, 6.0)])
    self.raise_on_calculate = False

    type(self).instances.append(self)

  def set_img0(self, img: np.ndarray) -> None:
    """Records the reference image set by the process."""

    self.img0 = np.copy(img)

  def calculate_displacement(self, img: np.ndarray):
    """Records processed frames and returns deterministic data."""

    self.images.append(np.copy(img))
    if self.raise_on_calculate:
      raise RuntimeError("lost patch")
    return self.return_value


class TestDICVEProcess(CameraProcessTestBase):
  """Unit tests for the DICVE CameraProcess wrapper."""

  @staticmethod
  def _patches() -> SpotsBoxes:
    """Returns deterministic patch boxes for DICVEProcess tests."""

    patches = SpotsBoxes()
    patches.set_spots([(1, 2, 3, 4)])
    patches.save_length()
    return patches

  def setUp(self) -> None:
    """Resets the fake tool registry."""

    DummyDICVETool.instances.clear()

  def test_init_forwards_arguments_to_tool(self) -> None:
    """Checks DICVETool instantiation arguments."""

    patches = self._patches()
    process = DICVEProcess(patches=patches,
                           method='Parabola',
                           alpha=1,
                           delta=2,
                           gamma=3,
                           finest_scale=4,
                           iterations=5,
                           gradient_iterations=6,
                           patch_size=7,
                           patch_stride=8,
                           border=0.1,
                           safe=False,
                           follow=False)

    with patch.object(dic_ve_module, 'DICVETool', DummyDICVETool):
      process.init()

    self.assertIsInstance(process._disve, DummyDICVETool)
    self.assertEqual(len(DummyDICVETool.instances), 1)
    self.assertEqual(DummyDICVETool.instances[0].kwargs, {
      'patches': patches,
      'method': 'Parabola',
      'alpha': 1,
      'delta': 2,
      'gamma': 3,
      'finest_scale': 4,
      'iterations': 5,
      'gradient_iterations': 6,
      'patch_size': 7,
      'patch_stride': 8,
      'border': 0.1,
      'safe': False,
      'follow': False,
    })

  def test_loop_sets_reference_then_sends_data_and_overlay(self) -> None:
    """Checks first-frame setup and data/overlay forwarding."""

    patches = self._patches()
    process = DICVEProcess(patches=patches)
    self._process = process
    self.set_test_logger(process)

    link = TestLink()
    process._outputs = [link]
    process._labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)',
                       'Disp(px)']
    sent_overlays = list()
    process.send_to_draw = sent_overlays.append

    with patch.object(dic_ve_module, 'DICVETool', DummyDICVETool):
      process.init()

    img0 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    process.img = img0
    process.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}

    process.loop()

    tool = DummyDICVETool.instances[0]
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
      'Coord(px)': [(1.0, 2.0)],
      'Eyy(%)': 3.0,
      'Exx(%)': 4.0,
      'Disp(px)': [(5.0, 6.0)],
    })
    self.assertEqual(len(tool.images), 1)
    np.testing.assert_array_equal(tool.images[0], img)
    self.assertEqual(sent_overlays, [patches])

  def test_loop_handles_lost_patch_without_raising(self) -> None:
    """Checks idle behavior after losing patches when configured not to raise."""

    process = DICVEProcess(patches=self._patches(), raise_on_exit=False)
    self._process = process
    self.set_test_logger(process)
    process._outputs = [TestLink()]
    process._labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)',
                       'Disp(px)']

    with patch.object(dic_ve_module, 'DICVETool', DummyDICVETool):
      process.init()

    process.img = np.zeros((3, 4), dtype=np.uint8)
    process.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    process.loop()

    tool = DummyDICVETool.instances[0]
    tool.raise_on_calculate = True
    process.metadata = {'ImageUniqueID': 2, 't(s)': 0.2}

    process.loop()

    self.assertTrue(process._lost_patch)
    self.assertFalse(process._outputs[0].sent.is_set())

    process.fps_count = 5
    with patch.object(dic_ve_module, 'sleep') as sleep:
      process.loop()

    sleep.assert_called_once_with(0.1)
    self.assertEqual(process.fps_count, 4)

  def test_loop_reraises_lost_patch_when_requested(self) -> None:
    """Checks RuntimeError propagation when raise_on_exit is enabled."""

    process = DICVEProcess(patches=self._patches(), raise_on_exit=True)
    self._process = process
    self.set_test_logger(process)

    with patch.object(dic_ve_module, 'DICVETool', DummyDICVETool):
      process.init()

    process.img = np.zeros((3, 4), dtype=np.uint8)
    process.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    process.loop()
    DummyDICVETool.instances[0].raise_on_calculate = True

    with self.assertRaises(RuntimeError):
      process.loop()

    self.assertTrue(process._lost_patch)
