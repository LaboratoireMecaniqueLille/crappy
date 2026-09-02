# coding: utf-8

from unittest.mock import patch, sentinel

from crappy.blocks.camera import Camera
from crappy.blocks.dic_ve import DICVE
import crappy.blocks.dic_ve as dic_ve_module
from crappy.tool.camera_config import SpotsBoxes

from .camera_wrapper_test_base import (CameraWrapperTestBase,
                                       RecordingProcess)


class TestDICVE(CameraWrapperTestBase):
  """Unit tests for the DICVE Block wrapper."""

  def setUp(self) -> None:
    """Resets the process test double before each test."""

    RecordingProcess.reset()

  def test_constructor_sets_defaults_and_custom_labels(self) -> None:
    """Checks DICVE defaults and supported label normalization."""

    block = DICVE(patches=[(1, 2, 3, 4)],
                  **self.camera_kwargs())

    self.assertEqual(block.labels, [
      't(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'Disp(px)',
    ])
    self.assertEqual(block._patches_int, [(1, 2, 3, 4)])

    labels = ('time', 'metadata', 'coords', 'eyy', 'exx', 'displacements')
    block = DICVE(patches=[(1, 2, 3, 4)], labels=labels,
                  **self.camera_kwargs())

    self.assertEqual(block.labels, list(labels))

  def test_constructor_validates_patches(self) -> None:
    """Checks missing, malformed, and unsupported patch declarations."""

    cases = (
      (None, False),
      ([], False),
      ([(0, 0, 2, 2)] * 5, True),
      ([(0, 0, 2)], False),
      ([[0, 0, 2, 2]], False),
      ([(-1, 0, 2, 2)], False),
      ([(0, 0, 0, 2)], False),
      ([(0, 0, 2, 0)], False),
    )

    for patches, config in cases:
      with self.subTest(patches=patches, config=config):
        with self.assertRaises(ValueError):
          DICVE(patches=patches, **self.camera_kwargs(config=config))

  def test_constructor_validates_labels(self) -> None:
    """Checks DICVE label count, type, and uniqueness validation."""

    cases = (
      ['too', 'few'],
      ['same'] * 6,
      ['time', 'meta', 'coords', 'eyy', 'exx', 1],
    )

    for labels in cases:
      with self.subTest(labels=labels):
        with self.assertRaises(ValueError):
          DICVE(patches=[(0, 0, 2, 2)], labels=labels,
                **self.camera_kwargs())

  def test_prepare_builds_patches_and_forwards_process_options(self) -> None:
    """Checks ROI conversion and DICVEProcess option forwarding."""

    block = DICVE(patches=[(1, 2, 3, 4), (5, 8, 3, 2)],
                  method='Parabola',
                  alpha=1,
                  delta=2,
                  gamma=3,
                  finest_scale=4,
                  iterations=5,
                  gradient_iterations=6,
                  patch_size=7,
                  patch_stride=8,
                  border=0.3,
                  safe=False,
                  follow=False,
                  raise_on_patch_exit=False,
                  **self.camera_kwargs())

    with (patch.object(dic_ve_module, 'DICVEProcess', RecordingProcess),
          patch.object(Camera, 'prepare') as camera_prepare):
      block.prepare()

    camera_prepare.assert_called_once_with()
    process = RecordingProcess.instances[-1]
    patches = process.kwargs.pop('patches')

    self.assertIs(block.process_proc, process)
    self.assertIs(block._patches, patches)
    self.assertIsInstance(patches, SpotsBoxes)
    self.assertEqual(len(patches), 2)
    self.assertEqual((patches.spot_1.x_start, patches.spot_1.x_end,
                      patches.spot_1.y_start, patches.spot_1.y_end),
                     (2, 6, 1, 4))
    self.assertEqual((patches.spot_2.x_start, patches.spot_2.x_end,
                      patches.spot_2.y_start, patches.spot_2.y_end),
                     (8, 10, 5, 8))
    self.assertEqual((patches.x_l0, patches.y_l0), (5, 4))
    self.assertEqual(process.kwargs, {
      'method': 'Parabola',
      'alpha': 1,
      'delta': 2,
      'gamma': 3,
      'finest_scale': 4,
      'iterations': 5,
      'gradient_iterations': 6,
      'patch_size': 7,
      'patch_stride': 8,
      'border': 0.3,
      'safe': False,
      'follow': False,
      'raise_on_exit': False,
    })

  def test_prepare_allows_gui_to_populate_patches(self) -> None:
    """Checks the configuration path starts from an empty SpotsBoxes."""

    block = DICVE(patches=None, **self.camera_kwargs(config=True))

    with (patch.object(dic_ve_module, 'DICVEProcess', RecordingProcess),
          patch.object(Camera, 'prepare')):
      block.prepare()

    patches = RecordingProcess.instances[-1].kwargs['patches']
    self.assertTrue(patches.empty())
    self.assertIs(block._patches, patches)

  def test_configure_forwards_camera_and_patches(self) -> None:
    """Checks DICVEConfig receives the current Camera and patches."""

    block = DICVE(patches=[(0, 0, 2, 2)],
                  **self.camera_kwargs(config=True))
    block._camera = sentinel.camera
    block._log_queue = sentinel.log_queue
    block._log_level = 30
    block.freq = 123
    block._patches = sentinel.patches

    with patch.object(dic_ve_module, 'DICVEConfig',
                      return_value=sentinel.config) as config:
      ret = block._configure()

    self.assertIs(ret, sentinel.config)
    config.assert_called_once_with(sentinel.camera,
                                   sentinel.log_queue,
                                   30,
                                   123,
                                   sentinel.patches)
