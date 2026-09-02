# coding: utf-8

from unittest.mock import patch, sentinel

import numpy as np

from crappy.blocks.camera import Camera
from crappy.blocks.dis_correl import DISCorrel
import crappy.blocks.dis_correl as dis_correl_module
from crappy.tool.camera_config import Box

from .camera_wrapper_test_base import (CameraWrapperTestBase,
                                       RecordingProcess)


class TestDISCorrel(CameraWrapperTestBase):
  """Unit tests for the DISCorrel Block wrapper."""

  def setUp(self) -> None:
    """Resets the process test double before each test."""

    RecordingProcess.reset()

  def test_constructor_sets_default_fields_and_labels(self) -> None:
    """Checks the default field projection and corresponding labels."""

    block = DISCorrel(patch=(1, 2, 3, 4), **self.camera_kwargs())

    self.assertEqual(block._fields, ['x', 'y', 'exx', 'eyy'])
    self.assertEqual(block.labels, [
      't(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)',
    ])
    self.assertEqual(block._patch_int, (1, 2, 3, 4))

  def test_constructor_normalizes_custom_fields_and_residual_label(self
                                                                    ) -> None:
    """Checks strings, arrays, and residual labels are normalized correctly."""

    block = DISCorrel(patch=(0, 0, 2, 2),
                      fields='x',
                      labels=['time', 'metadata', 'x'],
                      residual=True,
                      **self.camera_kwargs())

    self.assertEqual(block._fields, ['x'])
    self.assertEqual(block.labels, ['time', 'metadata', 'x', 'res'])

    field = np.zeros((2, 2, 2), dtype=np.float32)
    block = DISCorrel(patch=(0, 0, 2, 2),
                      fields=field,
                      labels=['time', 'metadata', 'field'],
                      **self.camera_kwargs())

    self.assertEqual(len(block._fields), 1)
    self.assertIs(block._fields[0], field)

  def test_constructor_validates_patch(self) -> None:
    """Checks missing, malformed, and empty ROI declarations."""

    cases = (
      (None, False),
      ((0, 0, 2), False),
      ([0, 0, 2, 2], False),
      ((-1, 0, 2, 2), False),
      ((0, 0, 0, 2), False),
      ((0, 0, 2, 0), False),
    )

    for roi, config in cases:
      with self.subTest(patch=roi, config=config):
        with self.assertRaises(ValueError):
          DISCorrel(patch=roi, **self.camera_kwargs(config=config))

  def test_constructor_validates_custom_fields_and_labels(self) -> None:
    """Checks custom field labeling and label validation."""

    with self.assertRaises(ValueError):
      DISCorrel(patch=(0, 0, 2, 2), fields=['r'],
                **self.camera_kwargs())

    cases = (
      ['too', 'few'],
      ['same'] * 6,
      ['time', 'meta', 'x', 'y', 'exx', 1],
    )

    for labels in cases:
      with self.subTest(labels=labels):
        with self.assertRaises(ValueError):
          DISCorrel(patch=(0, 0, 2, 2), labels=labels,
                    **self.camera_kwargs())

  def test_prepare_builds_box_and_forwards_process_options(self) -> None:
    """Checks ROI conversion and DISCorrelProcess option forwarding."""

    fields = ['r', 'z']
    block = DISCorrel(patch=(1, 2, 3, 4),
                      fields=fields,
                      labels=['time', 'metadata', 'rotation', 'zoom'],
                      alpha=1,
                      delta=2,
                      gamma=3,
                      finest_scale=4,
                      init=False,
                      iterations=5,
                      gradient_iterations=6,
                      patch_size=7,
                      patch_stride=8,
                      residual=True,
                      **self.camera_kwargs())

    with (patch.object(dis_correl_module, 'DISCorrelProcess', RecordingProcess),
          patch.object(Camera, 'prepare') as camera_prepare):
      block.prepare()

    camera_prepare.assert_called_once_with()
    process = RecordingProcess.instances[-1]
    box = process.kwargs.pop('patch')

    self.assertIs(block.process_proc, process)
    self.assertIs(block._patch, box)
    self.assertIsInstance(box, Box)
    self.assertEqual((box.x_start, box.x_end, box.y_start, box.y_end),
                     (2, 6, 1, 4))
    self.assertEqual(process.kwargs, {
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
      'residual': True,
    })

  def test_prepare_allows_gui_to_populate_box(self) -> None:
    """Checks the configuration path starts from an empty Box."""

    block = DISCorrel(patch=None, **self.camera_kwargs(config=True))

    with (patch.object(dis_correl_module, 'DISCorrelProcess', RecordingProcess),
          patch.object(Camera, 'prepare')):
      block.prepare()

    box = RecordingProcess.instances[-1].kwargs['patch']
    self.assertTrue(box.no_points())
    self.assertIs(block._patch, box)

  def test_configure_forwards_camera_and_box(self) -> None:
    """Checks DISCorrelConfig receives the current Camera and ROI."""

    block = DISCorrel(patch=(0, 0, 2, 2),
                      **self.camera_kwargs(config=True))
    block._camera = sentinel.camera
    block._log_queue = sentinel.log_queue
    block._log_level = 30
    block.freq = 123
    block._patch = sentinel.box

    with patch.object(dis_correl_module, 'DISCorrelConfig',
                      return_value=sentinel.config) as config:
      ret = block._configure()

    self.assertIs(ret, sentinel.config)
    config.assert_called_once_with(sentinel.camera,
                                   sentinel.log_queue,
                                   30,
                                   123,
                                   sentinel.box)
