# coding: utf-8

from unittest.mock import patch, sentinel

from crappy.blocks.camera import Camera
from crappy.blocks.video_extenso import VideoExtenso
import crappy.blocks.video_extenso as video_extenso_module

from .camera_wrapper_test_base import (CameraWrapperTestBase,
                                       RecordingProcess)


class TestVideoExtenso(CameraWrapperTestBase):
  """Unit tests for the VideoExtenso Block wrapper."""

  def setUp(self) -> None:
    """Resets wrapper test doubles before each test."""

    RecordingProcess.reset()

  def test_constructor_sets_defaults_and_custom_labels(self) -> None:
    """Checks VideoExtenso defaults and supported label normalization."""

    block = VideoExtenso(**self.camera_kwargs(config=True))

    self.assertEqual(block.labels, [
      't(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)',
    ])

    labels = ('time', 'metadata', 'coords', 'eyy', 'exx')
    block = VideoExtenso(labels=labels, **self.camera_kwargs(config=True))

    self.assertEqual(block.labels, list(labels))

  def test_constructor_requires_configuration_window(self) -> None:
    """Checks VideoExtenso rejects operation without spot configuration."""

    with self.assertRaises(ValueError):
      VideoExtenso(**self.camera_kwargs(config=False))

  def test_constructor_validates_labels(self) -> None:
    """Checks VideoExtenso label count, type, and uniqueness validation."""

    cases = (
      ['too', 'few'],
      ['same'] * 5,
      ['time', 'meta', 'coords', 'eyy', 1],
    )

    for labels in cases:
      with self.subTest(labels=labels):
        with self.assertRaises(ValueError):
          VideoExtenso(labels=labels, **self.camera_kwargs(config=True))

  def test_prepare_forwards_process_options(self) -> None:
    """Checks VideoExtensoProcess option forwarding."""

    block = VideoExtenso(raise_on_lost_spot=False,
                         white_spots=True,
                         update_thresh=True,
                         num_spots=3,
                         safe_mode=True,
                         border=9,
                         min_area=42,
                         blur=3,
                         **self.camera_kwargs(config=True))

    with (patch.object(video_extenso_module, 'VideoExtensoProcess',
                       RecordingProcess),
          patch.object(Camera, 'prepare') as camera_prepare):
      block.prepare()

    camera_prepare.assert_called_once_with()
    process = RecordingProcess.instances[-1]

    self.assertIs(block.process_proc, process)
    self.assertEqual(process.kwargs, {
      'white_spots': True,
      'num_spots': 3,
      'min_area': 42,
      'blur': 3,
      'update_thresh': True,
      'safe_mode': True,
      'border': 9,
      'raise_on_lost_spot': False,
    })

  def test_configure_forwards_camera_and_detector_options(self) -> None:
    """Checks VideoExtensoConfig receives Camera and detector options."""

    def transform(img):
      return img

    block = VideoExtenso(transform=transform,
                         **self.camera_kwargs(config=True))
    block._camera = sentinel.camera
    block._log_queue = sentinel.log_queue
    block._log_level = 30
    block.freq = 123

    with patch.object(video_extenso_module, 'VideoExtensoConfig',
                      return_value=sentinel.config) as config:
      ret = block._configure()

    self.assertIs(ret, sentinel.config)
    config.assert_called_once_with(sentinel.camera,
                                   sentinel.log_queue,
                                   30,
                                   123.0,
                                   transform,
                                   white_spots=False,
                                   num_spots=None,
                                   min_area=150,
                                   blur=5,
                                   update_thresh=False,
                                   safe_mode=False,
                                   border=5)
