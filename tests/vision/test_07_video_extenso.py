# coding: utf-8

from unittest.mock import Mock, patch, sentinel
import logging

import numpy as np

import crappy.blocks.vision.video_extenso as video_extenso_module
from crappy.blocks.vision import VideoExtensoProcessor
from crappy.tool.camera_config import VideoExtensoConfig
from crappy.tool.camera_config.config_tools import SpotsBoxes
from crappy.tool.image_processing import LostSpotError

from .vision_test_base import VisionTestBase


class RecordingVideoExtensoTool:
  """Small VideoExtensoTool double recording lifecycle and images."""

  instances: list['RecordingVideoExtensoTool'] = list()

  def __init__(self, **kwargs) -> None:
    """Stores constructor arguments and deterministic processing state."""

    self.kwargs = kwargs
    self.spots = kwargs['spots']
    self.images = list()
    self.return_value = ([(1.0, 2.0)], 3.0, 4.0)
    self.raise_on_get_data = False
    self.raise_on_stop = False
    self.start_calls = 0
    self.stop_calls = 0
    type(self).instances.append(self)

  def start_tracking(self) -> None:
    """Records tracker startup."""

    self.start_calls += 1

  def stop_tracking(self) -> None:
    """Records tracker cleanup and optionally simulates failure."""

    self.stop_calls += 1
    if self.raise_on_stop:
      raise RuntimeError('tracker cleanup failed')

  def get_data(self, image: np.ndarray):
    """Records an image and returns data or simulates a lost spot."""

    self.images.append(np.copy(image))
    if self.raise_on_get_data:
      raise LostSpotError
    return self.return_value


class TestVideoExtensoProcessor(VisionTestBase):
  """Unit tests for the VideoExtensoProcessor VisionBlock."""

  def setUp(self) -> None:
    """Resets the processing-tool registry."""

    super().setUp()
    RecordingVideoExtensoTool.instances.clear()

  def make_processor(self, **kwargs) -> VideoExtensoProcessor:
    """Creates and tracks a VideoExtensoProcessor."""

    processor = VideoExtensoProcessor(**kwargs)
    self.track_block(processor)
    return processor

  @staticmethod
  def spots(*spots) -> SpotsBoxes:
    """Builds configured spot boxes for preparation results."""

    boxes = SpotsBoxes()
    boxes.set_spots(list(spots) or [(1, 2, 3, 4)])
    boxes.save_length()
    return boxes

  @staticmethod
  def add_image_input(processor: VideoExtensoProcessor,
                      name: str = 've-image') -> Mock:
    """Registers a minimal input ImageLink double."""

    link = Mock()
    link.name = name
    processor.add_img_input(link)
    return link

  def feed_image(self,
                 processor: VideoExtensoProcessor,
                 image: np.ndarray,
                 metadata=None,
                 name: str = 've-image') -> None:
    """Installs one received image and makes receive_imgs report it."""

    if name not in processor.last_received:
      self.add_image_input(processor, name)
    if metadata is None:
      metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    processor.last_received[name].metadata = metadata
    processor.last_received[name].img = image
    processor.receive_imgs = Mock(return_value=[name])

  def test_constructor_sets_defaults_and_custom_labels(self) -> None:
    """Checks result labels and reserved overlay labeling."""

    processor = self.make_processor()
    self.assertEqual(processor.labels, [
      't(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'overlay',
    ])

    labels = ('time', 'metadata', 'coordinates', 'eyy', 'exx')
    custom = self.make_processor(labels=labels)
    self.assertEqual(custom.labels, [*labels, 'overlay'])

  def test_constructor_validates_labels_and_tracking_options(self) -> None:
    """Checks spot-detector and tracker argument validation."""

    invalid = (
      {'labels': ['too', 'few']},
      {'labels': ['same'] * 5},
      {'labels': ['a', 'b', 'c', 'd', 1]},
      {'raise_on_lost_spot': 1},
      {'white_spots': 1},
      {'num_spots': 0},
      {'num_spots': 5},
      {'num_spots': 1.5},
      {'min_area': -1},
      {'min_area': 1.5},
      {'blur': 0},
      {'blur': 2},
      {'blur': 1.5},
      {'update_thresh': 1},
      {'safe_mode': 1},
      {'border': -1},
      {'border': 1.5},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          self.make_processor(**options)

  def test_request_config_contains_all_detector_options(self) -> None:
    """Checks the mandatory source configurator and forwarded options."""

    processor = self.make_processor(white_spots=True,
                                    update_thresh=True,
                                    num_spots=3,
                                    safe_mode=True,
                                    border=9,
                                    min_area=42,
                                    blur=3)

    request = processor.request_config('camera')

    self.assertEqual(request.requester, processor.name)
    self.assertEqual(request.img_source, 'camera')
    self.assertIs(request.configurator, VideoExtensoConfig)
    self.assertTrue(request.required)
    self.assertEqual(request.args, tuple())
    self.assertEqual(request.kwargs, {
      'white_spots': True,
      'num_spots': 3,
      'min_area': 42,
      'blur': 3,
      'update_thresh': True,
      'safe_mode': True,
      'border': 9,
    })

  def test_prepare_validates_supported_topology(self) -> None:
    """Checks the one-image-input, no-other-input topology."""

    processor = self.make_processor()
    with self.assertRaises(IOError):
      processor.prepare()

    self.add_image_input(processor)
    processor.img_outputs.append(Mock())
    with self.assertRaises(IOError):
      processor.prepare()

    processor.img_outputs.clear()
    processor.inputs.append(Mock())
    with self.assertRaises(IOError):
      processor.prepare()

    processor.inputs.clear()
    self.add_image_input(processor, 'second-image')
    with self.assertRaises(IOError):
      processor.prepare()

  def test_prepare_builds_tool_and_starts_trackers(self) -> None:
    """Checks processing arguments, shared preparation, and tracker startup."""

    processor = self.make_processor(white_spots=True,
                                    update_thresh=True,
                                    safe_mode=True,
                                    border=9,
                                    blur=3)
    self.add_image_input(processor)
    configured = self.spots((2, 3, 4, 5))
    processor.recv_configs = Mock(return_value={
      'camera': (configured, 123),
    })
    processor._log_queue = sentinel.log_queue
    processor._log_level = logging.WARNING

    with (patch.object(video_extenso_module, 'VideoExtensoTool',
                       RecordingVideoExtensoTool),
          patch.object(video_extenso_module.VisionBlock,
                       'prepare') as inherited):
      processor.prepare()

    tool = RecordingVideoExtensoTool.instances[-1]
    self.assertIs(processor._ve, tool)
    self.assertIs(processor._spots, configured)
    self.assertEqual(processor._thresh, 123)
    self.assertEqual(tool.kwargs, {
      'spots': configured,
      'thresh': 123,
      'log_level': logging.WARNING,
      'log_queue': sentinel.log_queue,
      'white_spots': True,
      'update_thresh': True,
      'safe_mode': True,
      'border': 9,
      'blur': 3,
    })
    inherited.assert_called_once_with()
    self.assertEqual(tool.start_calls, 1)

  def test_prepare_rejects_missing_ambiguous_or_malformed_config(self) -> None:
    """Checks invalid source-configuration result sets."""

    cases = (
      ({}, RuntimeError),
      ({'one': (self.spots(), 1),
        'two': (self.spots(), 2)}, NotImplementedError),
      ({'camera': (self.spots(),)}, ValueError),
      ({'camera': object()}, TypeError),
      ({'camera': (None, 1)}, RuntimeError),
      ({'camera': (self.spots(), None)}, RuntimeError),
    )
    for configs, error in cases:
      with self.subTest(configs=configs, error=error):
        processor = self.make_processor()
        self.add_image_input(processor)
        processor.recv_configs = Mock(return_value=configs)
        processor._log_queue = Mock()
        with self.assertRaises(error):
          processor.prepare()

  def test_prepare_requires_logging_queue(self) -> None:
    """Checks tracker processes cannot start without logging infrastructure."""

    processor = self.make_processor()
    self.add_image_input(processor)
    processor.recv_configs = Mock(return_value={
      'camera': (self.spots(), 123),
    })

    with self.assertRaises(RuntimeError):
      processor.prepare()

  def test_loop_skips_when_no_new_image_or_after_spot_loss(self) -> None:
    """Checks both idle branches and handled-image frequency accounting."""

    processor = self.make_processor(display_freq=True)
    processor.receive_imgs = Mock(return_value=[])
    processor._print_freq = Mock()

    processor.loop()

    processor.receive_imgs.assert_called_once_with()
    processor._print_freq.assert_called_once_with(img_handled=False)

    processor.receive_imgs.reset_mock()
    processor._print_freq.reset_mock()
    processor._lost_spots = True
    processor.loop()
    processor.receive_imgs.assert_not_called()
    processor._print_freq.assert_called_once_with(img_handled=False)

  def test_loop_sends_results_and_overlay(self) -> None:
    """Checks formatted result publication and processed image forwarding."""

    processor = self.make_processor()
    spots = self.spots()
    tool = RecordingVideoExtensoTool(spots=spots)
    processor._ve = tool
    processor.send = Mock()
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    metadata = {'ImageUniqueID': 2, 't(s)': 0.2}
    self.feed_image(processor, image, metadata)

    processor.loop()

    processor.send.assert_called_once_with([
      0.2, metadata, [(1.0, 2.0)], 3.0, 4.0, spots,
    ])
    np.testing.assert_array_equal(tool.images[-1], image)
    self.assertEqual(processor._last_data, tool.return_value)

  def test_loop_suppresses_output_when_tracker_has_no_result(self) -> None:
    """Checks a handled image can legitimately produce no measurement."""

    processor = self.make_processor(display_freq=True)
    tool = RecordingVideoExtensoTool(spots=self.spots())
    tool.return_value = None
    processor._ve = tool
    processor.send = Mock()
    processor._print_freq = Mock()
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8))

    processor.loop()

    processor.send.assert_not_called()
    processor._print_freq.assert_called_once_with(img_handled=True)

  def test_loop_reraises_lost_spot_after_stopping_trackers(self) -> None:
    """Checks fatal spot loss shuts down trackers and propagates."""

    processor = self.make_processor(raise_on_lost_spot=True)
    tool = RecordingVideoExtensoTool(spots=self.spots())
    tool.raise_on_get_data = True
    processor._ve = tool
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8))

    with self.assertRaises(LostSpotError):
      processor.loop()

    self.assertEqual(tool.stop_calls, 1)
    self.assertFalse(processor._lost_spots)

  def test_loop_tolerates_lost_spot_clears_overlay_then_stays_idle(self
                                                                  ) -> None:
    """Checks nonfatal spot loss sends final data and disables processing."""

    processor = self.make_processor(raise_on_lost_spot=False)
    spots = self.spots()
    tool = RecordingVideoExtensoTool(spots=spots)
    tool.raise_on_get_data = True
    processor._ve = tool
    processor._last_data = ([(1, 2)], 3, 4)
    processor.send = Mock()
    metadata = {'ImageUniqueID': 3, 't(s)': 0.3}
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8), metadata)

    processor.loop()

    self.assertTrue(processor._lost_spots)
    self.assertEqual(tool.stop_calls, 1)
    processor.send.assert_called_once_with([
      0.3, metadata, [(1, 2)], 3, 4, list(),
    ])

    processor.receive_imgs.reset_mock()
    processor.loop()
    processor.receive_imgs.assert_not_called()
    self.assertEqual(tool.stop_calls, 1)

  def test_loop_requires_metadata_and_initialized_tool(self) -> None:
    """Checks clear failures for impossible partially initialized states."""

    processor = self.make_processor()
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8))
    processor.last_received['ve-image'].metadata = None
    with self.assertRaises(RuntimeError):
      processor.loop()

    processor.last_received['ve-image'].metadata = {
      'ImageUniqueID': 1,
      't(s)': 0.1,
    }
    with self.assertRaises(RuntimeError):
      processor.loop()

  def test_finish_stops_trackers_and_always_releases_shared_memory(self
                                                                  ) -> None:
    """Checks normal and exceptional tracker cleanup paths."""

    processor = self.make_processor()
    tool = RecordingVideoExtensoTool(spots=self.spots())
    processor._ve = tool

    with patch.object(video_extenso_module.VisionBlock, 'finish') as inherited:
      processor.finish()
    self.assertEqual(tool.stop_calls, 1)
    inherited.assert_called_once_with()

    tool.raise_on_stop = True
    with patch.object(video_extenso_module.VisionBlock, 'finish') as inherited:
      with self.assertRaises(RuntimeError):
        processor.finish()
    inherited.assert_called_once_with()


if __name__ == '__main__':
  import unittest
  unittest.main()
