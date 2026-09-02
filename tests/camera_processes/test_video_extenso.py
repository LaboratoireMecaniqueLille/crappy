# coding: utf-8

from unittest.mock import patch
import logging

import numpy as np

import crappy.blocks.camera_processes.video_extenso as video_extenso_module
from crappy.blocks.camera_processes.video_extenso import VideoExtensoProcess
from crappy.tool.camera_config import SpotsDetector
from crappy.tool.image_processing import LostSpotError

from tests.camera_process.camera_process_test_base import (CameraProcessTestBase,
                                                           TestLink)


class DummyVideoExtensoTool:
  """Small stand-in for the VideoExtenso image-processing tool."""

  instances: list["DummyVideoExtensoTool"] = list()

  def __init__(self, **kwargs) -> None:
    """Records constructor arguments and exposes deterministic outputs."""

    self.kwargs = kwargs
    self.spots = kwargs['spots']
    self.images = list()
    self.return_value = ([(1.0, 2.0)], 3.0, 4.0)
    self.raise_on_get_data = False
    self.start_calls = 0
    self.stop_calls = 0

    type(self).instances.append(self)

  def start_tracking(self) -> None:
    """Records that spot tracking was started."""

    self.start_calls += 1

  def stop_tracking(self) -> None:
    """Records that spot tracking was stopped."""

    self.stop_calls += 1

  def get_data(self, img: np.ndarray):
    """Records the processed frame and returns deterministic data."""

    self.images.append(np.copy(img))
    if self.raise_on_get_data:
      raise LostSpotError
    return self.return_value


class TestVideoExtensoProcess(CameraProcessTestBase):
  """Unit tests for the VideoExtenso CameraProcess wrapper."""

  @staticmethod
  def _detector() -> SpotsDetector:
    """Returns a configured detector with one spot to track."""

    detector = SpotsDetector(white_spots=True,
                             blur=3,
                             update_thresh=True,
                             safe_mode=True,
                             border=7)
    detector.thresh = 123
    detector.spots.set_spots([(1, 2, 3, 4)])
    detector.spots.save_length()
    return detector

  def setUp(self) -> None:
    """Resets the fake tool registry."""

    DummyVideoExtensoTool.instances.clear()

  def _init_process(self,
                    raise_on_lost_spot: bool = True
                    ) -> tuple[VideoExtensoProcess, DummyVideoExtensoTool]:
    """Creates a process and initializes its mocked processing tool."""

    process = VideoExtensoProcess(
      detector=self._detector(),
      raise_on_lost_spot=raise_on_lost_spot)
    self._process = process
    self.set_test_logger(process)

    with patch.object(video_extenso_module, 'VideoExtensoTool',
                      DummyVideoExtensoTool):
      process.init()

    return process, DummyVideoExtensoTool.instances[0]

  def test_init_forwards_detector_options_and_starts_tracking(self) -> None:
    """Checks tool arguments and tracker startup during initialization."""

    detector = self._detector()
    process = VideoExtensoProcess(detector=detector)
    self._process = process
    process._log_level = logging.DEBUG
    log_queue = object()
    process._log_queue = log_queue

    with patch.object(video_extenso_module, 'VideoExtensoTool',
                      DummyVideoExtensoTool):
      process.init()

    tool = DummyVideoExtensoTool.instances[0]
    self.assertIs(process._ve, tool)
    self.assertEqual(tool.kwargs, {
      'spots': detector.spots,
      'thresh': 123,
      'log_level': logging.DEBUG,
      'log_queue': log_queue,
      'white_spots': True,
      'update_thresh': True,
      'safe_mode': True,
      'border': 7,
      'blur': 3,
    })
    self.assertEqual(tool.start_calls, 1)
    self.assertEqual(tool.stop_calls, 0)

  def test_loop_sends_formatted_data_and_overlay(self) -> None:
    """Checks downstream data formatting and overlay forwarding."""

    process, tool = self._init_process()
    link = TestLink()
    process._outputs = [link]
    process._labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    sent_overlays = list()
    process.send_to_draw = sent_overlays.append

    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    metadata = {'ImageUniqueID': 2, 't(s)': 0.2}
    process.img = img
    process.metadata = metadata
    tool.return_value = ([(1.5, 2.5)], 3.5, 4.5)

    process.loop()

    self.assertTrue(link.sent.is_set())
    self.assertEqual(link.sent_values[-1], {
      't(s)': 0.2,
      'meta': metadata,
      'Coord(px)': [(1.5, 2.5)],
      'Eyy(%)': 3.5,
      'Exx(%)': 4.5,
    })
    np.testing.assert_array_equal(tool.images[0], img)
    self.assertEqual(sent_overlays, [tool.spots])

  def test_loop_forwards_overlay_when_get_data_returns_none(self) -> None:
    """Checks that missing data suppresses output but not the overlay."""

    process, tool = self._init_process()
    link = TestLink()
    process._outputs = [link]
    process._labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    sent_overlays = list()
    process.send_to_draw = sent_overlays.append
    process.img = np.zeros((3, 4), dtype=np.uint8)
    process.metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    tool.return_value = None

    process.loop()

    self.assertFalse(link.sent.is_set())
    self.assertEqual(len(tool.images), 1)
    self.assertEqual(sent_overlays, [tool.spots])

  def test_loop_reraises_lost_spot_when_requested(self) -> None:
    """Checks LostSpotError propagation and tracker shutdown."""

    process, tool = self._init_process(raise_on_lost_spot=True)
    process.img = np.zeros((3, 4), dtype=np.uint8)
    tool.raise_on_get_data = True

    with self.assertRaises(LostSpotError):
      process.loop()

    self.assertEqual(tool.stop_calls, 1)
    self.assertFalse(process._lost_spots)

  def test_loop_handles_lost_spot_and_then_stays_idle(self) -> None:
    """Checks non-raising lost-spot handling and subsequent idle behavior."""

    process, tool = self._init_process(raise_on_lost_spot=False)
    sent_overlays = list()
    process.send_to_draw = sent_overlays.append
    process.img = np.zeros((3, 4), dtype=np.uint8)
    tool.raise_on_get_data = True

    process.loop()

    self.assertTrue(process._lost_spots)
    self.assertEqual(tool.stop_calls, 1)
    self.assertEqual(sent_overlays, [list()])
    self.assertEqual(len(tool.images), 1)

    process.fps_count = 5
    with patch.object(video_extenso_module, 'sleep') as sleep:
      process.loop()

    sleep.assert_called_once_with(0.1)
    self.assertEqual(process.fps_count, 4)
    self.assertEqual(len(tool.images), 1)
    self.assertEqual(tool.stop_calls, 1)

  def test_finish_before_and_after_initialization(self) -> None:
    """Checks that finish is safe and stops an initialized tool."""

    process = VideoExtensoProcess(detector=self._detector())
    self._process = process
    self.set_test_logger(process)

    process.finish()
    self.assertIsNone(process._ve)

    with patch.object(video_extenso_module, 'VideoExtensoTool',
                      DummyVideoExtensoTool):
      process.init()

    tool = DummyVideoExtensoTool.instances[0]
    process.finish()

    self.assertEqual(tool.start_calls, 1)
    self.assertEqual(tool.stop_calls, 1)
