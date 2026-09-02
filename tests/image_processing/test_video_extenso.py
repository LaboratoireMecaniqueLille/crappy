# coding: utf-8

from dataclasses import dataclass
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import crappy.tool.image_processing.video_extenso.tracker as tracker_module
import crappy.tool.image_processing.video_extenso.video_extenso as ve_module
from crappy.tool.camera_config import Box, SpotsBoxes
from crappy.tool.image_processing.video_extenso.tracker import LostSpotError, Tracker
from crappy.tool.image_processing.video_extenso.video_extenso import VideoExtensoTool


class FakePipe:
  """Pipe-like test double for VideoExtensoTool tests."""

  def __init__(self, responses=None, send_error=None) -> None:
    self.responses = list(responses or [])
    self.send_error = send_error
    self.sent = list()
    self.closed = False

  def send(self, value) -> None:
    if self.send_error is not None:
      raise self.send_error
    self.sent.append(value)

  def poll(self, timeout=None) -> bool:
    return bool(self.responses)

  def recv(self):
    return self.responses.pop(0)

  def close(self) -> None:
    self.closed = True


class FakeTracker:
  """Process-like test double for tracker lifecycle tests."""

  def __init__(self, alive: bool = True, survive_terminate: bool = False):
    self.alive = alive
    self.survive_terminate = survive_terminate
    self.started = False
    self.join_calls = list()
    self.terminated = False
    self.killed = False

  def start(self) -> None:
    self.started = True

  def is_alive(self) -> bool:
    return self.alive

  def join(self, timeout=None) -> None:
    self.join_calls.append(timeout)

  def terminate(self) -> None:
    self.terminated = True
    if not self.survive_terminate:
      self.alive = False

  def kill(self) -> None:
    self.killed = True
    self.alive = False


@dataclass
class Prop:
  """Minimal object exposing a skimage-like bbox."""

  bbox: tuple[int, int, int, int]


class TestVideoExtensoTool(TestCase):
  """Unit tests for the VideoExtenso image-processing tool."""

  @staticmethod
  def _spots(*boxes: Box) -> SpotsBoxes:
    """Creates SpotsBoxes and initializes reference lengths."""

    spots = SpotsBoxes()
    for i, box in enumerate(boxes):
      spots[i] = box
    spots.save_length()
    return spots

  @staticmethod
  def _box(x_start: int,
           x_end: int,
           y_start: int,
           y_end: int,
           x_centroid: float | None = None,
           y_centroid: float | None = None) -> Box:
    """Creates a fully initialized Box."""

    return Box(x_start=x_start,
               x_end=x_end,
               y_start=y_start,
               y_end=y_end,
               x_centroid=(x_start + x_end) / 2
               if x_centroid is None else x_centroid,
               y_centroid=(y_start + y_end) / 2
               if y_centroid is None else y_centroid)

  def _make_tool(self, spots: SpotsBoxes, **kwargs) -> VideoExtensoTool:
    """Instantiates a VideoExtensoTool with quiet logging defaults."""

    kwargs.setdefault('thresh', 128)
    kwargs.setdefault('log_level', None)
    kwargs.setdefault('log_queue', None)
    return VideoExtensoTool(spots, **kwargs)

  def test_start_tracking_rejects_empty_spots(self) -> None:
    """Checks startup validation when no spot is configured."""

    tool = self._make_tool(SpotsBoxes())

    with self.assertRaises(AttributeError):
      tool.start_tracking()

  def test_start_tracking_creates_one_tracker_per_spot(self) -> None:
    """Checks tracker and pipe creation for configured spots."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots, update_thresh=True, white_spots=True)
    pipes = [(FakePipe(), FakePipe()), (FakePipe(), FakePipe())]
    created = list()

    def make_tracker(**kwargs):
      tracker = FakeTracker(alive=False)
      tracker.kwargs = kwargs
      created.append(tracker)
      return tracker

    with (patch.object(ve_module, 'Pipe', side_effect=pipes),
          patch.object(ve_module, 'Tracker', side_effect=make_tracker)):
      tool.start_tracking()

    self.assertEqual(tool._pipes, [pipes[0][0], pipes[1][0]])
    self.assertEqual(tool._trackers, created)
    self.assertTrue(all(tracker.started for tracker in created))
    self.assertIsNone(created[0].kwargs['thresh'])
    self.assertTrue(created[0].kwargs['white_spots'])

  def test_stop_tracking_stops_joins_kills_and_closes(self) -> None:
    """Checks tracker shutdown and pipe cleanup."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots)
    tracker_1 = FakeTracker(alive=True)
    tracker_2 = FakeTracker(alive=True, survive_terminate=True)
    pipe_1 = FakePipe()
    pipe_2 = FakePipe()
    tool._trackers = [tracker_1, tracker_2]
    tool._pipes = [pipe_1, pipe_2]
    tool._log = lambda *_: None
    tool._send = lambda pipe, value: pipe.send(value)

    tool.stop_tracking()

    self.assertEqual(pipe_1.sent, [('stop', 'stop', 'stop')])
    self.assertEqual(pipe_2.sent, [('stop', 'stop', 'stop')])
    self.assertEqual(tracker_1.join_calls, [0.1, 0.1, 0.1])
    self.assertEqual(tracker_2.join_calls, [0.1, 0.1, 0.1])
    self.assertTrue(tracker_1.terminated)
    self.assertTrue(tracker_2.terminated)
    self.assertFalse(tracker_1.killed)
    self.assertTrue(tracker_2.killed)
    self.assertTrue(pipe_1.closed)
    self.assertTrue(pipe_2.closed)

  def test_get_data_uses_latest_available_tracker_result(self) -> None:
    """Checks send/crop behavior and queued-result draining."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots, border=2)
    old_box = self._box(11, 21, 10, 20, x_centroid=15, y_centroid=15)
    new_box = self._box(12, 22, 10, 20, x_centroid=16, y_centroid=15)
    second_box = self._box(33, 43, 10, 20, x_centroid=38, y_centroid=15)
    pipe_1 = FakePipe([old_box, new_box])
    pipe_2 = FakePipe([second_box])
    tool._pipes = [pipe_1, pipe_2]
    sent = list()

    def send(pipe, value) -> None:
      sent.append((pipe, value[0], value[1], value[2].shape))

    tool._send = send

    ret = tool.get_data(np.zeros((50, 60), dtype=np.uint8))

    self.assertEqual(sent, [
      (pipe_1, 8, 8, (14, 14)),
      (pipe_2, 8, 28, (14, 14)),
    ])
    self.assertIs(tool.spots.spot_1, new_box)
    self.assertIs(tool.spots.spot_2, second_box)
    self.assertEqual(ret[0], [(15, 16), (15, 38)])
    self.assertEqual(ret[1], 0)
    self.assertAlmostEqual(ret[2], 10.0)

  def test_get_data_keeps_previous_box_without_tracker_reply(self) -> None:
    """Checks latest-known-state behavior when no fresh result is ready."""

    first_box = self._box(10, 20, 10, 20)
    spots = self._spots(first_box)
    tool = self._make_tool(spots, border=0)
    tool._pipes = [FakePipe()]
    tool._send = lambda *_: None

    self.assertEqual(tool.get_data(np.zeros((40, 40), dtype=np.uint8)),
                     ([(15.0, 15.0)], 0.0, 0.0))
    self.assertIs(tool.spots.spot_1, first_box)

  def test_get_data_raises_when_tracker_returns_error(self) -> None:
    """Checks propagation of tracker-side failures."""

    spots = self._spots(self._box(10, 20, 10, 20))
    tool = self._make_tool(spots)
    tool._pipes = [FakePipe(['stop'])]
    tool._send = lambda *_: None
    tool._log = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    with self.assertRaises(LostSpotError):
      tool.get_data(np.zeros((40, 40), dtype=np.uint8))

    self.assertEqual(stopped, [True])

  def test_safe_mode_overlap_raises(self) -> None:
    """Checks overlap handling in safe mode."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots, safe_mode=True)
    overlap_1 = self._box(10, 25, 10, 25)
    overlap_2 = self._box(20, 35, 20, 35)
    tool._pipes = [FakePipe([overlap_1]), FakePipe([overlap_2])]
    tool._send = lambda *_: None
    tool._log = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    with self.assertRaises(LostSpotError):
      tool.get_data(np.zeros((50, 50), dtype=np.uint8))

    self.assertEqual(stopped, [True])

  def test_overlap_helpers(self) -> None:
    """Checks Box and bbox overlap helpers."""

    ref = self._box(10, 20, 10, 20)

    self.assertTrue(VideoExtensoTool._overlap_box(
      ref, self._box(19, 30, 19, 30)))
    self.assertFalse(VideoExtensoTool._overlap_box(
      ref, self._box(20, 30, 20, 30)))

    self.assertTrue(VideoExtensoTool._overlap_bbox(
      Prop((10, 10, 20, 20)), Prop((19, 19, 30, 30))))
    self.assertFalse(VideoExtensoTool._overlap_bbox(
      Prop((10, 10, 20, 20)), Prop((20, 20, 30, 30))))

  def test_send_wrapper_on_linux(self) -> None:
    """Checks non-blocking send behavior on Linux."""

    tool = self._make_tool(self._spots(self._box(10, 20, 10, 20)))
    tool._system = 'Linux'
    pipe = FakePipe()

    with patch.object(ve_module, 'select', return_value=([], [pipe], [])):
      tool._send(pipe, ('payload', 'payload', 'payload'))

    self.assertEqual(pipe.sent, [('payload', 'payload', 'payload')])

    logs = list()
    tool._log = lambda level, msg: logs.append((level, msg))
    tool._last_warn = 0

    with (patch.object(ve_module, 'select', return_value=([], [], [])),
          patch.object(ve_module, 'time', return_value=2)):
      tool._send(pipe, ('other', 'other', 'other'))

    self.assertEqual(pipe.sent, [('payload', 'payload', 'payload')])
    self.assertEqual(len(logs), 1)


class TestTracker(TestCase):
  """Unit tests for the VideoExtenso Tracker process class."""

  def setUp(self) -> None:
    """Resets process-name state between tests."""

    Tracker.names = list()

  def _make_tracker(self, **kwargs) -> Tracker:
    """Creates a Tracker with a fake pipe."""

    kwargs.setdefault('pipe', FakePipe())
    kwargs.setdefault('logger_name', 'test')
    kwargs.setdefault('log_level', None)
    kwargs.setdefault('log_queue', None)
    return Tracker(**kwargs)

  def test_get_name_returns_unique_names(self) -> None:
    """Checks process name allocation."""

    self.assertEqual(Tracker.get_name('parent', 'Tracker'),
                     'parent.Tracker-1')
    self.assertEqual(Tracker.get_name('parent', 'Tracker'),
                     'parent.Tracker-2')

  def test_blur_validation(self) -> None:
    """Checks accepted and rejected blur values."""

    self._make_tracker(blur=None)
    self._make_tracker(blur=1)
    self._make_tracker(blur=5)

    for blur in (0, 2):
      with self.subTest(blur=blur):
        with self.assertRaises(ValueError):
          self._make_tracker(blur=blur)

  def test_evaluate_detects_black_spot(self) -> None:
    """Checks thresholding and absolute coordinates for dark spots."""

    tracker = self._make_tracker(thresh=128, blur=None, white_spots=False)
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[5:15, 6:16] = 0

    box = tracker._evaluate(100, 50, img)

    self.assertEqual((box.x_start, box.x_end), (106, 116))
    self.assertEqual((box.y_start, box.y_end), (55, 65))
    self.assertAlmostEqual(box.x_centroid, 110.5)
    self.assertAlmostEqual(box.y_centroid, 59.5)

  def test_evaluate_detects_white_spot(self) -> None:
    """Checks thresholding for bright spots."""

    tracker = self._make_tracker(thresh=128, blur=None, white_spots=True)
    img = np.zeros((20, 20), dtype=np.uint8)
    img[4:14, 3:13] = 255

    box = tracker._evaluate(10, 20, img)

    self.assertEqual((box.x_start, box.x_end), (13, 23))
    self.assertEqual((box.y_start, box.y_end), (24, 34))

  def test_evaluate_raises_when_spot_is_too_small(self) -> None:
    """Checks lost-spot detection for tiny thresholded objects."""

    tracker = self._make_tracker(thresh=128, blur=None, white_spots=False)
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10, 10] = 0

    with patch.object(tracker_module, 'threshold_otsu', return_value=128):
      with self.assertRaises(LostSpotError):
        tracker._evaluate(0, 0, img)

  def test_send_wrapper_on_linux(self) -> None:
    """Checks non-blocking child-to-parent send behavior on Linux."""

    pipe = FakePipe()
    tracker = self._make_tracker(pipe=pipe)
    tracker._system = 'Linux'

    with patch.object(tracker_module, 'select', return_value=([], [pipe], [])):
      tracker._send('value')

    self.assertEqual(pipe.sent, ['value'])

    logs = list()
    tracker._logger = None
    tracker._log = lambda level, msg: logs.append((level, msg))
    tracker._last_warn = 0

    with (patch.object(tracker_module, 'select', return_value=([], [], [])),
          patch.object(tracker_module, 'time', return_value=2)):
      tracker._send('other')

    self.assertEqual(pipe.sent, ['value'])
    self.assertEqual(len(logs), 1)
