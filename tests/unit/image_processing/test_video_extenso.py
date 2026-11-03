# coding: utf-8

from tests.support.processes import run_python
from dataclasses import dataclass
from inspect import signature, Parameter
from multiprocessing import Event
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from crappy._failure import FailureManager
from crappy._exceptions import ProcessExitError, ShutdownTimeoutError
from tests.support.assertions import exception_leaves

import crappy.tool.image_processing.video_extenso.tracker as tracker_module
import crappy.tool.image_processing.video_extenso.video_extenso as ve_module
from crappy.tool.camera_config import Box, SpotsBoxes
from crappy.tool.image_processing.video_extenso.tracker import LostSpotError, Tracker
from crappy.tool.image_processing.video_extenso.video_extenso import VideoExtensoTool


class FakePipe:
  """Pipe-like test double for VideoExtensoTool tests."""

  def __init__(self, responses=None, send_error=None, recv_error=None) -> None:
    self.responses = list(responses or [])
    self.send_error = send_error
    self.recv_error = recv_error
    self.sent = list()
    self.closed = False

  def send(self, value) -> None:
    if self.send_error is not None:
      raise self.send_error
    self.sent.append(value)

  def send_bytes(self, payload) -> None:
    from pickle import loads
    self.send(loads(payload))

  def recv_bytes(self):
    from pickle import dumps
    return dumps(self.recv())

  def poll(self, timeout=None) -> bool:
    return bool(self.responses)

  def recv(self):
    if self.recv_error is not None:
      raise self.recv_error
    return self.responses.pop(0)

  def close(self) -> None:
    self.closed = True


class FakeTracker:
  """Process-like test double for tracker lifecycle tests."""

  def __init__(self, alive: bool = True, survive_terminate: bool = False):
    self.alive = alive
    self.pid = None
    self.exitcode = None if alive else 0
    self.name = f'FakeTracker-{id(self)}'
    self.survive_terminate = survive_terminate
    self.started = False
    self.join_calls = list()
    self.terminated = False
    self.killed = False
    self.closed = False

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

  def close(self) -> None:
    self.closed = True


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
    kwargs.setdefault('white_spots', False)
    kwargs.setdefault('update_thresh', False)
    kwargs.setdefault('safe_mode', False)
    kwargs.setdefault('border', 5)
    kwargs.setdefault('blur', 5)
    manager = FailureManager('test VideoExtensoTool', 'main')
    manager._logger = Mock()
    kwargs.setdefault('failure_manager', manager)
    kwargs.setdefault('stop_event', Event())
    tool = VideoExtensoTool(spots, **kwargs)

    def release() -> None:
      with tool._failure_manager.collect('test resources', raise_on_exit=False):
        with tool._failure_manager.step('stop fake trackers'):
          VideoExtensoTool.stop_tracking(tool)

    self.addCleanup(release)
    return tool

  def test_reporting_dependencies_are_required(self) -> None:
    for factory, names in (
        (VideoExtensoTool, ('failure_manager', 'stop_event')),
        (Tracker, ('failure_queue', 'block_name'))):
      for name in names:
        with self.subTest(factory=factory, argument=name):
          self.assertIs(signature(factory).parameters[name].default,
                        Parameter.empty)

  def test_tool_keeps_owner_manager_logger_and_stop_event(self) -> None:
    manager = FailureManager('owner', 'main')
    logger = manager._logger = Mock()
    stop = Event()
    tool = self._make_tool(SpotsBoxes(), failure_manager=manager,
                           stop_event=stop)
    self.assertIs(tool._failure_manager, manager)
    self.assertIs(manager._logger, logger)
    self.assertIs(tool._owner_stop, stop)

  def test_destructor_accepts_partially_initialized_instance(self) -> None:
    """Tests cleanup after construction fails before attributes are set."""

    tool = VideoExtensoTool.__new__(VideoExtensoTool)

    tool.__del__()

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
    self.assertTrue(all(tracker.kwargs['stop_event'] is tool._tracking_stop
                        for tracker in created))

  def test_stop_tracking_stops_joins_kills_and_closes(self) -> None:
    """Checks tracker shutdown and pipe cleanup."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots)
    tracker_1 = FakeTracker(alive=True)
    tracker_2 = FakeTracker(alive=True, survive_terminate=True)
    pipe_1 = FakePipe()
    pipe_2 = FakePipe()
    tool._tracking_stop = Event()
    tool._trackers = [tracker_1, tracker_2]
    tool._started_trackers = list(tool._trackers)
    tool._pipes = [pipe_1, pipe_2]
    tool._log = lambda *_: None

    with (patch.object(ve_module, 'monotonic', return_value=0.0),
          self.assertRaises(ExceptionGroup) as caught):
      tool.stop_tracking()
    self.assertTrue(all(isinstance(error, ShutdownTimeoutError)
                        for error in exception_leaves(caught.exception)))

    self.assertTrue(tool._tracking_stop.is_set())
    self.assertEqual(pipe_1.sent, [])
    self.assertEqual(pipe_2.sent, [])
    self.assertEqual(tracker_1.join_calls, [3.0, 0.5])
    self.assertEqual(tracker_2.join_calls, [3.0, 0.5, 0.5])
    self.assertTrue(tracker_1.terminated)
    self.assertTrue(tracker_2.terminated)
    self.assertFalse(tracker_1.killed)
    self.assertTrue(tracker_2.killed)
    self.assertTrue(pipe_1.closed)
    self.assertTrue(pipe_2.closed)
    self.assertTrue(tracker_1.closed)
    self.assertTrue(tracker_2.closed)
    tool.stop_tracking()
    self.assertEqual(tracker_2.join_calls, [3.0, 0.5, 0.5])

  def test_shutdown_deadlines_are_shared_by_all_trackers(self) -> None:
    """Four stuck trackers cannot multiply the enclosing worker's wait."""

    tool = self._make_tool(SpotsBoxes())
    trackers = [FakeTracker(survive_terminate=True) for _ in range(4)]
    tool._trackers = list(trackers)
    tool._started_trackers = list(trackers)
    clock = [0.0]

    def join(tracker, timeout):
      tracker.join_calls.append(timeout)
      clock[0] += timeout

    for tracker in trackers:
      tracker.join = lambda timeout, tracker=tracker: join(tracker, timeout)

    with (patch.object(ve_module, 'monotonic', side_effect=lambda: clock[0]),
          self.assertRaises(ExceptionGroup) as caught):
      tool.stop_tracking()

    self.assertEqual(clock[0], 4.0)
    self.assertEqual([error.timeout for error in
                      exception_leaves(caught.exception)], [3.0] * 4)
    self.assertEqual(trackers[0].join_calls, [3.0, 0.5, 0.5])
    for tracker in trackers[1:]:
      self.assertEqual(tracker.join_calls, [0.0, 0.0, 0.0])
    self.assertTrue(all(tracker.terminated and tracker.killed and tracker.closed
                        for tracker in trackers))

  def test_partial_start_retains_both_pipe_endpoints(self) -> None:
    """A failed Tracker constructor or start cannot lose allocated Pipes."""

    for failed in ('constructor', 'start'):
      with self.subTest(failed=failed):
        tool = self._make_tool(self._spots(self._box(10, 20, 10, 20),
                                           self._box(30, 40, 10, 20)))
        pipes = [(FakePipe(), FakePipe()), (FakePipe(), FakePipe())]
        first, second = FakeTracker(alive=False), FakeTracker(alive=False)
        error = RuntimeError(failed)
        constructed = [first, error if failed == 'constructor' else second]
        with (patch.object(ve_module, 'Pipe', side_effect=pipes),
              patch.object(ve_module, 'Tracker', side_effect=constructed),
              patch.object(second, 'start', side_effect=error)):
          with self.assertRaises(RuntimeError) as caught:
            tool.start_tracking()
        self.assertIs(caught.exception, error)
        self.assertEqual(tool._started_trackers, [first])
        tool.stop_tracking()
        self.assertEqual(second.join_calls, [])
        for inlet, outlet in pipes:
          self.assertTrue(inlet.closed)
          self.assertTrue(outlet.closed)

  def test_cleanup_failures_do_not_skip_peers_or_prevent_retry(
      self) -> None:
    """Each tracker and Pipe is released independently, including on retry."""

    tool = self._make_tool(self._spots(self._box(10, 20, 10, 20),
                                       self._box(30, 40, 10, 20)))
    first = FakeTracker(survive_terminate=True)
    second = FakeTracker()
    inlet, outlet = FakePipe(), FakePipe()
    tool._tracking_stop = Event()
    tool._trackers = [first, second]
    tool._started_trackers = [first, second]
    tool._pipes = [inlet, outlet]
    errors = (OSError('join'), RuntimeError('terminate'), ValueError('close'))
    interrupt = KeyboardInterrupt()
    with (patch.object(first, 'join', side_effect=[errors[0], None, None]),
          patch.object(first, 'terminate', side_effect=errors[1]),
          patch.object(first, 'close', side_effect=errors[2]),
          patch.object(outlet, 'close', side_effect=interrupt)):
      with self.assertRaises(BaseExceptionGroup) as caught:
        tool.stop_tracking()
    leaves = exception_leaves(caught.exception)
    for error in (*errors, interrupt):
      self.assertIn(error, leaves)
    self.assertEqual(sum(isinstance(error, ShutdownTimeoutError)
                         for error in leaves), 2)
    self.assertIsNone(interrupt.__cause__)
    self.assertTrue(first.killed)
    self.assertTrue(second.closed)
    self.assertTrue(inlet.closed)
    tool.stop_tracking()
    tool.stop_tracking()
    self.assertTrue(tool._tracking_stop.is_set())
    self.assertEqual(inlet.sent, [])
    self.assertEqual(outlet.sent, [])
    self.assertTrue(outlet.closed)
    self.assertTrue(first.closed)

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
    tool._trackers = [FakeTracker(), FakeTracker()]
    tool._started_trackers = list(tool._trackers)
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
    tool._trackers.clear()

  def test_get_data_keeps_previous_box_without_tracker_reply(self) -> None:
    """Checks latest-known-state behavior when no fresh result is ready."""

    first_box = self._box(10, 20, 10, 20)
    spots = self._spots(first_box)
    tool = self._make_tool(spots, border=0)
    tool._pipes = [FakePipe()]
    tool._trackers = [FakeTracker()]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda *_: None

    self.assertEqual(tool.get_data(np.zeros((40, 40), dtype=np.uint8)),
                     ([(15.0, 15.0)], 0.0, 0.0))
    self.assertIs(tool.spots.spot_1, first_box)
    tool._trackers.clear()

  def test_get_data_returns_tracker_lost_spot_reason(self) -> None:
    """Checks a tracker loss reason survives draining earlier positions."""

    spots = self._spots(self._box(10, 20, 10, 20))
    tool = self._make_tool(spots)
    reason = 'The selected spot left the frame'
    tool._pipes = [FakePipe([self._box(11, 21, 10, 20), reason])]
    tool._trackers = [FakeTracker()]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda *_: None
    tool._log = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    self.assertEqual(tool.get_data(np.zeros((40, 40), dtype=np.uint8)), reason)

    self.assertEqual(stopped, [])
    tool._trackers.clear()

  def test_safe_mode_overlap_returns_lost_spot(self) -> None:
    """Checks overlap handling in safe mode."""

    spots = self._spots(self._box(10, 20, 10, 20),
                        self._box(30, 40, 10, 20))
    tool = self._make_tool(spots, safe_mode=True)
    overlap_1 = self._box(10, 25, 10, 25)
    overlap_2 = self._box(20, 35, 20, 35)
    tool._pipes = [FakePipe([overlap_1]), FakePipe([overlap_2])]
    tool._trackers = [FakeTracker(), FakeTracker()]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda *_: None
    tool._log = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    self.assertEqual(tool.get_data(np.zeros((50, 50), dtype=np.uint8)),
                     'Overlapping spots detected in safe mode')

    self.assertEqual(stopped, [])
    tool._trackers.clear()

  def test_get_data_rejects_missing_trackers(self) -> None:
    """Checks that tracking cannot silently run before startup."""

    tool = self._make_tool(self._spots(self._box(10, 20, 10, 20)))

    with self.assertRaises(RuntimeError):
      tool.get_data(np.zeros((40, 40), dtype=np.uint8))

  def test_get_data_preserves_lost_spot_from_dead_tracker(self) -> None:
    """Checks a queued loss report is read before checking liveness."""

    tool = self._make_tool(self._spots(self._box(10, 20, 10, 20)))
    reason = 'The selected spot was lost'
    tool._pipes = [FakePipe([reason], send_error=BrokenPipeError)]
    tool._trackers = [FakeTracker(alive=False)]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda pipe, value: pipe.send(value)
    tool._log = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    self.assertEqual(tool.get_data(np.zeros((40, 40), dtype=np.uint8)), reason)

    self.assertEqual(stopped, [])

  def test_get_data_rejects_dead_tracker_without_loss_report(self) -> None:
    """Checks an unexpectedly dead Tracker cannot return stale data."""

    tool = self._make_tool(self._spots(self._box(10, 20, 10, 20)))
    tool._pipes = [FakePipe([None], recv_error=EOFError)]
    tool._trackers = [FakeTracker(alive=False)]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda *_: None
    stopped = list()
    tool.stop_tracking = lambda: stopped.append(True)

    with self.assertRaises(ExceptionGroup) as caught:
      tool.get_data(np.zeros((40, 40), dtype=np.uint8))

    self.assertEqual(stopped, [True])

  def test_get_data_applies_final_box_from_dead_tracker(self) -> None:
    """Checks a dead Tracker's queued position is not discarded."""

    initial_box = self._box(10, 20, 10, 20)
    final_box = self._box(12, 22, 11, 21)
    tool = self._make_tool(self._spots(initial_box))
    tool._pipes = [FakePipe([final_box])]
    tool._trackers = [FakeTracker(alive=False)]
    tool._started_trackers = list(tool._trackers)
    tool._send = lambda *_: None
    with self.assertRaises(ExceptionGroup) as caught:
      tool.get_data(np.zeros((40, 40), dtype=np.uint8))

    self.assertIs(tool.spots.spot_1, final_box)

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
          patch.object(ve_module, 'monotonic', return_value=2)):
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
    kwargs.setdefault('stop_event', Event())
    kwargs.setdefault('failure_queue', Mock())
    kwargs.setdefault('block_name', 'test tracker owner')
    return Tracker(**kwargs)

  def test_run_stops_on_event_without_a_pipe_message(self) -> None:
    """An idle tracker notices the Event without any control packet."""

    pipe = FakePipe()
    stop = Event()
    tracker = self._make_tracker(pipe=pipe, stop_event=stop)
    with patch.object(pipe, 'poll', side_effect=lambda _: stop.set() or False):
      tracker.run()

    self.assertTrue(pipe.closed)
    self.assertEqual(pipe.sent, [])
    self.assertEqual(tracker._failure_manager.snapshot(), ())

  def test_get_name_returns_unique_names(self) -> None:
    """Checks process name allocation."""

    self.assertEqual(Tracker.get_name('parent', 'Tracker'),
                     'parent.Tracker-1')
    self.assertEqual(Tracker.get_name('parent', 'Tracker'),
                     'parent.Tracker-2')

  def test_blur_options_are_forwarded(self) -> None:
    """Trackers receive blur options already validated by their Block."""

    self._make_tracker(blur=None)
    self._make_tracker(blur=1)
    self._make_tracker(blur=5)

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

  def test_run_sends_only_reason_for_tolerated_lost_spot(self) -> None:
    """Checks explicit and fallback diagnostics stay ordinary data strings."""

    for reason in ('The selected spot left the frame', ''):
      with self.subTest(reason=reason):
        pipe = FakePipe([(0, 0, np.zeros((20, 20), dtype=np.uint8))])
        tracker = self._make_tracker(pipe=pipe)
        with patch.object(tracker, '_evaluate',
                          side_effect=LostSpotError(reason)):
          tracker.run()

        self.assertEqual(pipe.sent, [reason or 'The selected spot was lost'])
        self.assertTrue(pipe.closed)
        self.assertEqual(tracker._failure_manager.snapshot(), ())

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
          patch.object(tracker_module, 'monotonic', return_value=2)):
      tracker._send('other')

    self.assertEqual(pipe.sent, ['value'])
    self.assertEqual(len(logs), 1)


class FaultTracker(Tracker):
  """A real tracker injecting either an expected loss or an actual failure."""

  def __init__(self, *args, fault='error', peer_ready=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.fault = fault
    self.peer_ready = peer_ready

  def _evaluate(self, x_start, y_start, img):
    if self.peer_ready is not None and not self.peer_ready.wait(5):
      raise RuntimeError('test peer did not start')
    if self.fault == 'healthy':
      return super()._evaluate(x_start, y_start, img)
    if self.fault == 'exit':
      import os
      os._exit(9)
    if self.fault == 'loss':
      raise LostSpotError('expected spot loss')
    original = ValueError('tracker programming error')
    original.__cause__ = KeyError('original cause')
    original.add_note('original tracker note')
    raise BaseExceptionGroup('tracker errors', [original, KeyboardInterrupt()])


class ReadyTracker(Tracker):
  """Signals that a healthy peer has processed its first frame."""

  def __init__(self, *args, peer_ready, **kwargs):
    super().__init__(*args, **kwargs)
    self.peer_ready = peer_ready

  def _evaluate(self, x_start, y_start, img):
    self.peer_ready.set()
    return super()._evaluate(x_start, y_start, img)


class TestTrackerReporting(TestCase):
  """Checks real tracker failures independently of normal spot loss."""

  def make_tool(self, fault, raise_on_lost_spot=False, with_peer=False):
    from multiprocessing import Event
    from functools import partial
    from crappy._failure import FailureManager

    spots = SpotsBoxes()
    spots[0] = Box(x_start=5, x_end=15, y_start=5, y_end=15,
                   x_centroid=10, y_centroid=10)
    if with_peer:
      spots[1] = Box(x_start=16, x_end=19, y_start=5, y_end=15,
                     x_centroid=17.5, y_centroid=10)
    spots.save_length()
    self.manager = FailureManager('test tracker owner', 'main')
    self.stop = Event()
    tool = VideoExtensoTool(
        spots, thresh=128, log_level=None, log_queue=None, white_spots=False,
        update_thresh=False, safe_mode=False, border=0, blur=None,
        failure_manager=self.manager, stop_event=self.stop,
        raise_on_lost_spot=raise_on_lost_spot)
    from unittest.mock import Mock
    self.manager._logger = Mock()

    def release():
      with self.manager.collect('test cleanup', raise_on_exit=False):
        with self.manager.step('release real trackers'):
          tool.stop_tracking()

    self.addCleanup(release)
    if with_peer:
      peer_ready = Event()
      created = []

      def make_tracker(**kwargs):
        if created:
          tracker = ReadyTracker(peer_ready=peer_ready, **kwargs)
        else:
          tracker = FaultTracker(fault=fault, peer_ready=peer_ready, **kwargs)
        created.append(tracker)
        return tracker

      factory = make_tracker
    else:
      factory = partial(FaultTracker, fault=fault)
    with patch.object(ve_module, 'Tracker', factory):
      tool.start_tracking()
    return tool

  def request_until_failure(self, tool):
    from time import monotonic, sleep
    deadline = monotonic() + 5
    while monotonic() < deadline:
      try:
        result = tool.get_data(np.zeros((20, 20), dtype=np.uint8))
      except BaseExceptionGroup as error:
        return error
      if isinstance(result, str):
        return result
      sleep(0.01)
    self.fail('Tracker did not return its result or failure')

  def test_stop_during_worker_startup(self):
    """Cold tracker imports must not turn an immediate stop into a failure."""

    tool = self.make_tool('healthy', with_peer=True)
    tool.stop_tracking()
    self.assertEqual(self.manager.snapshot(), ())
    self.assertEqual(tool._trackers, [])
    self.assertIsNone(tool._failure_thread)

  def test_programming_failure_is_not_misclassified_as_lost_spot(self):
    tool = self.make_tool('error')
    error = self.request_until_failure(tool)
    self.assertIsInstance(error, BaseExceptionGroup)
    leaves = exception_leaves(error)
    self.assertEqual(tuple(type(leaf) for leaf in leaves),
                     (ValueError, KeyboardInterrupt))
    self.assertIsInstance(leaves[0].__cause__, KeyError)
    self.assertIn('original tracker note', leaves[0].__notes__)
    tool.stop_tracking()
    self.assertEqual(len(self.manager.snapshot()), 1)
    report = self.manager.snapshot()[0][0]
    self.assertEqual(report.block_name, 'test tracker owner')
    self.assertEqual(report.phase, 'tracking')
    self.manager._logger.error.assert_not_called()
    self.assertTrue(self.stop.is_set())
    self.assertIsNone(tool._failure_thread)
    self.assertEqual(tool._failure_queues, [])

  def test_lost_spot_is_a_normal_result_and_not_a_failure_report(self):
    tool = self.make_tool('loss')
    result = self.request_until_failure(tool)
    self.assertEqual(result, 'expected spot loss')
    tool.stop_tracking()
    self.assertFalse(self.stop.is_set())
    self.assertEqual(self.manager.snapshot(), ())
    self.manager._logger.error.assert_not_called()

  def test_hard_exit_retains_the_process_identity(self):
    tool = self.make_tool('exit')
    tracker = tool._trackers[0]
    name, pid = tracker.name, tracker.pid
    error = self.request_until_failure(tool)
    self.assertIsInstance(error, ExceptionGroup)
    failure, = exception_leaves(error)
    self.assertIsInstance(failure, ProcessExitError)
    self.assertEqual((failure.process_name, failure.pid, failure.exitcode),
                     (name, pid, 9))
    self.assertEqual(len(self.manager.snapshot()), 1)
    tool.stop_tracking()
    self.assertEqual(len(self.manager.snapshot()), 1)

  def test_fatal_lost_spot_retains_original_tracker_ownership(self):
    tool = self.make_tool('loss', raise_on_lost_spot=True)
    tracker = tool._trackers[0]
    name, pid = tracker.name, tracker.pid
    failure = self.request_until_failure(tool)
    self.assertIsInstance(failure, ExceptionGroup)
    error, = exception_leaves(failure)
    self.assertIsInstance(error, LostSpotError)
    self.assertEqual(str(error), 'expected spot loss')
    report = self.manager.snapshot()[0][0]
    self.assertEqual((report.process_name, report.pid), (name, pid))
    self.assertIn('raise LostSpotError', report.formatted_traceback)
    tool.stop_tracking()
    self.assertEqual(len(self.manager.snapshot()), 1)
    self.manager._logger.error.assert_not_called()

  def test_one_dead_tracker_does_not_skip_healthy_peer_shutdown(self):
    tool = self.make_tool('exit', with_peer=True)
    healthy = tool._trackers[1]
    failure = self.request_until_failure(tool)
    error, = exception_leaves(failure)
    self.assertIsInstance(error, ProcessExitError)
    tool.stop_tracking()
    self.assertEqual(tool._trackers, [])
    self.assertEqual(tool._started_trackers, [])
    self.assertEqual(tool._pipes, [])
    self.assertEqual(tool._failure_queues, [])
    self.assertIsNone(tool._failure_thread)
    self.assertEqual(len(self.manager.snapshot()), 1)

  def test_healthy_tracker_stops_without_a_pipe_message(self):
    """A real idle tracker exits through its Event, not the image Pipe."""

    tool = self.make_tool('healthy')
    pipe = tool._pipes[0]
    tool._send(pipe, (5, 5, np.zeros((10, 10), dtype=np.uint8)))
    self.assertTrue(pipe.poll(5), 'Tracker did not process its first frame')
    self.assertIsInstance(pipe.recv(), Box)

    with patch.object(tool, '_send') as send:
      tool.stop_tracking()

    send.assert_not_called()
    self.assertTrue(tool._tracking_stop.is_set())
    self.assertFalse(self.stop.is_set())
    self.assertEqual(self.manager.snapshot(), ())
    self.assertEqual(tool._trackers, [])
    self.assertEqual(tool._failure_queues, [])
    self.assertIsNone(tool._failure_thread)

  def test_reporting_in_available_start_methods(self):
    from multiprocessing import get_all_start_methods
    import subprocess
    import sys

    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = ('from tests.unit.image_processing.test_video_extenso import '
                  f'exercise_tracker_context; exercise_tracker_context({method!r})')
        result = run_python(['-c', script], timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_tracker_context(method):
  """Runs real error, lost-spot and crash cases without sharing global state."""

  from multiprocessing import set_start_method
  import unittest
  set_start_method(method)
  names = ('test_stop_during_worker_startup',
           'test_programming_failure_is_not_misclassified_as_lost_spot',
           'test_lost_spot_is_a_normal_result_and_not_a_failure_report',
           'test_fatal_lost_spot_retains_original_tracker_ownership',
           'test_one_dead_tracker_does_not_skip_healthy_peer_shutdown',
           'test_healthy_tracker_stops_without_a_pipe_message',
           'test_hard_exit_retains_the_process_identity')
  suite = unittest.TestSuite(TestTrackerReporting(name) for name in names)
  if not unittest.TextTestRunner().run(suite).wasSuccessful():
    raise SystemExit(1)
