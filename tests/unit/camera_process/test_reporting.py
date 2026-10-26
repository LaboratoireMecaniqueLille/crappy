# coding: utf-8

"""Camera-owned failure delivery, including real workers and hard exits."""

from multiprocessing import Event, Queue, get_all_start_methods
import os
import subprocess
import sys
import numpy as np
from threading import Thread
from time import monotonic, sleep
from unittest.mock import Mock, patch

from crappy._exceptions import ProcessExitError, StopRequested
from crappy._failure import FailureManager
from crappy import Block
from crappy.blocks.camera import Camera
from crappy.blocks.camera_processes import Displayer, VideoExtensoProcess
from crappy.blocks.vision import CameraSource, VideoExtensoProcessor
from crappy.links import ImageLink
from crappy.tool.camera_config import CameraConfig
import crappy.blocks.camera as camera_module

from tests.support.camera_processes import CameraProcessTestBase, TestCameraProcess
from tests.unit.camera_process.test_camera_block import CameraBlockTestBase, TrackingCameraProcess
from tests.support.blocks import exception_leaves
from tests.support.vision import StubVisionBlock


class ReportingWorker(TestCameraProcess):
  """Raises an original grouped failure, or exits without Python cleanup."""

  def __init__(self, phase: str = 'main', hard_exit: bool = False,
               large_finish: bool = False) -> None:
    super().__init__(stop_on_loop=False)
    self.phase = phase
    self.hard_exit = hard_exit
    self.large_finish = large_finish
    self.release_finish = Event()

  def fail(self) -> None:
    if self.hard_exit:
      os._exit(7)
    cause = KeyError('original cause')
    error = ValueError('original worker error')
    error.__cause__ = cause
    error.add_note('original note')
    raise BaseExceptionGroup('worker group', [error, KeyboardInterrupt()])

  def init(self) -> None:
    super().init()
    if self.phase == 'init':
      self.fail()

  def loop(self) -> None:
    super().loop()
    if self.phase == 'main':
      self.fail()

  def finish(self) -> None:
    super().finish()
    if self.large_finish:
      if not self.release_finish.wait(5):
        raise RuntimeError('test did not release finish')
      raise OSError('finish payload ' + 'x' * 256_000)
    if self.phase == 'finish':
      self.fail()


def _image_generator(_, __) -> np.ndarray:
  """Picklable software source for nested Block/worker integration."""

  return np.ones((4, 5), dtype=np.uint8)


class ReportingCamera(Camera):
  """A complete Block owning a real faulting helper process."""

  def __init__(self, phase: str) -> None:
    super().__init__('unused', config=False, config_backend='tkinter',
                     image_generator=_image_generator, img_shape=(4, 5),
                     img_dtype='uint8', debug=None)
    self.phase = phase

  def prepare(self) -> None:
    self.process_proc = ReportingWorker(self.phase)
    super().prepare()


class ReportingSource(CameraSource):
  """Vision source failing after buffer setup or image publication."""

  def __init__(self, phase: str) -> None:
    super().__init__('unused', config=False, config_backend='tkinter',
                     image_generator=_image_generator, img_shape=(4, 5),
                     img_dtype='uint8', debug=None)
    self.phase = phase

  def prepare(self) -> None:
    super().prepare()
    if self.phase == 'prepare':
      raise ValueError('Vision preparation')

  def loop(self) -> None:
    super().loop()
    raise OSError('Vision acquisition')


class HeadlessHistogramConfig(CameraConfig):
  """Exercises configuration ownership without a GUI event loop."""

  def __init__(self, camera, log_queue, log_level, max_freq, transform,
               failure_manager):
    from multiprocessing import Event
    from crappy.tool.camera_config.base import ConfigurationLifecycle
    from crappy.tool.camera_config.config_tools import HistogramProcess

    super().__init__(camera, log_queue, log_level, max_freq, transform,
                     failure_manager)
    self.stop_event = Event()
    self.processing_event = Event()
    self.img_in, self.img_out, self.failure_queue = Queue(), Queue(), Queue()
    self.worker = HistogramProcess(
        self.stop_event, self.processing_event, self.img_in, self.img_out,
        self._log_level, self._log_queue, failure_queue=self.failure_queue,
        block_name=self._failure_manager.block_name,
        owner_path=f'{self._failure_manager.owner_path}.HistogramProcess')
    self._lifecycle = ConfigurationLifecycle(
        self.stop_event, self.worker, (self.img_in, self.img_out),
        self._failure_manager, self.failure_queue)

  def watch_shutdown(self, requested):
    self.shutdown_requested = requested

  def run(self):
    self.worker.start()
    self._lifecycle.mark_histogram_started()
    self.img_in.put(('invalid histogram request',))
    deadline = monotonic() + 5
    while not self._lifecycle.failed and monotonic() < deadline:
      sleep(0.01)
    if not self._lifecycle.failed:
      raise RuntimeError('test timed out waiting for histogram failure')
    self.stop()
    self._failure_manager.raise_if_failed('configuration failures')

  def stop(self):
    self._lifecycle.close_resources()


class HistogramCamera(Camera):
  configurator = HeadlessHistogramConfig

  def __init__(self):
    super().__init__('unused', config=True, config_backend='tkinter',
                     image_generator=_image_generator, img_shape=(4, 5),
                     img_dtype='uint8', debug=None)


def _tracking_spots():
  from crappy.tool.camera_config import SpotsBoxes, Box
  spots = SpotsBoxes()
  spots[0] = Box(x_start=0, x_end=4, y_start=0, y_end=4,
                 x_centroid=2, y_centroid=2)
  spots.save_length()
  return spots


class NestedTrackerProcess(VideoExtensoProcess):
  def init(self):
    from tests.unit.image_processing.test_video_extenso import FaultTracker
    from crappy.tool.image_processing.video_extenso import video_extenso
    self.set_config(_tracking_spots(), 128)
    with patch.object(video_extenso, 'Tracker', FaultTracker):
      super().init()


class TrackerCamera(Camera):
  def __init__(self):
    super().__init__('unused', config=False, config_backend='tkinter',
                     image_generator=_image_generator, img_shape=(4, 5),
                     img_dtype='uint8', debug=None)
    self.labels = ['t(s)', 'meta', 'Coo', 'Eyy', 'Exx']

  def prepare(self):
    self.process_proc = NestedTrackerProcess(
        False, 1, 1, None, False, False, 0, True)
    super().prepare()


class NestedVisionTracker(VideoExtensoProcessor):
  def request_config(self, source):
    return None

  def recv_configs(self):
    return {'source': (_tracking_spots(), 128)}

  def prepare(self):
    from tests.unit.image_processing.test_video_extenso import FaultTracker
    from crappy.tool.image_processing.video_extenso import video_extenso
    with patch.object(video_extenso, 'Tracker', FaultTracker):
      super().prepare()


class TestCameraReporting(CameraBlockTestBase):
  """Workers report through their Camera into the existing upstream manager."""

  def exercise_worker(self, phase: str, hard_exit: bool = False,
                      large_finish: bool = False,
                      with_peer: bool = False) -> None:
    camera = self.make_camera(save_images=with_peer)
    peer = None
    if with_peer:
      peer = ReportingWorker('idle')
      patcher = patch.object(camera_module, 'ImageSaver', return_value=peer)
      patcher.start()
      self.addCleanup(patcher.stop)
    worker = camera.process_proc = ReportingWorker(
        phase, hard_exit=hard_exit, large_finish=large_finish)
    upstream = Queue()
    self._queues.append(upstream)
    manager = camera._failure_manager = FailureManager(camera.name, 'prepare')
    manager._logger = Mock()
    manager.set_queue(upstream)
    root = FailureManager('coordinator', 'main')
    root._logger = Mock()
    self.addCleanup(worker.release_finish.set)
    if phase == 'init':
      # A bounded outer guard makes a lost readiness notification a test
      # failure instead of leaving the suite waiting at the local barrier.
      errors = []

      def prepare() -> None:
        try:
          camera.prepare()
        except BaseException as error:
          errors.append(error)

      thread = Thread(target=prepare, daemon=True)
      thread.start()
      thread.join(5)
      if thread.is_alive():
        camera._cam_barrier.abort()
        thread.join(2)
        self.fail('worker failure did not release Camera preparation')
      self.assertEqual(len(errors), 1)
      self.assertIsInstance(errors[0], StopRequested)
    else:
      camera.prepare()
      if phase == 'main':
        camera.loop()
        self.assertTrue(camera._stop_event.wait(5))
      else:
        camera._stop_event_cam.set()
        if large_finish:
          self.assertTrue(worker.finished.wait(5))
          worker.release_finish.set()

    name, pid = worker.name, worker.pid
    if hard_exit:
      with self.assertRaises(ExceptionGroup) as caught:
        camera.finish()
      self.assertIsInstance(exception_leaves(caught.exception)[0],
                            ProcessExitError)
    else:
      camera.finish()
    deadline = monotonic() + 3
    while not root.snapshot() and monotonic() < deadline:
      root.receive((upstream,))
      sleep(0.01)
    self.assertEqual(len(root.snapshot()), 1)
    report = root.snapshot()[0][0]
    if not hard_exit:
      self.assertEqual(report.block_name, camera.name)
      self.assertEqual(report.process_name, name)
      self.assertEqual(report.pid, pid)
      self.assertEqual(report.owner_path, name)
      self.assertEqual(report.phase, 'finish' if large_finish else phase)
      manager._logger.error.assert_not_called()
    with self.assertRaises(BaseExceptionGroup) as caught:
      root.raise_if_failed('camera failures', group_by_source=True)
    leaves = exception_leaves(caught.exception)
    if hard_exit:
      self.assertEqual(len(leaves), 1)
      self.assertIsInstance(leaves[0], ProcessExitError)
      self.assertEqual(leaves[0].process_name, name)
      self.assertEqual(leaves[0].pid, pid)
      self.assertEqual(leaves[0].exitcode, 7)
      self.assertEqual(leaves[0].phase, phase)
    elif large_finish:
      self.assertIsInstance(leaves[0], OSError)
      self.assertGreater(len(str(leaves[0])), 256_000)
    else:
      self.assertEqual(tuple(type(error) for error in leaves),
                       (ValueError, KeyboardInterrupt))
      self.assertIsInstance(leaves[0].__cause__, KeyError)
      self.assertIn('original note', leaves[0].__notes__)
    root._logger.error.assert_not_called()
    self.assertIsNone(camera._worker_failure_thread)
    self.assertEqual(camera._worker_failure_queues, {})
    self.assertIsNone(camera._manager)
    if peer is not None:
      self.assertTrue(peer.finished.is_set())
      self.assertFalse(peer.looped.is_set())
    camera.finish()
    self.assertEqual(len(manager.snapshot()), 1)

  def test_worker_failure_during_init(self) -> None:
    self.exercise_worker('init')

  def test_worker_failure_during_main(self) -> None:
    self.exercise_worker('main')

  def test_worker_failure_during_finish(self) -> None:
    self.exercise_worker('finish')

  def test_hard_exit_during_init(self) -> None:
    self.exercise_worker('init', hard_exit=True, with_peer=True)

  def test_hard_exit_during_main(self) -> None:
    self.exercise_worker('main', hard_exit=True)

  def test_finish_report_is_drained_while_worker_flushes(self) -> None:
    self.exercise_worker('finish', large_finish=True)

  def test_receiver_failure_notifies_and_preserves_cleanup(self) -> None:
    camera = self.make_camera()
    worker = camera.process_proc = TrackingCameraProcess()
    camera.prepare()
    manager = camera._failure_manager
    manager._logger = Mock()
    error = OSError('report reception')
    queue = camera._worker_failure_queues[worker]
    with patch.object(queue, 'get_nowait', side_effect=error):
      self.assertTrue(camera._stop_event.wait(3))
      camera.finish()
    self.assertTrue(worker.closed)
    self.assertIsNone(camera._worker_failure_thread)
    self.assertEqual(camera._worker_failure_queues, {})
    self.assertIs(manager.snapshot()[0][1], error)
    manager._logger.error.assert_called_once()

  def test_real_reporting_in_all_start_methods(self) -> None:
    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = ('from tests.unit.camera_process.test_reporting import '
                  f'_exercise_context; _exercise_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_public_camera_and_vision_runs_in_all_start_methods(self) -> None:
    for method in get_all_start_methods():
      with self.subTest(method=method):
        if sys.platform == 'darwin' and method == 'fork':
          self.skipTest('fork is unsafe with macOS native libraries '
                        '(bpo-33725)')
        script = ('from tests.unit.camera_process.test_reporting import '
                  f'_exercise_public_context; _exercise_public_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


  def test_nested_configuration_and_trackers_reach_public_boundary(self):
    for method in get_all_start_methods():
      with self.subTest(method=method):
        if sys.platform == 'darwin' and method == 'fork':
          self.skipTest('fork is unsafe with macOS native libraries '
                        '(bpo-33725)')
        script = ('from tests.unit.camera_process.test_reporting import '
                  f'_exercise_nested_context; _exercise_nested_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script], text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _exercise_context(method: str) -> None:
  """Runs worker ownership checks in an isolated multiprocessing context."""

  from multiprocessing import set_start_method
  import unittest

  set_start_method(method)
  names = ('test_worker_failure_during_init', 'test_worker_failure_during_main',
           'test_worker_failure_during_finish', 'test_hard_exit_during_init',
           'test_hard_exit_during_main',
           'test_finish_report_is_drained_while_worker_flushes')
  suite = unittest.TestSuite(TestCameraReporting(name) for name in names)
  result = unittest.TextTestRunner().run(suite)
  if not result.wasSuccessful():
    raise SystemExit(1)


def _exercise_public_context(method: str) -> None:
  """Exercises both architectures through the public orchestration boundary."""

  from multiprocessing import set_start_method

  set_start_method(method)
  for vision, phase in ((False, 'init'), (False, 'main'),
                         (True, 'prepare'), (True, 'main')):
    if vision:
      source = ReportingSource(phase)
      consumer = StubVisionBlock(debug=None)
      ImageLink(source, consumer)
    else:
      source = ReportingCamera(phase)
    try:
      Block.start_all(log_level=None)
    except BaseExceptionGroup as error:
      leaves = exception_leaves(error)
      expected = ((ValueError if phase == 'prepare' else OSError,)
                  if vision else (ValueError, KeyboardInterrupt))
      assert tuple(type(leaf) for leaf in leaves) == expected, error
      if not vision:
        assert isinstance(leaves[0].__cause__, KeyError)
    else:
      raise AssertionError('Camera failure did not reach the public boundary')
    assert source.exitcode == 0, source.exitcode
    assert not Block.instances


class TestReportingBoundaries(CameraProcessTestBase):
  """Independent logging, reporting and notification failures are retained."""

  def test_worker_bootstrap_failure_still_finishes_and_reports(self) -> None:
    self._process = TestCameraProcess()
    shared = self.make_shared()
    error = OSError('logger setup failed')
    with patch.object(self._process, '_set_logger', side_effect=error):
      self._process.run()
    self.assertTrue(shared.stop_event.is_set())
    self.assertTrue(shared.barrier.broken)
    self.assertTrue(self._process.finished.is_set())
    self.assertIs(self._process._failure_manager.snapshot()[0][1], error)
    self.assertEqual(self._process._failure_manager.snapshot()[0][0].phase,
                     'bootstrap')

  def test_reporting_and_queue_cleanup_faults_cannot_skip_finish(self) -> None:
    self._process = TestCameraProcess(raise_in='init')
    shared = self.make_shared()
    queue = self._process._failure_queue = Mock()
    send, close, flush = (OSError('send'), RuntimeError('close'),
                          KeyboardInterrupt('flush'))
    queue.put.side_effect = send
    queue.close.side_effect = close
    queue.join_thread.side_effect = flush
    logger = Mock()
    self._process._logger = logger
    with patch.object(self._process, '_set_logger'):
      self._process.run()
    retained = [error for _, error in self._process._failure_manager.snapshot()]
    for error in (send, close, flush):
      self.assertIn(error, retained)
    queue.close.assert_called_once()
    queue.join_thread.assert_called_once()
    self.assertTrue(self._process.finished.is_set())
    self.assertTrue(shared.barrier.broken)
    self.assertTrue(shared.stop_event.is_set())
    self.assertIsNone(self._process._failure_manager._queue)

  def test_source_reports_before_cleanup_without_logging_reraised_error(self
                                                                       ) -> None:
    self._process = TestCameraProcess()
    self.make_shared()
    logger = self._process._logger = Mock()
    queue = self._process._failure_queue = Mock()
    origin = ValueError('source failure')

    def finish() -> None:
      logger.error.assert_called_once()
      queue.put.assert_called_once()
      raise origin

    with (patch.object(self._process, '_set_logger'),
          patch.object(self._process, 'init', side_effect=origin),
          patch.object(self._process, 'finish', side_effect=finish)):
      self._process.run()
    logger.error.assert_called_once()
    queue.put.assert_called_once()
    self.assertEqual(len(self._process._failure_manager.snapshot()), 1)
    queue.close.assert_called_once()
    queue.join_thread.assert_called_once()

  def test_overlay_active_eof_is_reported_without_thread_traceback(self) -> None:
    worker = Displayer('test', 30, backend='cv2')
    self.make_shared(process=worker)
    manager = worker._failure_manager = FailureManager(worker.name, 'main')
    manager._logger = Mock()
    error = EOFError('active overlay EOF')
    worker._to_draw_conn = Mock()
    worker._to_draw_conn.poll.side_effect = error
    worker._thread_target()
    self.assertTrue(worker._stop_event.is_set())
    self.assertTrue(worker._cam_barrier.broken)
    self.assertIs(manager.snapshot()[0][1], error)
    manager._logger.error.assert_called_once()

  def test_overlay_shutdown_eof_is_not_a_failure(self) -> None:
    worker = Displayer('test', 30, backend='cv2')
    self.make_shared(process=worker)
    manager = worker._failure_manager = FailureManager(worker.name, 'main')
    manager._logger = Mock()
    connection = worker._to_draw_conn = Mock()

    def close_during_poll() -> None:
      worker._stop_event.set()
      raise EOFError('closed during shutdown')

    connection.poll.side_effect = close_during_poll
    worker._thread_target()
    self.assertEqual(manager.snapshot(), ())
    manager._logger.error.assert_not_called()

  def test_overlay_failure_cannot_skip_other_notifications(self) -> None:
    worker = Displayer('test', 30, backend='cv2')
    self.make_shared(process=worker)
    manager = worker._failure_manager = FailureManager(worker.name, 'main')
    manager._logger = Mock()
    origin, stop_error = ValueError('overlay'), OSError('stop notification')
    worker._to_draw_conn = Mock()
    worker._to_draw_conn.poll.side_effect = origin
    worker._stop_event = Mock()
    worker._stop_event.is_set.return_value = False
    worker._stop_event.set.side_effect = stop_error
    worker._thread_target()
    self.assertEqual(tuple(error for _, error in manager.snapshot()),
                     (origin, stop_error))
    self.assertTrue(worker._cam_barrier.broken)
    self.assertTrue(worker._stop_thread)


def _exercise_nested_context(method):
  """Checks all nested report channels through start_all(), with logging off."""

  from multiprocessing import set_start_method
  set_start_method(method)
  # Repeat the fast-exit case to expose races between status checks and cleanup.
  for kind in ('histogram',) * 5 + ('legacy tracker', 'vision tracker'):
    if kind == 'histogram':
      source = HistogramCamera()
    elif kind == 'legacy tracker':
      source = TrackerCamera()
    else:
      source = CameraSource('unused', config=False, config_backend='tkinter',
                            image_generator=_image_generator, img_shape=(4, 5),
                            img_dtype='uint8', debug=None)
      processor = NestedVisionTracker(debug=None, blur=None)
      ImageLink(source, processor)
    try:
      Block.start_all(log_level=None)
    except BaseExceptionGroup as error:
      leaves = exception_leaves(error)
      expected = (ValueError,) if kind == 'histogram' else (
          ValueError, KeyboardInterrupt)
      assert tuple(type(leaf) for leaf in leaves) == expected, error
      if kind != 'histogram':
        assert isinstance(leaves[0].__cause__, KeyError)
    else:
      raise AssertionError('Nested worker failure did not reach start_all()')
    assert source.exitcode == 0, source.exitcode
    assert not Block.instances
