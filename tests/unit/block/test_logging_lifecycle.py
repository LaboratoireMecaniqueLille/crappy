# coding: utf-8

import logging
from io import StringIO
from multiprocessing import Event, Queue, get_all_start_methods
from pathlib import Path
from queue import Empty, Queue as LocalQueue
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Event as ThreadEvent, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crappy import Block
from crappy._failure import FailureManager
from crappy.blocks.meta_block import block as block_module
from crappy.blocks.camera_processes.camera_process import CameraProcess
from crappy.tool.camera_config.config_tools.histogram_process import HistogramProcess
from crappy.tool.image_processing.video_extenso.tracker import Tracker

from tests.support.blocks import BlockTestBase, TestBlock


class TestBlockLoggingProducer(TestBlock):
  """Exercises source capture now; automatic hook capture remains Step 5."""

  def loop(self) -> None:
    try:
      raise ValueError('original cause')
    except ValueError as cause:
      try:
        error = ExceptionGroup('outer', [TypeError('first leaf'),
                                        ExceptionGroup('nested', [cause])])
        error.add_note('failure note')
        raise error from cause
      except ExceptionGroup as error:
        self._failure_manager.capture_exception(error, phase='main')
    super().loop()

  def finish(self) -> None:
    # Enough data to require the queue reader to keep running while the
    # producer exits under spawn/forkserver. Fork logs through inherited sinks.
    for index in range(100):
      self.log(logging.ERROR, f'finished-record-{index:03d} ' + 'x' * 2048)
    super().finish()


def _exercise_logging_context(method: str) -> None:
  """Runs real logger/Queue lifecycle checks in an isolated interpreter."""

  from multiprocessing import set_start_method

  set_start_method(method)
  logger = logging.getLogger('crappy')
  application_handler = logging.NullHandler()
  logger.addHandler(application_handler)
  original_state = (logger.level, logger.disabled, logger.propagate)
  disable = logging.root.manager.disable
  cases = ((logging.DEBUG, True), (logging.CRITICAL, True),
           (logging.DEBUG, None), (None, True), (logging.DEBUG, True))
  for level, debug in cases:
    block = TestBlockLoggingProducer()
    block.debug = debug
    with (TemporaryDirectory() as directory,
          TemporaryFile(mode='w+') as console,
          TemporaryFile(mode='w+') as output,
          patch.object(block_module, 'system', return_value='Linux'),
          patch.object(block_module, 'Path', return_value=Path(directory)),
          patch.object(block_module, 'stdout', output),
          patch.object(block_module, 'stderr', console)):
      Block.prepare_all(log_level=level)
      manager = Block.run_failure_manager
      thread = Block.log_thread
      failure_thread = Block.failure_thread
      assert thread.is_alive() == (method != 'fork')
      assert (thread.ident is None) == (method == 'fork')
      assert failure_thread is not None and failure_thread.is_alive()
      handlers = tuple(Block.log_handlers)
      Block.launch_all(no_raise=True)
      assert block.finished.is_set()
      assert block.exitcode == 0, block.exitcode
      assert not thread.is_alive()
      assert (thread.ident is None) == (method == 'fork')
      assert not failure_thread.is_alive()
      assert block._log_queue is None
      assert Block.logger is None and not Block.log_handlers
      assert logger.handlers == [application_handler], logger.handlers
      assert (logger.level, logger.disabled, logger.propagate) == original_state
      assert logging.root.manager.disable == disable
      assert all(handler.stream is None for handler in handlers
                 if isinstance(handler, logging.FileHandler))
      reports = manager.snapshot()
      assert len(reports) == 1, reports
      report = reports[0][0]
      console.seek(0)
      displayed = console.read()
      if level == logging.DEBUG and debug is True:
        saved = (Path(directory) / 'logs.txt').read_text()
        assert str(report) in displayed and str(report) in saved
        assert 'failure note' in displayed and 'original cause' in displayed
        assert displayed.count(f'Failure {report.failure_id}:') == 1
        assert saved.count(f'Failure {report.failure_id}:') == 1
        line = next(line for line in displayed.splitlines()
                    if f'Failure {report.failure_id}:' in line)
        assert line in saved.splitlines()
        assert all(displayed.count(f'finished-record-{index:03d} ') == 1
                   for index in range(100))
        assert all(saved.count(f'finished-record-{index:03d} ') == 1
                   for index in range(100))
      else:
        assert not displayed, displayed
        if level is not None:
          saved = (Path(directory) / 'logs.txt').read_text()
          assert str(report) not in saved and 'finished-record-' not in saved
      # Final raising must not produce another runtime log.
      try:
        manager.raise_if_failed('collected failures')
      except ExceptionGroup:
        pass
      else:
        raise AssertionError('filtered reports were not retained')
      console.seek(0)
      assert console.read() == displayed
  logger.removeHandler(application_handler)
  application_handler.close()


class TestLoggingLifecycle(BlockTestBase):
  """Checks logger shutdown, queue removal, and session isolation."""

  def setUp(self) -> None:
    super().setUp()
    Block._ensure_failure_manager()
    Block.run_failure_manager._logger = Mock()

  @staticmethod
  def _record(message: str = 'test') -> logging.LogRecord:
    return logging.LogRecord('crappy.test', logging.INFO, __file__, 0,
                             message, (), None)

  @staticmethod
  def _setup_cleanup() -> None:
    Block.reset()
    Block.stop_event = Event()
    Block.logger = Mock(spec=logging.Logger)
    Block._ensure_failure_manager()
    Block.run_failure_manager._logger = Block.logger

  def test_missing_queue_is_silent(self) -> None:
    """There is nothing to forward when the queue has been removed."""

    queue = Mock()
    Block.log_queue = None
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)

    queue.get.assert_not_called()
    get_logger.assert_not_called()

  def test_queue_argument_is_required(self) -> None:
    """The worker receives its startup Queue explicitly."""

    with self.assertRaises(TypeError):
      Block._log_target()

  def test_stop_flag_drains_available_records(self) -> None:
    """Stopping consumes buffered records before the first empty read."""

    queue = Mock()
    record = self._record()
    queue.get.side_effect = [record, Empty]
    Block.log_queue = queue
    Block.thread_stop = True
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)
      get_logger.return_value.handle.assert_called_once_with(record)

    self.assertEqual(queue.get.call_count, 2)

  def test_stop_during_read_keeps_record(self) -> None:
    """A successful in-flight read still belongs to the shutdown drain."""

    record = self._record()
    queue = Mock()

    def stop_while_reading(*args, **kwargs):
      Block.thread_stop = True
      queue.get.side_effect = Empty
      return record

    queue.get.side_effect = stop_while_reading
    Block.log_queue = queue
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)

    self.assertEqual(queue.get.call_count, 2)
    get_logger.return_value.handle.assert_called_once_with(record)

  def test_reset_during_read_discards_record(self) -> None:
    """Reset must discard even a successful in-flight queue read."""

    record = self._record()
    queue = Mock()

    def reset_while_reading(*args, **kwargs):
      Block.reset()
      return record

    queue.get.side_effect = reset_while_reading
    Block.log_queue = queue
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)

    queue.get.assert_called_once_with(block=True, timeout=0.05)
    get_logger.return_value.handle.assert_not_called()

  def test_replaced_queue_discards_record(self) -> None:
    """An old worker must neither forward old data nor read the new queue."""

    old_queue = Mock()
    new_queue = Mock()
    record = self._record()

    def replace_while_reading(*args, **kwargs):
      Block.log_queue = new_queue
      return record

    old_queue.get.side_effect = replace_while_reading
    Block.log_queue = old_queue
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(old_queue)

    old_queue.get.assert_called_once_with(block=True, timeout=0.05)
    new_queue.get.assert_not_called()
    get_logger.assert_not_called()

  def test_replaced_queue_after_timeout_stops_worker(self) -> None:
    """An empty read must not let an old worker switch sessions either."""

    old_queue = Mock()
    new_queue = Mock()

    def replace_while_reading(*args, **kwargs):
      Block.log_queue = new_queue
      raise Empty

    old_queue.get.side_effect = replace_while_reading
    Block.log_queue = old_queue
    Block._log_target(old_queue)

    old_queue.get.assert_called_once_with(block=True, timeout=0.05)
    new_queue.get.assert_not_called()

  def test_restart_uses_only_new_queue(self) -> None:
    """A late old worker must not resume when a new session clears stop."""

    old_queue = Mock()
    old_record = self._record('old')
    new_queue = Mock()
    new_record = self._record('new')
    new_queue.get.side_effect = [new_record, Empty]
    reading = ThreadEvent()
    release = ThreadEvent()

    def delayed_read(*args, **kwargs):
      reading.set()
      release.wait(3.0)
      return old_record

    old_queue.get.side_effect = delayed_read
    Block.log_queue = old_queue
    worker = Thread(target=Block._log_target, args=(old_queue,), daemon=True)

    with patch.object(logging, 'getLogger') as get_logger:
      get_logger.return_value.handle.side_effect = (
          lambda record: setattr(Block, 'thread_stop', True))
      worker.start()
      try:
        self.assertTrue(reading.wait(1.0))
        Block.reset()
        Block.log_queue = new_queue
        Block._ensure_failure_manager()
        Block._log_target(new_queue)
        # Reproduce a restarted session's cleared stop flag while the old
        # worker is still finishing a read from the previous session.
        Block.thread_stop = False
      finally:
        release.set()
        worker.join(1.0)

      self.assertFalse(worker.is_alive())
      get_logger.return_value.handle.assert_called_once_with(new_record)

    old_queue.get.assert_called_once_with(block=True, timeout=0.05)
    self.assertEqual(new_queue.get.call_count, 2)

  def test_delayed_worker_start_cannot_attach_to_new_queue(self) -> None:
    """A worker constructed before reset must not attach to a later session."""

    old_queue = Mock()
    new_queue = Mock()
    Block.log_queue = old_queue
    worker = Thread(target=Block._log_target, args=(old_queue,), daemon=True)
    Block.reset()
    Block.log_queue = new_queue
    handler = Mock(spec=logging.Handler)
    Block.log_handlers = [handler]
    worker.start()
    worker.join(1.0)

    self.assertFalse(worker.is_alive())
    old_queue.get.assert_not_called()
    new_queue.get.assert_not_called()
    handler.flush.assert_not_called()
    handler.close.assert_not_called()

  def test_cleanup_stops_logging_thread(self) -> None:
    """Both normal and error termination stop an idle logging worker."""

    for failing in (False, True):
      with self.subTest(failing=failing):
        self._setup_cleanup()
        if failing:
          Block.run_failure_manager.capture_exception(ValueError('origin'))
        Block.log_queue = Queue()
        worker = Thread(target=Block._log_target, args=(Block.log_queue,),
                         daemon=True)
        Block.log_thread = worker
        worker.start()
        try:
          if failing:
            with self.assertRaises(ExceptionGroup):
              Block._finalize_run()
          else:
            Block._finalize_run()
          self.assertFalse(worker.is_alive())
        finally:
          Block.log_queue = None
          worker.join(1.0)

  def test_cleanup_allows_slow_handler_to_finish(self) -> None:
    """A handler taking over 0.1 seconds can still stop within the deadline."""

    self._setup_cleanup()
    Block.log_queue = Queue()
    Block.log_queue.put(self._record())
    handling = ThreadEvent()
    release = ThreadEvent()

    def handle(record) -> None:
      handling.set()
      release.wait(3.0)

    def release_after_delay() -> None:
      # Longer than the old join timeout, shorter than the new one.
      release.wait(0.2)
      release.set()

    worker = Thread(target=Block._log_target, args=(Block.log_queue,),
                     daemon=True)
    Block.log_thread = worker
    releaser = Thread(target=release_after_delay, daemon=True)
    with patch.object(logging, 'getLogger') as get_logger:
      get_logger.return_value.handle.side_effect = handle
      worker.start()
      try:
        self.assertTrue(handling.wait(1.0))
        releaser.start()
        Block._finalize_run()
        self.assertFalse(worker.is_alive())
      finally:
        release.set()
        worker.join(1.0)
        if releaser.ident is not None:
          releaser.join(1.0)

  def test_cleanup_reports_thread_stop_failure(self) -> None:
    """A still-live worker is an error, including with no_raise enabled."""

    for no_raise in (False, True):
      with self.subTest(no_raise=no_raise):
        self._setup_cleanup()
        logger = Block.logger
        worker = Mock(spec=Thread)
        worker.ident = 1
        worker.is_alive.return_value = True
        worker.join.side_effect = lambda **kwargs: self.assertTrue(
            Block.thread_stop)
        Block.log_thread = worker
        Block.no_raise = no_raise

        errors = StringIO()
        with patch('crappy._failure.sys.stderr', errors):
          with self.assertRaises(ExceptionGroup):
            Block._finalize_run()

        worker.join.assert_called_once_with(timeout=1.0)
        self.assertIsNone(Block.run_failure_manager)
        logger.exception.assert_not_called()
        self.assertIn('did not stop', errors.getvalue())
        self.assertFalse(any('terminated gracefully' in call.kwargs['msg']
                             for call in logger.log.call_args_list))

  def test_stop_failure_bypasses_locked_logging_handler(self) -> None:
    """Reporting a stuck handler must not acquire that handler's lock."""

    self._setup_cleanup()
    logger = logging.getLogger('crappy')
    # The pending INFO record is forwarded with Logger.handle, whereas the
    # main process's initial cleanup INFO messages are suppressed.
    logger.setLevel(logging.ERROR)
    Block.logger = logger
    handling = ThreadEvent()
    release = ThreadEvent()
    completed = ThreadEvent()
    timed_out = ThreadEvent()

    class StuckHandler(logging.Handler):
      def emit(self, record) -> None:
        handling.set()
        release.wait(3.0)

    logger.addHandler(StuckHandler())
    Block.log_queue = Queue()
    record = self._record()
    record.name = 'crappy'
    Block.log_queue.put(record)
    worker = Thread(target=Block._log_target, args=(Block.log_queue,),
                     daemon=True)
    Block.log_thread = worker

    def release_if_hung() -> None:
      if not completed.wait(2.0):
        timed_out.set()
        release.set()

    guard = Thread(target=release_if_hung, daemon=True)
    errors = StringIO()
    worker.start()
    try:
      self.assertTrue(handling.wait(1.0))
      guard.start()
      with patch('crappy._failure.sys.stderr', errors):
        with self.assertRaises(ExceptionGroup):
          Block._finalize_run()
    finally:
      completed.set()
      release.set()
      worker.join(1.0)
      if guard.ident is not None:
        guard.join(1.0)

    self.assertFalse(timed_out.is_set(), "Cleanup blocked on the log handler")
    self.assertFalse(worker.is_alive())
    self.assertIn('did not stop', errors.getvalue())

  def test_earlier_cleanup_failure_still_stops_thread(self) -> None:
    """Manager exceptions or interrupts must not skip logger shutdown."""

    for error in (ValueError('Manager failed'), KeyboardInterrupt()):
      with self.subTest(error=type(error)):
        self._setup_cleanup()
        worker = Mock(spec=Thread)
        worker.ident = 1
        worker.is_alive.return_value = False
        worker.join.side_effect = lambda **kwargs: self.assertTrue(
            Block.thread_stop)
        Block.log_thread = worker
        Block.shared_mgr = Mock()
        Block.shared_mgr.shutdown.side_effect = error

        expected = (BaseExceptionGroup if isinstance(error, KeyboardInterrupt)
                    else ExceptionGroup)
        with self.assertRaises(expected):
          Block._finalize_run()

        worker.join.assert_called_once_with(timeout=1.0)

  def test_cleanup_starts_unstarted_log_thread_only_for_queued_logging(self):
    """Partial startup keeps fork's inherited logging route unchanged."""

    for method in ('fork', 'spawn', 'forkserver'):
      with self.subTest(method=method):
        self._setup_cleanup()
        worker = Mock(spec=Thread)
        worker.ident = None
        worker.is_alive.return_value = False
        Block.log_thread = worker
        with patch.object(block_module, 'get_start_method', return_value=method):
          Block._finalize_run()
        if method == 'fork':
          worker.start.assert_not_called()
        else:
          worker.start.assert_called_once_with()
        worker.join.assert_not_called()

  def test_logging_off_is_scoped_and_restored(self) -> None:
    """Crappy opt-out does not suppress an independent application logger."""

    application = logging.getLogger('application.logging-test')
    application.setLevel(logging.ERROR)
    application.propagate = False
    handler = Mock(spec=logging.Handler)
    handler.level = logging.ERROR
    application.addHandler(handler)
    self.addCleanup(application.removeHandler, handler)
    disable = logging.root.manager.disable
    Block.log_level = None
    Block._set_logger()
    self.assertTrue(Block.logger.disabled)
    application.error('application still logs')
    handler.handle.assert_called_once()
    self.assertEqual(logging.root.manager.disable, disable)
    Block.reset()
    self.assertFalse(logging.getLogger('crappy').disabled)

  def test_worker_logging_off_is_scoped_and_reversible(self) -> None:
    """Every worker setup disables its own logger, not the interpreter."""

    setters = (Block._set_block_logger, CameraProcess._set_logger,
               Tracker._set_logger, HistogramProcess._set_logger)
    for setter in setters:
      for method in get_all_start_methods():
        with self.subTest(owner=setter.__qualname__, method=method):
          owner = SimpleNamespace(name='crappy.logging-owner-test',
                                  _log_level=None, _log_queue=Mock())
          logger = logging.getLogger(owner.name)
          disable = logging.root.manager.disable
          with patch(f'{setter.__module__}.get_start_method',
                     return_value=method):
            setter(owner)
            self.assertTrue(logger.disabled)
            self.assertEqual(logging.root.manager.disable, disable)
            owner._log_level = logging.CRITICAL
            setter(owner)
            self.assertFalse(logger.disabled)
            self.assertEqual(logger.level, logging.CRITICAL)
            self.assertEqual(logging.root.manager.disable, disable)
          for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

  def test_reader_failure_is_collected_and_signals_stop(self) -> None:
    """Both queue and handler errors reach the manager without escaping."""

    for operation in ('receive', 'handle'):
      with self.subTest(operation=operation):
        manager = FailureManager('main', 'bootstrap')
        manager._logger = Mock()
        Block.run_failure_manager = manager
        Block.stop_event = Event()
        Block.ready_barrier = Mock()
        queue = Mock()
        Block.log_queue = queue
        error = OSError(f'{operation} failed')
        queue.get.return_value = self._record()
        if operation == 'receive':
          queue.get.side_effect = error
        with patch.object(logging, 'getLogger') as get_logger:
          get_logger.return_value.handle.side_effect = error
          Block._log_target(queue)
        self.assertIs(manager.snapshot()[0][1], error)
        self.assertEqual(manager.snapshot()[0][0].phase, 'logging')
        manager._logger.error.assert_called_once()
        self.assertTrue(Block.stop_event.is_set())
        Block.ready_barrier.abort.assert_called_once_with()

  def test_broken_reader_logger_cannot_skip_stop_signals(self) -> None:
    """Emergency reporting does not recursively use the broken logger."""

    manager = FailureManager('main', 'bootstrap')
    logger = logging.getLogger('crappy.logging-broken-handler')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Mock(spec=logging.Handler)
    handler.level = logging.DEBUG
    handler.handle.side_effect = OSError('broken handler')
    logger.addHandler(handler)
    self.addCleanup(logger.removeHandler, handler)
    manager._logger = logger
    Block.run_failure_manager = manager
    Block.stop_event = Event()
    Block.ready_barrier = Mock()
    record = self._record()
    record.name = logger.name
    queue = LocalQueue()
    queue.put(record)
    Block.log_queue = queue
    with patch('crappy._failure.sys.stderr', StringIO()) as errors:
      Block._log_target(queue)
    self.assertTrue(Block.stop_event.is_set())
    self.assertTrue(manager.snapshot())
    self.assertIn('broken handler', errors.getvalue())
    self.assertLessEqual(handler.handle.call_count, 2)

  def test_handler_shutdown_failures_do_not_skip_peer_handlers(self) -> None:
    """Flush and close are independent, even with no_raise enabled."""

    for no_raise in (False, True):
      with self.subTest(no_raise=no_raise):
        self._setup_cleanup()
        manager = FailureManager('main', 'bootstrap')
        manager._logger = Mock()
        Block.run_failure_manager = manager
        Block.no_raise = no_raise
        first, second = Mock(spec=logging.Handler), Mock(spec=logging.Handler)
        first.flush.side_effect = OSError('flush failed')
        first.close.side_effect = ValueError('close failed')
        Block.log_handlers = [first, second]
        with self.assertRaises(ExceptionGroup):
          Block._finalize_run()
        first.close.assert_called_once_with()
        second.flush.assert_called_once_with()
        second.close.assert_called_once_with()
        self.assertEqual(len(manager.snapshot()), 2)
        self.assertIsNone(Block.logger)
        self.assertEqual(Block.log_handlers, [])

  def test_abandoned_reader_releases_only_its_old_handlers(self) -> None:
    """A late handler return cannot close resources from a subsequent run."""

    self._setup_cleanup()
    old_handler, new_handler = (Mock(spec=logging.Handler),
                                Mock(spec=logging.Handler))
    Block.log_handlers = [old_handler]
    queue = LocalQueue()
    queue.put(self._record())
    Block.log_queue = queue
    handling, release = ThreadEvent(), ThreadEvent()

    def handle(record) -> None:
      handling.set()
      release.wait(3.0)

    worker = Thread(target=Block._log_target, args=(queue,), daemon=True)
    Block.log_thread = worker
    with patch.object(logging, 'getLogger') as get_logger:
      get_logger.return_value.handle.side_effect = handle
      worker.start()
      try:
        self.assertTrue(handling.wait(1.0))
        Block.reset()
        old_handler.flush.assert_not_called()
        old_handler.close.assert_not_called()
        Block.log_handlers = [new_handler]
        Block.log_queue = LocalQueue()
      finally:
        release.set()
        worker.join(1.0)
    self.assertFalse(worker.is_alive())
    old_handler.flush.assert_called_once_with()
    old_handler.close.assert_called_once_with()
    new_handler.flush.assert_not_called()
    new_handler.close.assert_not_called()

  def test_real_logging_delivery_and_repeat_runs(self) -> None:
    """Full reports and late records survive each supported start method."""

    for method in get_all_start_methods():
      with self.subTest(method=method):
        script = ('from tests.unit.block.test_logging_lifecycle import '
                  '_exercise_logging_context; '
                  f'_exercise_logging_context({method!r})')
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
