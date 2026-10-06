# coding: utf-8

import logging
from io import StringIO
from multiprocessing import Event
from queue import Empty, Queue
from threading import Event as ThreadEvent, Thread
from unittest.mock import Mock, patch

from crappy import Block
from crappy._global import CrappyFail
from crappy.blocks.meta_block import block as block_module

from .block_test_base import BlockTestBase


class TestLoggingLifecycle(BlockTestBase):
  """Checks logger shutdown, queue removal, and session isolation."""

  @staticmethod
  def _record(message: str = 'test') -> logging.LogRecord:
    return logging.LogRecord('crappy.test', logging.INFO, __file__, 0,
                             message, (), None)

  @staticmethod
  def _setup_cleanup() -> None:
    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.logger = Mock(spec=logging.Logger)

  def test_missing_queue_is_silent(self) -> None:
    """There is nothing to forward when the queue has been removed."""

    queue = Mock()
    Block.log_queue = None
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)

    queue.get.assert_not_called()
    get_logger.assert_not_called()

  def test_invalid_queue_argument_raises(self) -> None:
    """Passing None to the worker is an invalid call, unlike queue removal."""

    with self.assertRaisesRegex(RuntimeError, "The log queue doesn't exist"):
      Block._log_target(None)

  def test_stop_flag_prevents_reads(self) -> None:
    """An already stopped worker must not read queued messages."""

    queue = Mock()
    Block.log_queue = queue
    Block.thread_stop = True
    Block._log_target(queue)

    queue.get.assert_not_called()

  def test_stop_during_read_discards_record(self) -> None:
    """A record arriving during shutdown must not be forwarded."""

    record = self._record()
    queue = Mock()

    def stop_while_reading(*args, **kwargs):
      Block.thread_stop = True
      return record

    queue.get.side_effect = stop_while_reading
    Block.log_queue = queue
    with patch.object(logging, 'getLogger') as get_logger:
      Block._log_target(queue)

    queue.get.assert_called_once_with(block=True, timeout=0.05)
    get_logger.assert_not_called()

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
    get_logger.assert_not_called()

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
    new_queue.get.return_value = new_record
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
    new_queue.get.assert_called_once_with(block=True, timeout=0.05)

  def test_delayed_worker_start_cannot_attach_to_new_queue(self) -> None:
    """A worker constructed before reset must not attach to a later session."""

    old_queue = Mock()
    new_queue = Mock()
    Block.log_queue = old_queue
    worker = Thread(target=Block._log_target, args=(old_queue,), daemon=True)
    Block.reset()
    Block.log_queue = new_queue
    worker.start()
    worker.join(1.0)

    self.assertFalse(worker.is_alive())
    old_queue.get.assert_not_called()
    new_queue.get.assert_not_called()

  def test_cleanup_stops_logging_thread(self) -> None:
    """Both normal and error termination stop an idle logging worker."""

    for failing in (False, True):
      with self.subTest(failing=failing):
        self._setup_cleanup()
        if failing:
          Block.raise_event.set()
        Block.log_queue = Queue()
        worker = Thread(target=Block._log_target, args=(Block.log_queue,),
                         daemon=True)
        Block.log_thread = worker
        worker.start()
        try:
          if failing:
            with self.assertRaises(CrappyFail):
              Block._cleanup()
          else:
            Block._cleanup()
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
        Block._cleanup()
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
        failure = Block.raise_event
        logger = Block.logger
        worker = Mock(spec=Thread)
        worker.ident = 1
        worker.is_alive.return_value = True
        worker.join.side_effect = lambda **kwargs: self.assertTrue(
            Block.thread_stop)
        Block.log_thread = worker
        Block.no_raise = no_raise

        errors = StringIO()
        with patch.object(block_module, 'stderr', errors):
          if no_raise:
            Block._cleanup()
          else:
            with self.assertRaises(CrappyFail):
              Block._cleanup()

        worker.join.assert_called_once_with(timeout=1.0)
        self.assertTrue(failure.is_set())
        logger.exception.assert_not_called()
        self.assertIn('did not terminate', errors.getvalue())
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
      with patch.object(block_module, 'stderr', errors):
        with self.assertRaises(CrappyFail):
          Block._cleanup()
    finally:
      completed.set()
      release.set()
      worker.join(1.0)
      if guard.ident is not None:
        guard.join(1.0)

    self.assertFalse(timed_out.is_set(), "Cleanup blocked on the log handler")
    self.assertFalse(worker.is_alive())
    self.assertIn('did not terminate', errors.getvalue())

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

        expected = (KeyboardInterrupt if isinstance(error, KeyboardInterrupt)
                    else CrappyFail)
        with self.assertRaises(expected):
          Block._cleanup()

        worker.join.assert_called_once_with(timeout=1.0)

  def test_cleanup_skips_unstarted_thread(self) -> None:
    """With fork the logging Thread is created but never started."""

    self._setup_cleanup()
    worker = Mock(spec=Thread)
    worker.ident = None
    Block.log_thread = worker
    Block._cleanup()

    worker.join.assert_not_called()
