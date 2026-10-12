# coding: utf-8

"""Base Block ownership of reporting Queues and monitoring Threads."""

import gc
from inspect import signature
import logging
from multiprocessing import Event
from queue import Empty, Queue
import subprocess
import sys
from threading import Event as ThreadEvent, Thread, current_thread
from time import monotonic, sleep, time
from unittest.mock import Mock, patch
from weakref import ref

from crappy import Block
from crappy._exceptions import CrappyFail
from crappy._failure import FailureManager
from crappy.blocks.meta_block import block as block_module

from .block_test_base import BlockTestBase, TestBlock


class TestBlockFailureProducer(TestBlock):
  """Uses the new transport explicitly; automatic hook capture is Step 5."""

  def __init__(self, wait_finish=False):
    super().__init__()
    self.release_finish = Event()
    self._wait_finish = wait_finish

  def loop(self):
    assert self._failure_manager is not None
    self._failure_manager.capture_exception(ValueError('x' * 131072),
                                            phase='main')
    super().loop()

  def finish(self):
    super().finish()
    if self._wait_finish and not self.release_finish.wait(10):
      raise RuntimeError('test did not release finish')


class TestBlockBlockedFailureFlush(TestBlock):
  """Leaves the producer's public feeder join waiting for process shutdown."""

  def __init__(self):
    super().__init__()
    self.flush_entered = Event()

  def prepare(self):
    super().prepare()
    self._failure_queue.join_thread = self._wait_for_flush

  def _wait_for_flush(self):
    """Simulates a stuck feeder without touching Queue internals."""

    self.flush_entered.set()
    while True:
      sleep(0.1)


def _exercise_block_context(method):
  """Checks Block bootstrap and shutdown under an isolated global context."""

  from multiprocessing import set_start_method

  set_start_method(method)
  block = TestBlockFailureProducer()
  Block.prepare_all(log_level=logging.CRITICAL)
  manager, queue, thread = (Block.run_failure_manager,
                            block._failure_queue, Block.failure_thread)
  assert manager is not None and queue is not None and thread is not None
  assert block._failure_manager is None
  assert thread.is_alive()
  Block.launch_all(no_raise=True)
  assert block.finished.is_set()
  assert not block.is_alive()
  assert not thread.is_alive()
  assert block._failure_queue is None
  try:
    queue.get_nowait()
  except ValueError:
    pass
  else:
    raise AssertionError('parent Queue was not closed')
  assert len(manager.snapshot()) == 1
  assert manager.snapshot()[0][0].phase == 'main'
  assert Block.run_failure_manager is None


class TestFailureLifecycle(BlockTestBase):
  def test_failure_monitor_uses_class_owned_objects(self):
    """Only the monitor remains; it binds to the subclass without arguments."""

    self.assertIsInstance(vars(Block)['_failure_target'], classmethod)
    self.assertIs(TestBlock._failure_target.__self__, TestBlock)
    self.assertEqual(tuple(signature(Block._failure_target).parameters), ())
    self.assertNotIn('_start_failure_thread', vars(Block))
    self.assertNotIn('_close_failure_queue', vars(Block))
    self.assertIsInstance(vars(Block)['_stdout_filter'], staticmethod)

  def test_bootstrap_and_normal_shutdown(self):
    """Public closure releases the parent's Queue references after draining."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    manager, queue, thread = (Block.run_failure_manager,
                              self._block._failure_queue,
                              Block.failure_thread)
    assert manager is not None and queue is not None and thread is not None
    queue_ref = ref(queue)
    self.assertIsNone(self._block._failure_manager)
    self.assertEqual(Block.failure_queues, (queue,))
    self.assertTrue(thread.is_alive())
    Block.launch_all()
    self.assertFalse(thread.is_alive())
    with self.assertRaises(ValueError):
      queue.get_nowait()
    self.assertIsNone(self._block._failure_queue)
    self.assertEqual(manager.snapshot(), ())
    self.assertEqual(Block.failure_queues, ())
    self.assertIsNone(Block.failure_thread)
    self.assertIsNone(Block.run_failure_manager)
    del queue
    gc.collect()
    self.assertIsNone(queue_ref())

  def test_report_arrives_before_blocked_finish(self):
    """Keep reception active while a producer has not finished yet."""

    self._block = TestBlockFailureProducer(wait_finish=True)
    self.addCleanup(self._block.release_finish.set)
    Block.prepare_all(log_level=logging.CRITICAL)
    manager, thread = Block.run_failure_manager, Block.failure_thread
    Block.ready_barrier.wait(timeout=5)
    Block.shared_t0.value = time()
    Block.start_event.set()
    self.assertTrue(self.wait_until(lambda: bool(manager.snapshot())))
    self.assertTrue(self._block.finished.wait(3))
    self.assertTrue(self._block.is_alive())
    self.assertIsNone(manager.snapshot()[0][1])
    Block.no_raise = True
    self._block.release_finish.set()
    Block._cleanup()
    self.assertFalse(thread.is_alive())
    self.assertEqual(len(manager.snapshot()), 1)

  def test_receivers_start_after_the_fork_batch(self):
    """No reporting Thread is started before the last process."""

    self._block = TestBlock()
    other = TestBlock()
    real_start = TestBlock.start
    started = []

    def start_block(block):
      self.assertIsNone(Block.failure_thread)
      started.append(block)
      real_start(block)

    with patch.object(TestBlock, 'start', start_block):
      Block.prepare_all(log_level=logging.CRITICAL)
    self.assertEqual(set(started), {self._block, other})
    self.assertTrue(Block.failure_thread.is_alive())
    Block.launch_all()

  def test_partial_startup_drains_and_releases_allocated_queues(self):
    """A failed start batch still starts reception during bounded cleanup."""

    self._block = TestBlock()
    other = TestBlock()
    real_start = TestBlock.start
    started, queue_refs, threads = [], [], []

    def start_block(block):
      if started:
        raise RuntimeError('second process start failed')
      real_start(block)
      started.append(block)

    def create_thread(*args, **kwargs):
      thread = Thread(*args, **kwargs)
      if kwargs.get('name') == 'crappy.failure-reader':
        queue_refs.extend(ref(queue) for queue in Block.failure_queues)
        threads.append(thread)
      return thread

    with patch.object(TestBlock, 'start', start_block), \
         patch.object(block_module, 'Thread', side_effect=create_thread):
      with self.assertRaises(CrappyFail):
        Block.prepare_all(log_level=logging.CRITICAL)
    self.assertEqual(len(queue_refs), 2)
    self.assertFalse(other.is_alive())
    self.assertTrue(all(not thread.is_alive() for thread in threads))
    self.assertFalse(started[0].is_alive())
    self.assertIsNone(Block.run_failure_manager)
    self.assertIsNone(self._block._failure_queue)
    self.assertIsNone(other._failure_queue)
    gc.collect()
    self.assertTrue(all(queue_ref() is None for queue_ref in queue_refs))

  def test_final_drain_runs_even_when_stop_was_already_requested(self):
    """The class-owned Queue still gets its final drain after a stop."""

    source = FailureManager('Block', 'main')
    source._logger = Mock()
    queue = Queue()
    source.set_queue(queue)
    report = source.capture_exception(ValueError('last report'))
    manager = FailureManager('main', 'main')
    manager._logger = Mock()
    Block.run_failure_manager = manager
    Block.failure_queues = (queue,)
    Block.failure_thread_stop = ThreadEvent()
    Block.failure_thread_stop.set()
    Block.failure_thread = current_thread()
    Block._failure_target()
    self.assertEqual(manager.snapshot(), ((report, None),))
    manager._logger.error.assert_not_called()

  def test_monitor_failure_unblocks_existing_run_events(self):
    """A receive-loop failure still requests every independent stop action."""

    manager = FailureManager('main', 'bootstrap')
    manager._logger = Mock()
    Block.run_failure_manager = manager
    Block.failure_thread_stop = ThreadEvent()
    Block.failure_thread = current_thread()
    stop, failed, barrier = Mock(), Mock(), Mock()
    Block.stop_event = stop
    Block.raise_event = failed
    Block.ready_barrier = barrier
    with patch.object(manager, 'receive', side_effect=RuntimeError('monitor')):
      Block._failure_target()
    self.assertEqual(manager.snapshot()[0][0].phase, 'monitoring')
    stop.set.assert_called_once()
    failed.set.assert_called_once()
    barrier.abort.assert_called_once()

  def test_notification_failures_do_not_skip_other_stop_operations(self):
    """A failed stop signal does not skip the failure flag or Barrier."""

    manager = FailureManager('main', 'bootstrap')
    manager._logger = Mock()
    manager.capture_exception(ValueError('origin'))
    Block.run_failure_manager = manager
    Block.failure_thread_stop = ThreadEvent()
    Block.failure_thread = current_thread()
    stop, failed, barrier = Mock(), Mock(), Mock()
    stop.set.side_effect = OSError('stop')
    Block.stop_event = stop
    Block.raise_event = failed
    Block.ready_barrier = barrier
    Block._failure_target()
    self.assertEqual(len(manager.snapshot()), 2)
    failed.set.assert_called()
    barrier.abort.assert_called()

  def test_blocked_receiver_is_abandoned_without_touching_a_new_run(self):
    """Class-owned startup snapshots keep a blocked old receiver isolated."""

    manager = FailureManager('main', 'bootstrap')
    manager._logger = Mock()
    queue = Mock(spec_set=('get_nowait', 'close'))
    entered, release = ThreadEvent(), ThreadEvent()

    def receive():
      entered.set()
      release.wait(10)
      raise Empty

    queue.get_nowait.side_effect = receive
    Block.run_failure_manager = manager
    Block.failure_queues = (queue,)
    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.failure_thread_stop = ThreadEvent()
    Block.failure_thread = Thread(target=Block._failure_target, daemon=True)
    thread = Block.failure_thread
    thread.start()
    self.assertTrue(entered.wait(2))
    self.addCleanup(release.set)
    Block.no_raise = True
    started = monotonic()
    Block._cleanup()
    self.assertLess(monotonic() - started, 3)
    self.assertTrue(thread.is_alive())
    self.assertTrue(any('failure receiver did not stop' in
                        report.exception_message
                        for report, _ in manager.snapshot()))
    queue.close.assert_called_once()
    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    new_manager, new_stop = Block.run_failure_manager, Block.stop_event
    release.set()
    thread.join(2)
    self.assertFalse(thread.is_alive())
    self.assertFalse(new_stop.is_set())
    self.assertEqual(new_manager.snapshot(), ())
    Block.launch_all()

  def test_unstarted_old_receiver_does_not_attach_to_a_new_run(self):
    """A receiver scheduled after reset must not consume the new session."""

    old_thread = Thread(target=Block._failure_target, daemon=True)
    Block.failure_thread = old_thread
    Block.reset()
    manager = FailureManager('main', 'bootstrap')
    Block.run_failure_manager = manager
    Block.failure_thread_stop = ThreadEvent()
    new_thread = Thread(target=Mock(), daemon=True)
    Block.failure_thread = new_thread
    old_thread.start()
    old_thread.join(2)
    self.assertFalse(old_thread.is_alive())
    self.assertIs(Block.failure_thread, new_thread)
    self.assertFalse(Block.failure_thread_stop.is_set())
    self.assertEqual(manager.snapshot(), ())

  def test_blocked_feeder_join_is_bounded_by_process_shutdown(self):
    """The existing parent deadline bounds a child's direct public flush."""

    self._block = TestBlockBlockedFailureFlush()
    Block.prepare_all(log_level=logging.CRITICAL)
    started = monotonic()
    Block.launch_all()
    self.assertLess(monotonic() - started, 5)
    self.assertTrue(self._block.flush_entered.is_set())
    self.assertFalse(self._block.is_alive())
    self.assertNotEqual(self._block.exitcode, 0)
    self.assertIsNone(self._block._failure_queue)

  def test_close_and_flush_errors_remain_independent(self):
    """Inline producer shutdown attempts both public operations."""

    self._block = TestBlock()
    queue = Mock(spec_set=('close', 'join_thread', 'put'))
    close_error, flush_error = OSError('close'), KeyboardInterrupt('flush')
    queue.close.side_effect = close_error
    queue.join_thread.side_effect = flush_error
    self._block._failure_queue = queue
    self._block._logger = Mock()
    self._block._ready_barrier = Mock()
    self._block._start_event = Mock()
    self._block._start_event.wait.return_value = True
    self._block._stop_event = Mock()
    self._block._stop_event.is_set.return_value = True
    with patch.object(self._block, '_set_block_logger'):
      with self.assertRaises(BaseExceptionGroup) as caught:
        self._block.run()
    self.assertEqual(caught.exception.exceptions, (close_error, flush_error))
    queue.close.assert_called_once()
    queue.join_thread.assert_called_once()
    queue.put.assert_not_called()
    self.assertIsNone(self._block._failure_queue)
    self.assertEqual(len(self._block._failure_manager.snapshot()), 2)

  def test_parent_close_failure_does_not_skip_other_queues(self):
    """Receiving-only Queue copies only need the public close operation."""

    manager = FailureManager('main', 'finish')
    manager._logger = Mock()
    failed, good = (Mock(spec_set=('close',)) for _ in range(2))
    error = OSError('close failed')
    failed.close.side_effect = error
    Block.run_failure_manager = manager
    Block.failure_queues = (failed, good)
    Block.failure_thread = Mock(spec=Thread)
    Block.failure_thread.ident = None
    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._cleanup()
    failed.close.assert_called_once()
    good.close.assert_called_once()
    self.assertIs(manager.snapshot()[0][1], error)

  def test_block_lifecycle_in_available_start_methods(self):
    """Exercise class-owned monitoring and public shutdown in subprocesses."""

    from multiprocessing import get_all_start_methods
    script = ('import sys; from tests.block.test_12_failure_lifecycle import '
              '_exercise_block_context; _exercise_block_context(sys.argv[1])')
    for method in get_all_start_methods():
      with self.subTest(method=method):
        result = subprocess.run([sys.executable, '-c', script, method],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
