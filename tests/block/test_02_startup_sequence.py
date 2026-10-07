# coding: utf-8

from crappy import Block
from crappy._global import CrappyFail
from crappy.blocks.meta_block import block as block_module
from multiprocessing import synchronize, queues, get_start_method, Event
from multiprocessing.sharedctypes import Synchronized
from threading import Thread
from queue import Empty
from time import monotonic, sleep
from platform import system
import logging
import unittest
import signal
from unittest.mock import Mock, patch, call
from weakref import WeakSet

from .block_test_base import BlockTestBase, TestBlock


class TestBlockNoResponse(TestBlock):
  """Test Block that deliberately ignores the stop sequence for a while."""

  def loop(self) -> None:
    """Sleeps long enough for the cleanup code to have to intervene."""

    self.looped.set()
    sleep(10)


class TestBlockRaise(TestBlock):
  """Test Block raising after a peer has entered its unresponsive loop."""

  def __init__(self, wait_for_loop: synchronize.Event) -> None:
    super().__init__()
    self._wait_for_loop = wait_for_loop

  def loop(self) -> None:
    """Waits for the peer's loop to start before raising."""

    if not self._wait_for_loop.wait(3.0):
      raise RuntimeError("The peer Block never entered its loop")
    raise ValueError


class TestBlockIgnoreTerminate(TestBlockNoResponse):
  """POSIX child deliberately ignoring SIGTERM to require kill escalation."""

  def prepare(self) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    super().prepare()


class TestStartupSequence(BlockTestBase):
  """Tests the class methods driving the startup and shutdown sequence.

  These tests cover the interactions between prepare_all, renice_all,
  launch_all and _cleanup.
  """

  def test_prepare_all_prepared(self) -> None:
    """Tests that calling prepare_all twice fails cleanly."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # Manually break the current startup context before attempting the second
    # call, otherwise the first prepared Block would still be waiting
    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    with self.assertRaises(CrappyFail):
      Block.prepare_all(log_level=logging.CRITICAL)

  def test_prepare_all_launched(self) -> None:
    """Tests that prepare_all refuses an inconsistent launched state."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    Block.prepared_all = False
    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.prepare_all(log_level=logging.CRITICAL)

  def test_prepare_all_setup(self) -> None:
    """Tests that prepare_all configures all shared objects properly."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # The Block should have started and reached its prepare stage
    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(Block.prepared_all)
    self.assertFalse(Block.launched_all)
    self.assertEqual(Block._run_blocks, (self._block,))

    # The shared synchronization objects should be instantiated and left in
    # their initial state
    self.assertIsInstance(Block.ready_barrier, synchronize.Barrier)
    self.assertEqual(Block.ready_barrier.parties, len(Block._run_blocks) + 1)
    self.assertIsInstance(Block.shared_t0, Synchronized)
    self.assertEqual(Block.shared_t0.value, -1.0)

    for event in (Block.stop_event, Block.start_event, Block.pause_event,
                  Block.raise_event, Block.kbi_event):
      with self.subTest(event=event):
        self.assertIsNotNone(event)
        self.assertFalse(event.is_set())

    # Logging should also be configured at the class level
    self.assertIsInstance(Block.log_queue, queues.Queue)
    self.assertIsInstance(Block.log_thread, Thread)
    if get_start_method() != 'fork':
      self.assertTrue(Block.log_thread.is_alive())
    else:
      self.assertFalse(Block.log_thread.is_alive())

    # The Block instance must reference the exact same shared objects
    for cls, inst in zip((Block.stop_event, Block.start_event,
                          Block.pause_event, Block.raise_event,
                          Block.kbi_event, Block.ready_barrier,
                          Block.shared_t0, Block.log_queue),
                         (self._block._stop_event, self._block._start_event,
                          self._block._pause_event, self._block._raise_event,
                          self._block._kbi_event, self._block._ready_barrier,
                          self._block._instance_t0, self._block._log_queue)):
      self.assertIs(inst, cls)

    self.assertTrue(self._block.is_alive())
    self.assertFalse(Block.thread_stop)

    # Abort the barrier to force the cleanup path
    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))
    self.assertTrue(self.wait_until(lambda: not Block.log_thread.is_alive()))

    self.assertTrue(self._block.finished.wait(3.0))
    self._block.join(1.0)

    self.assertFalse(self._block.looped.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

    Block.reset()

  @unittest.skipIf(system() not in ('Linux', 'Darwin'),
                   "Test irrelevant on Windows")
  def test_renice_all(self) -> None:
    """Tests that renice_all applies the requested niceness."""

    self._block = TestBlock()
    self._block.niceness = 5

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # The unit under test is command construction. Mocking the system call
    # avoids depending on external ``ps``/``renice`` executables or permissions.
    with patch.object(Block, 'instances', WeakSet()), \
         patch.object(block_module.subprocess, 'call') as call:
      Block.renice_all(allow_root=False)

    call.assert_called_once_with(
        ['renice', '5', '-p', str(self._block.pid)],
        stdout=block_module.subprocess.DEVNULL)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block._stop_event.wait(3.0))

    Block.reset()

  def test_renice_all_not_prepared(self) -> None:
    """Tests that renice_all cannot be called before prepare."""

    self._block = TestBlock()

    with self.assertRaises(RuntimeError):
      Block.renice_all(allow_root=False)

    Block.reset()

  def test_renice_all_launched(self) -> None:
    """Tests that renice_all aborts on an inconsistent launched flag."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.renice_all(allow_root=False)

  def test_launch_all(self) -> None:
    """Tests the normal end-to-end startup sequence."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

    self.assertFalse(self._block.is_alive())

  def test_launch_all_no_prepared(self) -> None:
    """Tests that launch_all refuses to run before prepare."""

    self._block = TestBlock()

    with self.assertRaises(RuntimeError):
      Block.launch_all()

    Block.reset()

  def test_launch_all_launched(self) -> None:
    """Tests that launch_all aborts on a duplicated launch."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.launch_all()

  def test_stop_all(self) -> None:
    """Tests that stop_all can stop a running Crappy session."""

    def stop():
      """Stops the running session after a short delay."""

      if self._block.looped.wait(3.0):
        Block.stop_all()

    stop_thread = Thread(target=stop)

    self._block = TestBlock(stop=False)

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    stop_thread.start()

    Block.launch_all()

    self.assertFalse(self._block.is_alive())

  def test_restart(self) -> None:
    """Tests that a fresh Crappy session can start after a completed one."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

  def test_log_thread_stops_after_reset(self) -> None:
    """A logging worker must stop when reset discards its startup context."""

    queue = Mock()

    def reset_while_reading(*args, **kwargs):
      Block.reset()
      raise Empty

    queue.get.side_effect = reset_while_reading
    Block.log_queue = queue
    Block._log_target(queue)

    queue.get.assert_called_once_with(block=True, timeout=0.05)

  def test_cleanup(self) -> None:
    """Tests the different raise/no-raise combinations of _cleanup."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.raise_event.set()

    with self.assertRaises(CrappyFail):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.raise_event.set()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.kbi_event.set()

    with self.assertRaises(KeyboardInterrupt):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.kbi_event.set()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.raise_event.set()
    Block.kbi_event.set()

    with self.assertRaises(CrappyFail):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.raise_event.set()
    Block.kbi_event.set()

    Block._cleanup()

  def test_block_not_responding(self) -> None:
    """Tests that cleanup terminates a Block that does not stop by itself."""

    self._block = TestBlockNoResponse()
    peer = TestBlockRaise(self._block.looped)
    blocks = (self._block, peer)

    def reap_blocks() -> None:
      # Keep process references across Block.reset, including assertion failures.
      for inst in blocks:
        if inst.is_alive():
          inst.kill()
        if inst.pid is not None:
          inst.join(1.0)

    self.addCleanup(reap_blocks)
    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # Exercise the three-second forced-termination path without making the
    # unit test spend three wall-clock seconds in it.
    cleanup_start = monotonic()

    def accelerated_time() -> float:
      return cleanup_start + 100 * (monotonic() - cleanup_start)

    real_wait = block_module.connection.wait

    def accelerated_wait(objects, timeout=None):
      # Only accelerate the main grace-period wait, not Process.join internals.
      scaled = (timeout / 100
                if timeout is not None and isinstance(objects, dict) else timeout)
      return real_wait(objects, timeout=scaled)

    with patch.object(block_module, 'monotonic',
                      side_effect=accelerated_time), \
         patch.object(block_module.connection, 'wait',
                      side_effect=accelerated_wait), \
         patch.object(self._block, 'terminate',
                      wraps=self._block.terminate) as terminate:
      with self.assertRaises(CrappyFail):
        Block.launch_all()

      terminate.assert_called_once()

    self.assertTrue(self._block.looped.is_set())
    for inst in blocks:
      inst.join(1.0)
      self.assertFalse(inst.is_alive())

  def test_cleanup_waits_for_all_sentinels(self) -> None:
    """Ready Blocks are reaped without ending the wait for their peers."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    blocks = (Mock(spec=Block), Mock(spec=Block))
    Block._run_blocks = blocks
    for sentinel, block in enumerate(blocks, start=10):
      block.sentinel = sentinel
      block.is_alive.return_value = True
      block.join.side_effect = lambda *, timeout, block=block: setattr(
          block.is_alive, 'return_value', False)
    observed = list()
    remaining = list(blocks)

    def report_exit(objects, timeout):
      observed.append((set(objects), timeout))
      block = remaining.pop(0)
      # Sentinel readiness alone need not make is_alive() return False.
      self.assertTrue(block.is_alive())
      block.join.assert_not_called()
      return [block.sentinel]

    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(block_module.connection, 'wait', side_effect=report_exit):
      Block._cleanup()

    self.assertEqual(observed, [({10, 11}, 0.5), ({11}, 0.5)])
    for block in blocks:
      block.join.assert_called_once_with(timeout=3.0)
      block.terminate.assert_not_called()
      block.kill.assert_not_called()

  @unittest.skipIf(system() == 'Windows', 'SIGTERM handling is POSIX-only')
  def test_cleanup_kills_block_ignoring_terminate(self) -> None:
    """A real SIGTERM-ignoring child is killed and reaped before reset."""

    self._block = TestBlockIgnoreTerminate()
    peer = TestBlockRaise(self._block.looped)
    blocks = (self._block, peer)

    def reap_blocks():
      for inst in blocks:
        if inst.is_alive():
          inst.kill()
        if inst.pid is not None:
          inst.join(timeout=1.0)

    self.addCleanup(reap_blocks)
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(timeout=3.0))
    cleanup_start = monotonic()
    real_wait = block_module.connection.wait

    def accelerated_time():
      return cleanup_start + 100 * (monotonic() - cleanup_start)

    def accelerated_wait(objects, timeout=None):
      # Process.join must retain a real timeout long enough to reap the child.
      scaled = (timeout / 100
                if timeout is not None and isinstance(objects, dict) else timeout)
      return real_wait(objects, timeout=scaled)

    with (patch.object(block_module, 'monotonic', side_effect=accelerated_time),
          patch.object(block_module.connection, 'wait',
                       side_effect=accelerated_wait),
          patch.object(self._block, 'terminate',
                       wraps=self._block.terminate) as terminate,
          patch.object(self._block, 'kill', wraps=self._block.kill) as kill,
          self.assertRaises(CrappyFail)):
      Block.launch_all()

    terminate.assert_called_once()
    kill.assert_called_once()
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, -signal.SIGKILL)

  def test_cleanup_does_not_kill_after_successful_termination(self) -> None:
    """Give terminate() time to complete before considering kill()."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    block = Mock(spec=Block)
    block.sentinel = 10
    block.is_alive.return_value = True
    Block._run_blocks = (block,)

    def join(*, timeout):
      if block.terminate.called:
        block.is_alive.return_value = False

    block.join.side_effect = join
    with (patch.object(block_module, 'monotonic', return_value=0.0),
          patch.object(block_module.connection, 'wait', return_value=[10])):
      Block._cleanup()

    block.terminate.assert_called_once()
    block.kill.assert_not_called()
    self.assertEqual(block.join.call_args_list,
                     [call(timeout=3.0), call(timeout=1.0)])

  def test_cleanup_escalation_shares_deadlines(self) -> None:
    """Termination and kill stages each have one budget for all Blocks."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    blocks = (Mock(spec=Block), Mock(spec=Block))
    Block._run_blocks = blocks
    for sentinel, block in enumerate(blocks, start=10):
      block.sentinel = sentinel
      block.is_alive.return_value = True
      block.kill.side_effect = lambda block=block: setattr(
          block.is_alive, 'return_value', False)

    with (patch.object(block_module, 'monotonic',
                       side_effect=(0.0, 0.0, 0.0, 0.0,
                                    3.0, 3.1, 3.8, 4.0, 4.4, 4.9)),
          patch.object(block_module.connection, 'wait', return_value=[10, 11])):
      Block._cleanup()

    for block, terminate_timeout, kill_timeout in zip(blocks,
                                                     (0.9, 0.2), (0.6, 0.1)):
      block.terminate.assert_called_once()
      block.kill.assert_called_once()
      self.assertEqual(block.join.call_count, 3)
      self.assertAlmostEqual(block.join.call_args_list[1].kwargs['timeout'],
                             terminate_timeout)
      self.assertAlmostEqual(block.join.call_args_list[2].kwargs['timeout'],
                             kill_timeout)

  def test_cleanup_join_uses_remaining_deadline(self) -> None:
    """Reaping multiple ready Blocks must not restart the shutdown budget."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    blocks = (Mock(spec=Block), Mock(spec=Block))
    Block._run_blocks = blocks
    for sentinel, block in enumerate(blocks, start=10):
      block.sentinel = sentinel
      block.is_alive.return_value = True
      block.join.side_effect = lambda *, timeout, block=block: setattr(
          block.is_alive, 'return_value', False)

    with patch.object(block_module, 'monotonic',
                      side_effect=(0.0, 0.0, 1.0, 2.0)), \
         patch.object(block_module.connection, 'wait', return_value=[10, 11]):
      Block._cleanup()

    blocks[0].join.assert_called_once_with(timeout=2.0)
    blocks[1].join.assert_called_once_with(timeout=1.0)

  def test_cleanup_join_stays_positive_at_deadline(self) -> None:
    """A sentinel ready at the deadline still needs more than join(0)."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    block = Mock(spec=Block)
    block.sentinel = 10
    block.is_alive.return_value = True
    block.join.side_effect = lambda *, timeout: setattr(
        block.is_alive, 'return_value', False)
    Block._run_blocks = (block,)

    with patch.object(block_module, 'monotonic',
                      side_effect=(0.0, 2.9, 3.0)), \
         patch.object(block_module.connection, 'wait', return_value=[10]):
      Block._cleanup()

    block.join.assert_called_once_with(timeout=0.1)
    block.terminate.assert_not_called()

  def test_cleanup_does_not_reap_unstarted_blocks(self) -> None:
    """Partial startup must not access an unstarted Block's sentinel or join."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    self._block = TestBlock()
    started = Mock(spec=Block)
    started.sentinel = 10
    started.is_alive.return_value = True
    started.join.side_effect = lambda *, timeout: setattr(
        started.is_alive, 'return_value', False)
    Block._run_blocks = (started, self._block)

    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(block_module.connection, 'wait', return_value=[10]), \
         patch.object(self._block, 'join') as unstarted_join:
      Block._cleanup()

    started.join.assert_called_once_with(timeout=3.0)
    unstarted_join.assert_not_called()

  def test_cleanup_join_does_not_hide_a_still_running_block(self) -> None:
    """Joining is not itself proof that the Block actually stopped."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    failure = Block.raise_event
    block = Mock(spec=Block)
    block.name = 'crappy.still-running'
    block.sentinel = 10
    block.is_alive.return_value = True
    Block._run_blocks = (block,)

    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(block_module.connection, 'wait', return_value=[10]):
      with self.assertRaises(CrappyFail):
        Block._cleanup()

    self.assertEqual(block.join.call_args_list,
                     [call(timeout=3.0), call(timeout=1.0), call(timeout=1.0)])
    block.terminate.assert_called_once()
    block.kill.assert_called_once()
    self.assertTrue(failure.is_set())

  def test_cleanup_caps_wait_at_remaining_deadline(self) -> None:
    """The last sentinel wait must use only the remaining shutdown budget."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    block = Mock(spec=Block)
    block.sentinel = 10
    block.is_alive.return_value = True
    block.terminate.side_effect = lambda: setattr(
        block.is_alive, 'return_value', False)
    Block._run_blocks = (block,)

    with patch.object(block_module, 'monotonic',
                      side_effect=(0.0, 0.0, 2.8, 3.0, 3.0, 3.0)), \
         patch.object(block_module.connection, 'wait',
                      return_value=[]) as wait:
      Block._cleanup()

    self.assertEqual(wait.call_count, 2)
    self.assertAlmostEqual(wait.call_args_list[0].kwargs['timeout'], 0.5)
    self.assertAlmostEqual(wait.call_args_list[1].kwargs['timeout'], 0.2)
    block.terminate.assert_called_once()
