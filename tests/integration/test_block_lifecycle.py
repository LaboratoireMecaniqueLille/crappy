"""Real Block workers: lifecycle phases, startup failure, time and stop behavior."""

import logging
from multiprocessing import Pipe

from crappy import Block
from crappy._exceptions import T0NotSetError
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.integration.scenarios.block_lifecycle import (
    TestBlockRaisePrepare, TestBlockRaiseBegin, TestBlockRaiseLoop,
    TestBlockRaiseSchedulerStop, TestBlockRaiseFinish, TestBlockT0)


class TestRunCycle(BlockTestBase):
  def test_run_closes_inherited_config_connections(self) -> None:
    """Tests that run closes stale configuration Pipe endpoints."""

    self._block = TestBlock()
    recv_conn, send_conn = Pipe(duplex=False)
    self.addCleanup(recv_conn.close)
    self.addCleanup(send_conn.close)
    self._block._config_connections_to_close.extend((recv_conn, send_conn))

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.run()

    self.assertTrue(recv_conn.closed)
    self.assertTrue(send_conn.closed)

  def test_normal_run(self) -> None:
    """Tests the nominal prepare/begin/loop/finish sequence."""

    self._block = TestBlock()

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    # The Block should stop on its own without reporting any error.
    self.assertFalse(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_raise_prepare(self) -> None:
    """Tests the behavior when prepare raises."""

    self._block = TestBlockRaisePrepare()

    self.configure_run(self._block, t0=0.0, start=False)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    # An exception during prepare should break the barrier and propagate to the
    # failure Queue.
    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_raise_begin(self) -> None:
    """Tests the behavior when begin raises."""

    self._block = TestBlockRaiseBegin()

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_raise_loop(self) -> None:
    """Tests the behavior when loop raises."""

    self._block = TestBlockRaiseLoop()

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_scheduler_stop(self) -> None:
    """SchedulerStop stops cleanly without marking the run as failed."""

    self._block = TestBlockRaiseSchedulerStop(stop=False)

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.start()
    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)
    self.assertFalse(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())
    self.assertEqual(self._block.loops.value, 1)

    Block.reset()

  def test_raise_finish(self) -> None:
    """Tests the behavior when the Block ends with an exception path."""

    self._block = TestBlockRaiseFinish()

    self.configure_run(self._block, t0=0.0, start=True)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_brake_barrier(self) -> None:
    """Tests the behavior when the ready barrier is already broken."""

    self._block = TestBlock()

    self.configure_run(self._block, t0=0.0, start=False)

    self._block._ready_barrier.abort()

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    self.assertFalse(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_no_start(self) -> None:
    """Tests the timeout path when the start Event is never set."""

    self._block = TestBlock()

    self.configure_run(self._block, t0=0.0, start=False)

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()


class TestBlockTime(BlockTestBase):
  def test_missing_t0(self) -> None:
    """Tests that reading t0 before it is set raises an error."""

    self._block = TestBlockT0(stop=False)

    self.configure_run(self._block, t0=-1.0, start=True)

    with self.assertRaises(T0NotSetError):
      _ = self._block.t0

    self._block.start()

    self._block.join(4.0)
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

    # Block.run should retain and send the error through its failure Queue.
    self.assertTrue(self.failures(self._block))

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_t0_normal(self) -> None:
    """Tests that t0 is available during a normal Block launch."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    manager = Block.run_failure_manager
    Block.launch_all()

    self.assertFalse(manager.snapshot())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertGreater(self._block.t0, 0)
