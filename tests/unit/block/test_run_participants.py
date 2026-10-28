# coding: utf-8

from gc import collect
import logging
from unittest.mock import patch
from weakref import ref, WeakSet

from crappy import Block
from crappy.blocks.meta_block import block as block_module
from crappy.links import GraphStructureError, link_graph

from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock, link
from tests.support.assertions import exception_leaves


class TestRunParticipants(BlockTestBase):
  """Checks graph validation and ownership of the execution participants."""

  def test_missing_blocks_rejected_before_startup(self) -> None:
    """Lost ordinary Blocks, linked or isolated, must prevent startup."""

    self._block = TestBlock(retain=False)
    linked = TestBlock(retain=False)
    isolated = TestBlock(retain=False)
    link(self._block, linked)
    missing_names = sorted((linked.name, isolated.name))
    linked_ref, isolated_ref = ref(linked), ref(isolated)
    del linked, isolated
    collect()
    self.assertIsNone(linked_ref())
    self.assertIsNone(isolated_ref())

    with patch.object(Block, '_set_logger') as set_logger, \
         patch.object(block_module, 'Barrier') as barrier, \
         patch.object(TestBlock, 'start') as start:
      with self.assertRaises(ExceptionGroup) as raised:
        Block.prepare_all(log_level=logging.CRITICAL)

    self.assertIsInstance(exception_leaves(raised.exception)[0], GraphStructureError)
    self.assertIn(', '.join(missing_names), str(exception_leaves(raised.exception)[0]))
    self.assertIn('possibly garbage-collected', str(exception_leaves(raised.exception)[0]))
    set_logger.assert_called_once_with()
    barrier.assert_not_called()
    start.assert_not_called()
    self.assertFalse(Block.prepared_all)
    self.assertEqual(Block._run_blocks, ())
    self.assertFalse(link_graph.nodes)

  def test_all_blocks_lost_is_not_an_empty_experiment(self) -> None:
    """Validation must also run when the WeakSet is completely empty."""

    block = TestBlock(retain=False)
    name, block_ref = block.name, ref(block)
    del block
    collect()
    self.assertIsNone(block_ref())
    self.assertFalse(Block.instances)

    with self.assertRaises(ExceptionGroup) as raised:
      Block.prepare_all(log_level=logging.CRITICAL)

    self.assertIsInstance(exception_leaves(raised.exception)[0], GraphStructureError)
    self.assertIn(name, str(exception_leaves(raised.exception)[0]))
    self.assertIsNone(Block.ready_barrier)
    self.assertEqual(Block._run_blocks, ())

  def test_live_block_absent_from_graph_is_rejected(self) -> None:
    """A registry/graph mismatch in either direction is invalid."""

    self._block = TestBlock(retain=False)
    link_graph.nodes.pop(self._block.name)

    with self.assertRaises(ExceptionGroup) as raised:
      Block.prepare_all(log_level=logging.CRITICAL)

    self.assertIsInstance(exception_leaves(raised.exception)[0], GraphStructureError)
    self.assertIn('absent from the LinkGraph', str(exception_leaves(raised.exception)[0]))
    self.assertIn(self._block.name, str(exception_leaves(raised.exception)[0]))
    self.assertIsNone(Block.ready_barrier)

  def test_snapshot_retains_blocks_until_reset(self) -> None:
    """Dropping caller references during preparation cannot lose a Block."""

    holder = [TestBlock(retain=False)]
    block_ref = ref(holder[0])
    set_logger = Block._set_logger

    def drop_reference() -> None:
      holder.clear()
      collect()
      self.assertIsNotNone(block_ref())
      set_logger()

    # Suppress process startup so multiprocessing's own child registry cannot
    # keep the Block alive and hide a missing strong snapshot.
    with patch.object(Block, '_set_logger', side_effect=drop_reference), \
         patch.object(TestBlock, 'start') as start:
      Block.prepare_all(log_level=logging.CRITICAL)

    collect()
    self.assertIsNotNone(block_ref())
    self.assertEqual(Block._run_blocks, (block_ref(),))
    self.assertEqual(Block.ready_barrier.parties, 2)
    self.assertIs(block_ref()._ready_barrier, Block.ready_barrier)
    start.assert_called_once_with()

    Block._finalize_run()
    collect()
    self.assertEqual(Block._run_blocks, ())
    self.assertIsNone(block_ref())

  def test_early_setup_failure_releases_snapshot(self) -> None:
    """A failure before resources are allocated must not retain Blocks."""

    holder = [TestBlock(retain=False)]
    block_ref = ref(holder[0])

    def fail_setup() -> None:
      holder.clear()
      raise RuntimeError('Logger setup failed')

    with patch.object(Block, '_set_logger', side_effect=fail_setup):
      with self.assertRaises(ExceptionGroup) as caught:
        Block.prepare_all(log_level=logging.CRITICAL)

    self.assertIn('Logger setup failed', str(exception_leaves(caught.exception)[0]))
    collect()
    self.assertEqual(Block._run_blocks, ())
    self.assertIsNone(block_ref())
    self.assertFalse(Block.prepared_all)

  def test_process_start_failure_resets_snapshot(self) -> None:
    """Startup errors must release the snapshot through normal cleanup."""

    self._block = TestBlock(retain=False)
    with patch.object(TestBlock, 'start',
                      side_effect=RuntimeError('Process start failed')):
      with self.assertRaises(ExceptionGroup):
        Block.prepare_all(log_level=logging.CRITICAL)

    self.assertEqual(Block._run_blocks, ())
    self.assertFalse(Block.prepared_all)
    self.assertIsNone(Block.log_queue)

  def test_launch_uses_prepared_snapshot(self) -> None:
    """Watchdog, exit waiting and cleanup share the original participants."""

    self._block = TestBlock(retain=False)
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    blocks = Block._run_blocks
    watchdog_target = Block._watchdog_target
    real_wait = block_module.connection.wait

    def wait_for_blocks(objects, timeout=None):
      self.assertTrue(objects, 'Waiting must use the retained participants')
      return real_wait(objects, timeout=timeout)

    with patch.object(Block, 'instances', WeakSet()), \
         patch.object(Block, '_watchdog_target',
                      wraps=watchdog_target) as watchdog, \
         patch.object(block_module.connection, 'wait',
                      side_effect=wait_for_blocks):
      Block.launch_all()

    watchdog.assert_called_once_with()
    self.assertTrue(self._block.looped.is_set())
    self.assertFalse(self._block.is_alive())
    self.assertEqual(Block._run_blocks, ())

  def test_empty_experiment_does_not_wait_on_empty_sentinels(self) -> None:
    """An actually empty graph can complete without an indefinite exit wait."""

    Block.prepare_all(log_level=logging.CRITICAL)
    # Patch only Block's reference, not Queue's use of the same module.
    with patch.object(block_module, 'connection',
                      wraps=block_module.connection) as connection:
      Block.launch_all()

    connection.wait.assert_not_called()
    self.assertEqual(Block._run_blocks, ())
