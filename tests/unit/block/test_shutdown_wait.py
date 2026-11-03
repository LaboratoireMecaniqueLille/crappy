"""Bounded shutdown waits, joins and escalation with controlled collaborators."""

from multiprocessing import Event
from unittest.mock import Mock, patch
from crappy import Block
from crappy.blocks.meta_block import block as block_module
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.clock import Clock


class TestShutdownWait(BlockTestBase):
  def test_cleanup_collects_without_resetting_or_raising(self) -> None:
    """Cleanup keeps the run context available for finalization."""

    Block.stop_event = Event()
    Block._set_logger()
    Block._ensure_failure_manager()
    manager = Block.run_failure_manager
    error = ValueError('origin')
    manager.capture_exception(error)
    Block._cleanup()
    self.assertIs(Block.run_failure_manager, manager)
    self.assertIsNotNone(Block.stop_event)
    self.assertTrue(Block.stop_event.is_set())
    with self.assertRaises(ExceptionGroup):
      Block._finalize_run()
    self.assertIsNone(Block.run_failure_manager)

  def test_cleanup_waits_for_all_sentinels(self) -> None:
    """Ready Blocks are reaped without ending the wait for their peers."""

    Block.stop_event = Event()
    blocks = (Mock(spec=Block, inputs=[], outputs=[]),
              Mock(spec=Block, inputs=[], outputs=[]))
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

    self.assertEqual([sentinels for sentinels, _ in observed], [{10, 11}, {11}])
    self.assertTrue(all(0 < timeout <= 10 for _, timeout in observed))
    for block in blocks:
      self.assertTrue(block.join.called)
      self.assertGreater(block.join.call_args.kwargs['timeout'], 0)
      block.terminate.assert_not_called()
      block.kill.assert_not_called()


  def test_cleanup_does_not_kill_after_successful_termination(self) -> None:
    """Give terminate() time to complete before considering kill()."""

    Block.stop_event = Event()
    block = Mock(spec=Block, inputs=[], outputs=[])
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
    budgets = [entry.kwargs['timeout'] for entry in block.join.call_args_list]
    self.assertGreaterEqual(len(budgets), 2)
    self.assertTrue(all(timeout > 0 for timeout in budgets))

  def test_cleanup_escalation_shares_deadlines(self) -> None:
    """Time spent reaping a peer reduces each escalation stage's budget."""

    Block.stop_event = Event()
    clock = Clock()
    blocks = (Mock(spec=Block, inputs=[], outputs=[]),
              Mock(spec=Block, inputs=[], outputs=[]))
    Block._run_blocks = blocks
    observed = {'normal': [], 'terminate': [], 'kill': []}
    for sentinel, block in enumerate(blocks, start=10):
      block.sentinel = sentinel
      block.is_alive.return_value = True
      block.kill.side_effect = lambda block=block: setattr(
          block.is_alive, 'return_value', False)

      def join(*, timeout, block=block):
        stage = ('kill' if block.kill.called else
                 'terminate' if block.terminate.called else 'normal')
        observed[stage].append(timeout)
        clock.advance(0.2)

      block.join.side_effect = join

    with (patch.object(block_module, 'monotonic', clock),
          patch.object(block_module.connection, 'wait', return_value=[10, 11])):
      Block._cleanup()

    for block in blocks:
      block.terminate.assert_called_once()
      block.kill.assert_called_once()
    for stage, budgets in observed.items():
      with self.subTest(stage=stage):
        self.assertEqual(len(budgets), 2)
        self.assertGreater(budgets[0], budgets[1])
        self.assertGreater(budgets[1], 0)
        self.assertAlmostEqual(budgets[0] - budgets[1], 0.2)

  def test_cleanup_join_uses_remaining_deadline(self) -> None:
    """Waiting and joining peers consume one shared shutdown budget."""

    Block.stop_event = Event()
    clock = Clock()
    blocks = (Mock(spec=Block, inputs=[], outputs=[]),
              Mock(spec=Block, inputs=[], outputs=[]))
    Block._run_blocks = blocks
    for sentinel, block in enumerate(blocks, start=10):
      block.sentinel = sentinel
      block.is_alive.return_value = True

      def join(*, timeout, block=block):
        self.assertGreater(timeout, 0)
        clock.advance(1)
        block.is_alive.return_value = False

      block.join.side_effect = join

    def ready(*args, **kwargs):
      clock.advance(1)
      return [10, 11]

    with (patch.object(block_module, 'monotonic', clock),
          patch.object(block_module.connection, 'wait', side_effect=ready)):
      Block._cleanup()

    budgets = [block.join.call_args.kwargs['timeout'] for block in blocks]
    self.assertGreater(budgets[0], budgets[1])
    self.assertAlmostEqual(budgets[0] - budgets[1], 1)
    for block in blocks:
      block.terminate.assert_not_called()

  def test_cleanup_join_stays_positive_at_deadline(self) -> None:
    """Sentinel readiness at the deadline still allows the process to reap."""

    Block.stop_event = Event()
    clock = Clock()
    block = Mock(spec=Block, inputs=[], outputs=[])
    block.sentinel = 10
    block.is_alive.return_value = True
    block.join.side_effect = lambda *, timeout: setattr(
        block.is_alive, 'return_value', False)
    Block._run_blocks = (block,)

    def ready(*args, **kwargs):
      clock.advance(10)  # Reach the documented normal shutdown deadline.
      return [10]

    with (patch.object(block_module, 'monotonic', clock),
          patch.object(block_module.connection, 'wait', side_effect=ready)):
      Block._cleanup()

    self.assertGreater(block.join.call_args.kwargs['timeout'], 0)
    block.terminate.assert_not_called()

  def test_cleanup_does_not_reap_unstarted_blocks(self) -> None:
    """Partial startup must not access an unstarted Block's sentinel or join."""

    Block.stop_event = Event()
    self._block = TestBlock()
    started = Mock(spec=Block, inputs=[], outputs=[])
    started.sentinel = 10
    started.is_alive.return_value = True
    started.join.side_effect = lambda *, timeout: setattr(
        started.is_alive, 'return_value', False)
    Block._run_blocks = (started, self._block)

    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(block_module.connection, 'wait', return_value=[10]), \
         patch.object(self._block, 'join') as unstarted_join:
      Block._cleanup()

    self.assertTrue(started.join.called)
    self.assertGreater(started.join.call_args.kwargs['timeout'], 0)
    unstarted_join.assert_not_called()

  def test_cleanup_join_does_not_hide_a_still_running_block(self) -> None:
    """Joining is not itself proof that the Block actually stopped."""

    Block.stop_event = Event()
    block = Mock(spec=Block, inputs=[], outputs=[])
    block.name = 'crappy.still-running'
    block.sentinel = 10
    block.is_alive.return_value = True
    Block._run_blocks = (block,)

    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(block_module.connection, 'wait', return_value=[10]):
      Block._cleanup()

    self.assertTrue(block.join.called)
    block.terminate.assert_called_once()
    block.kill.assert_called_once()
    self.assertTrue(Block.run_failure_manager.snapshot())

  def test_cleanup_caps_wait_at_remaining_deadline(self) -> None:
    """The final sentinel poll cannot overrun the remaining normal budget."""

    Block.stop_event = Event()
    clock = Clock()
    block = Mock(spec=Block, inputs=[], outputs=[])
    block.sentinel = 10
    block.is_alive.return_value = True
    block.terminate.side_effect = lambda: setattr(
        block.is_alive, 'return_value', False)
    Block._run_blocks = (block,)
    budgets = []

    def wait(*args, timeout):
      self.assertGreater(timeout, 0)
      self.assertLessEqual(timeout, 10 - clock.now)
      budgets.append(timeout)
      # Simulate a long first wait, then consume the small remaining budget.
      clock.advance(9.8 if len(budgets) == 1 else timeout)
      return []

    with (patch.object(block_module, 'monotonic', clock),
          patch.object(block_module.connection, 'wait', side_effect=wait)):
      Block._cleanup()

    self.assertGreaterEqual(len(budgets), 2)
    self.assertLess(budgets[-1], budgets[0])
    block.terminate.assert_called_once()
