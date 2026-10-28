"""Resource ownership for tests of Blocks, separate from observable doubles."""

import logging
from multiprocessing import Barrier, Event, Queue as CreateQueue, Value
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
import unittest
from unittest.mock import Mock

from crappy import Block
from crappy._failure import FailureManager
from crappy.links import Link, link_graph

from tests.support.assertions import exception_leaves
from tests.support.blocks import TestBlock
from tests.support.processes import stop_process, wait_until, wait_until_stable
from tests.support.resources import close_queue, drain_queue, isolate_logger, run_cleanups


class BlockTestBase(unittest.TestCase):
  """Clean up owned resources even when a subclass's setup fails."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._block = None

  def setUp(self):
    super().setUp()
    registry = Block.classes.copy()

    def restore_test_classes():
      for name, cls in tuple(Block.classes.items()):
        if name not in registry and cls.__module__.startswith('tests.'):
          Block.classes.pop(name)
      Block.classes.update(registry)

    self.addCleanup(restore_test_classes)
    isolate_logger(self, 'crappy', logging.CRITICAL)
    self.addCleanup(setattr, Block, 'logger', Block.logger)
    self.addCleanup(setattr, Block, 'log_level', Block.log_level)
    Block.logger, Block.log_level = None, logging.CRITICAL
    self._received_failures = {}
    self._first_test_block = len(TestBlock.test_instances)
    self._first_test_queue = len(TestBlock.test_queues)
    self.addCleanup(self._cleanup_blocks)

  wait_until = staticmethod(wait_until)
  wait_until_stable = staticmethod(wait_until_stable)

  def configure_run(self, block, *, t0=0.0, start=True):
    """Give a standalone worker the synchronization normally owned by a run."""

    self._block = block
    block._ready_barrier = Barrier(1)
    block._start_event = Event()
    block._stop_event = Event()
    block._pause_event = Event()
    block._instance_t0 = Value('d', t0)
    block._log_queue = CreateQueue()
    if start:
      block._start_event.set()
    return block

  def failures(self, block):
    """Drain standalone child reports rather than inspecting legacy flags."""

    if block.pid is not None:
      manager = self._received_failures.setdefault(block, FailureManager('main', 'main'))
      manager._logger = Mock()
      manager.receive((block._failure_queue,))
    else:
      manager = block._failure_manager
    try:
      manager.raise_if_failed('test failures')
    except BaseExceptionGroup as errors:
      return exception_leaves(errors)
    return ()

  def _cleanup_blocks(self):
    blocks = list(dict.fromkeys([*TestBlock.test_instances[self._first_test_block:],
                                 *tuple(Block.instances),
                                 *([self._block] if self._block is not None else [])]))
    queues = {id(queue): queue for queue in TestBlock.test_queues[self._first_test_queue:]}
    links = {}
    for block in blocks:
      for name in ('test_failure_queue', '_failure_queue', '_log_queue'):
        queue = getattr(block, name, None)
        if isinstance(queue, Queue):
          queues[id(queue)] = queue
      for link in (*block.inputs, *block.outputs):
        if issubclass(type(link), Link):
          links[id(link)] = link

    def stop(block):
      if not isinstance(block, BaseProcess):
        return

      def request_stop():
        for name in ('_stop_event', '_start_event'):
          event = getattr(block, name, None)
          if event is not None:
            event.set()
        barrier = getattr(block, '_ready_barrier', None)
        if barrier is not None:
          barrier.abort()

      def drain():
        for queue in queues.values():
          drain_queue(queue)

      stop_process(block, request_stop, drain=drain)

    manager = FailureManager('test fixture', 'cleanup')
    manager._logger = Mock()
    try:
      run_cleanups(*(lambda block=block: stop(block) for block in blocks),
                   *(lambda link=link: link.close(manager) for link in links.values()),
                   *(lambda queue=queue: close_queue(queue) for queue in queues.values()),
                   lambda: manager.raise_if_failed('test fixture cleanup failed'),
                   Block.reset)
    finally:
      del TestBlock.test_instances[self._first_test_block:]
      del TestBlock.test_queues[self._first_test_queue:]
    self.assertFalse(Block.instances)
    self.assertFalse(link_graph.nodes)
    self.assertFalse(link_graph.link_names())
