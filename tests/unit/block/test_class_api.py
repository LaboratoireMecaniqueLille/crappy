# coding: utf-8

from crappy import Block
from crappy._exceptions import DefinitionError
from crappy.links import link_graph
import logging
from multiprocessing import Event

from tests.support.blocks import BlockTestBase, TestBlock, link


class TestClassAPI(BlockTestBase):
  """Tests covering the class-level API of Block.

  These tests focus on the bookkeeping performed directly by the Block class,
  independently of the full multiprocessing startup sequence.
  """

  def test_failure_state_is_created_by_run(self) -> None:
    block = TestBlock()
    self.assertFalse(hasattr(block, '_failure_manager'))
    for name in ('raise_event', 'kbi_event', '_raise_event', '_kbi_event'):
      self.assertFalse(hasattr(block, name))

  def test_init_subclass(self) -> None:
    """Tests that subclass registration rejects duplicate class names."""

    class CustomBlock(TestBlock):
      ...

    self.assertIn('CustomBlock', Block.classes.keys())

    with self.assertRaises(DefinitionError):
      class CustomBlock(TestBlock):
        ...

    self.assertIn('CustomBlock', Block.classes.keys())
    self.assertIs(Block.classes['CustomBlock'], CustomBlock)

    Block.reset()

  def test_instances(self) -> None:
    """Tests the tracking of instantiated Blocks and generated names."""

    instances = list()

    self.assertEqual(0, len(Block.instances))

    for _ in range(10):
      instances.append(TestBlock())

    # Every instantiated Block should be registered exactly once and receive a
    # unique automatically generated name.
    self.assertEqual(10, len(Block.instances))
    self.assertEqual(10, len(set(Block.names)))

    for i in range(10):
      with self.subTest(i=i):
        self.assertIn(instances[i], Block.instances)
        self.assertIn(instances[i].name, Block.names)
        self.assertIn(instances[i].name, link_graph.nodes)

    class CustomBlock2(TestBlock):
      ...

    a = CustomBlock2()

    self.assertEqual(11, len(set(Block.names)))
    self.assertIn(a.name, Block.names)

    del instances, a
    Block.reset()

    self.assertEqual(0, len(Block.instances))
    self.assertFalse(link_graph.nodes)

  def test_reset(self) -> None:
    """Reset discards the old graph and allows independent complete runs."""

    source = TestBlock()
    target = TestBlock()
    link(source, target, name='reset-link')

    self.assertEqual(len(link_graph.nodes), 2)
    self.assertEqual(link_graph.link_names(), ('reset-link',))

    Block.reset()
    self.assertFalse(Block.instances)
    self.assertFalse(link_graph.nodes)
    self.assertFalse(link_graph.link_names())

    for run in range(2):
      with self.subTest(run=run):
        block = TestBlock()
        self.addCleanup(block._failure_queue.close)
        Block.start_all(log_level=logging.CRITICAL)
        for event in (block.prepared, block.begun, block.looped, block.finished):
          self.assertTrue(event.is_set())
        self.assertFalse(block.is_alive())
        self.assertEqual(block.exitcode, 0)
        self.assertFalse(Block.instances)
        self.assertFalse(link_graph.nodes)

  def test_stop_all(self) -> None:
    """Tests that Block.stop_all sets the shared stop Event."""

    Block.stop_event = Event()
    self.assertFalse(Block.stop_event.is_set())

    Block.stop_all()

    self.assertTrue(Block.stop_event.is_set())

    Block.reset()
