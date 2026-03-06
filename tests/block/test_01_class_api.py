# coding: utf-8

from crappy import Block
from crappy._global import DefinitionError
from multiprocessing import Value, Barrier, Event, Queue
from threading import Thread

from .block_test_base import BlockTestBase, TestBlock


class TestClassAPI(BlockTestBase):
  """Tests covering the class-level API of Block.

  These tests focus on the bookkeeping performed directly by the Block class,
  independently of the full multiprocessing startup sequence.
  """

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
        self.assertEqual(Block.names[i], f"crappy.TestBlock-{i+1}")

    class CustomBlock2(TestBlock):
      ...

    a = CustomBlock2()

    self.assertEqual(11, len(set(Block.names)))
    self.assertEqual(Block.names[10], "crappy.CustomBlock2-1")

    del instances, a
    Block.reset()

    self.assertEqual(0, len(Block.instances))

  def test_reset(self) -> None:
    """Tests that Block.reset clears all shared class attributes."""

    TestBlock()

    Block.ready_barrier = Barrier(1)
    Block.shared_t0 = Value('d', -1.0)
    Block.start_event = Event()
    Block.pause_event = Event()
    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.log_queue = Queue()
    Block.log_thread = Thread(target=Block._log_target)

    Block.reset()

  def test_stop_all(self) -> None:
    """Tests that Block.stop_all sets the shared stop Event."""

    Block.stop_event = Event()
    self.assertFalse(Block.stop_event.is_set())

    Block.stop_all()

    self.assertTrue(Block.stop_event.is_set())

    Block.reset()
