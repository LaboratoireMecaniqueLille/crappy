# coding: utf-8

import unittest
from crappy import Block
from threading import Thread
from multiprocessing import get_start_method
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import synchronize
from time import sleep
import subprocess
from platform import system


class BlockStop(Block):
  """"""

  def __init__(self, niceness: int = 0) -> None:
    """"""

    super().__init__()

    self.niceness = niceness

  def loop(self) -> None:
    """"""

    self.stop()


class TestBlockCommands(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    Block.reset()

  def test_get_name(self) -> None:
    """"""

    self.assertEqual(Block.get_name('test'), 'crappy.test-1')

  def test_reset(self) -> None:
    """"""

    self.assertEqual(len(Block.instances), 0)
    self.assertEqual(len(Block.names), 0)
    self.assertFalse(Block.prepared_all)
    self.assertFalse(Block.launched_all)

    _ = BlockStop()

    self.assertEqual(len(Block.instances), 1)
    self.assertEqual(len(Block.names), 1)
    self.assertFalse(Block.prepared_all)
    self.assertFalse(Block.launched_all)

    Block.prepare_all(log_level=None)

    self.assertEqual(len(Block.instances), 1)
    self.assertEqual(len(Block.names), 1)
    self.assertTrue(Block.prepared_all)
    self.assertFalse(Block.launched_all)

    Block.launch_all()

    self.assertEqual(len(Block.instances), 1)
    self.assertEqual(len(Block.names), 1)
    self.assertTrue(Block.prepared_all)
    self.assertTrue(Block.launched_all)

    Block.reset()

    self.assertEqual(len(Block.instances), 0)
    self.assertEqual(len(Block.names), 0)
    self.assertFalse(Block.prepared_all)
    self.assertFalse(Block.launched_all)

  def test_stop_all(self) -> None:
    """"""

    _ = Block()
    __ = Block()

    def stop() -> None:
      """"""

      sleep(1)
      Block.stop_all()

    thread = Thread(target=stop)
    thread.start()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_prepare(self) -> None:
    """"""

    _ = BlockStop()

    self.assertIsNone(Block.shared_t0)
    self.assertIsNone(Block.ready_barrier)
    self.assertIsNone(Block.start_event)
    self.assertIsNone(Block.stop_event)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    self.assertIsNone(Block.prepare_all(log_level=None))

    self.assertIsInstance(Block.shared_t0, Synchronized)
    self.assertIsInstance(Block.ready_barrier, synchronize.Barrier)
    self.assertIsInstance(Block.start_event, synchronize.Event)
    self.assertIsInstance(Block.stop_event, synchronize.Event)

    self.assertTrue(all(block.is_alive() for block in Block.instances))

    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_prepare_fail(self) -> None:
    """"""

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)

    self.assertTrue(all(block.is_alive() for block in Block.instances))

    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_renice(self) -> None:
    """"""

    block_1 = BlockStop()
    block_2 = BlockStop(10)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)

    if system() in ('Linux', 'Darwin'):
      self.assertEqual(int(subprocess.run(['ps', '-p', str(block_1.pid),
                                           '-o', 'ni='],
                                          capture_output=True).stdout), 0)
      self.assertEqual(int(subprocess.run(['ps', '-p', str(block_2.pid),
                                           '-o', 'ni='],
                                          capture_output=True).stdout), 0)

      self.assertIsNone(Block.renice_all(False))

      self.assertEqual(int(subprocess.run(['ps', '-p', str(block_1.pid),
                                           '-o', 'ni='],
                                          capture_output=True).stdout), 0)
      self.assertEqual(int(subprocess.run(['ps', '-p', str(block_2.pid),
                                           '-o', 'ni='],
                                          capture_output=True).stdout), 10)

    else:
      self.assertIsNone(Block.renice_all(False))

    self.assertTrue(all(block.is_alive() for block in Block.instances))

    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_launch(self) -> None:
    """"""

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)

    self.assertEqual(Block.shared_t0.value, -1.0)
    self.assertFalse(Block.ready_barrier.broken)
    self.assertFalse(Block.start_event.is_set())
    self.assertFalse(Block.stop_event.is_set())
    self.assertFalse(Block.thread_stop)

    self.assertTrue(all(block.is_alive() for block in Block.instances))

    self.assertIsNone(Block.launch_all())

    self.assertGreater(Block.shared_t0.value, 0)
    self.assertFalse(Block.ready_barrier.broken)
    self.assertTrue(Block.start_event.is_set())
    self.assertTrue(Block.stop_event.is_set())
    if get_start_method() == 'spawn':
      self.assertTrue(Block.thread_stop)
    else:
      self.assertFalse(Block.thread_stop)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_launch_fail_no_prepare(self) -> None:
    """"""

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_launch_fail_launched(self) -> None:
    """"""

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)
    self.assertTrue(all(block.is_alive() for block in Block.instances))
    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_restart(self) -> None:
    """"""

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)
    self.assertTrue(all(block.is_alive() for block in Block.instances))
    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.reset()

    _ = BlockStop()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))

    Block.prepare_all(log_level=None)
    self.assertTrue(all(block.is_alive() for block in Block.instances))
    Block.launch_all()

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
