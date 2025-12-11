# coding: utf-8

from crappy import Block
from crappy._global import CrappyFail
from multiprocessing import synchronize, queues, get_start_method, Event
from multiprocessing.sharedctypes import Synchronized
from threading import Thread
from time import sleep
import subprocess
from platform import system
import unittest

from .block_test_base import BlockTestBase, TestBlock


class TestBlockNoResponse(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    sleep(10)


class TestBlockRaise(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    raise ValueError


class TestStartupSequence(BlockTestBase):
  """"""

  def test_prepare_all_prepared(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    sleep(0.5)

    with self.assertRaises(CrappyFail):
      Block.prepare_all()

  def test_prepare_all_launched(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    sleep(0.5)

    Block.prepared_all = False
    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.prepare_all()

  def test_prepare_all_setup(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(Block.prepared_all)
    self.assertFalse(Block.launched_all)

    self.assertIsInstance(Block.ready_barrier, synchronize.Barrier)
    self.assertEqual(Block.ready_barrier.parties, len(Block.instances) + 1)
    self.assertIsInstance(Block.shared_t0, Synchronized)
    self.assertEqual(Block.shared_t0.value, -1.0)

    for event in (Block.stop_event, Block.start_event, Block.pause_event,
                  Block.raise_event, Block.kbi_event):
      with self.subTest(event=event):
        self.assertIsNotNone(event)
        self.assertFalse(event.is_set())

    self.assertIsInstance(Block.log_queue, queues.Queue)
    self.assertIsInstance(Block.log_thread, Thread)
    if get_start_method() == 'spawn':
      self.assertTrue(Block.log_thread.is_alive())
    else:
      self.assertFalse(Block.log_thread.is_alive())

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

    Block.ready_barrier.abort()
    Block.thread_stop = True

    sleep(0.5)

    self.assertTrue(Block.stop_event.is_set())
    self.assertFalse(Block.log_thread.is_alive())

    self.assertTrue(self._block.finished.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

    Block.reset()

  @unittest.skipIf(system() not in ('Linux', 'Darwin'),
                   "Test irrelevant on Windows")
  def test_renice_all(self) -> None:
    """"""

    self._block = TestBlock()
    self._block.niceness = 5

    Block.prepare_all()

    sleep(0.5)

    self.assertEqual(int(subprocess.run(['ps', '-p', str(self._block.pid),
                                         '-o', 'ni='],
                                        capture_output=True).stdout), 0)

    Block.renice_all(allow_root=False)

    self.assertEqual(int(subprocess.run(['ps', '-p', str(self._block.pid),
                                         '-o', 'ni='],
                                        capture_output=True).stdout), 5)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    sleep(0.5)

    Block.reset()

  def test_renice_all_not_prepared(self) -> None:
    """"""

    self._block = TestBlock()

    with self.assertRaises(RuntimeError):
      Block.renice_all(allow_root=False)

    Block.reset()

  def test_renice_all_launched(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    sleep(0.5)

    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.renice_all(allow_root=False)

  def test_launch_all(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.launch_all()

    sleep(0.5)

    for inst in Block.instances:
      self.assertFalse(inst.is_alive())

  def test_launch_all_no_prepared(self) -> None:
    """"""

    self._block = TestBlock()

    with self.assertRaises(RuntimeError):
      Block.launch_all()

    Block.reset()

  def test_launch_all_launched(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.launch_all()

  def test_stop_all(self) -> None:
    """"""

    def stop():
      """"""

      sleep(0.5)
      Block.stop_all()

    stop_thread = Thread(target=stop)

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    stop_thread.start()

    Block.launch_all()

    sleep(0.5)

    for inst in Block.instances:
      self.assertFalse(inst.is_alive())

  def test_restart(self) -> None:
    """"""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.launch_all()

    sleep(0.5)

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.launch_all()

    sleep(0.5)

  def test_cleanup(self) -> None:
    """"""

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
    """"""

    self._block = TestBlockNoResponse()
    _ = TestBlockRaise()

    Block.prepare_all()

    sleep(0.5)

    with self.assertRaises(CrappyFail):
      Block.launch_all()

    sleep(0.5)

    for inst in Block.instances:
      self.assertFalse(inst.is_alive())
