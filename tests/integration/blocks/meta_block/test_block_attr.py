# coding: utf-8

import unittest
from crappy import Block, link
from crappy._global import T0NotSetError
from typing import Optional
from time import time
from multiprocessing import Value, Barrier, Event, Queue


class BlockFreq(Block):
  """"""

  def __init__(self, freq: Optional[float]) -> None:
    """"""

    super().__init__()

    self.freq = freq
    self.n_loops = 0

    self._instance_t0 = Value('d', time())
    self._ready_barrier = Barrier(1)
    self._stop_event = Event()
    self._start_event = Event()
    self._log_queue = Queue()
    self._log_level = None

    self._start_event.set()

  def loop(self) -> None:
    """"""

    self.n_loops += 1
    if time() - self.t0 > 1:
      self.stop()


class TestBlockAttr(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    Block.reset()

  def test_input_output(self) -> None:
    """"""

    input_ = Block()
    output = Block()

    self.assertEqual(len(input_.outputs), 0)
    self.assertEqual(len(output.inputs), 0)

    link(input_, output)

    self.assertEqual(len(input_.outputs), 1)
    self.assertEqual(len(output.inputs), 1)

  def test_name(self) -> None:
    """"""

    block_1 = Block()
    block_2 = Block()

    self.assertEqual(block_1.name, "crappy.Block-1")
    self.assertEqual(block_2.name, "crappy.Block-2")

  def test_freq(self) -> None:
    """"""

    for freq, expected in zip((2, 100, None), (1, 90, 200)):
      with self.subTest(freq=freq):
        block = BlockFreq(freq)
        self.assertEqual(block.n_loops, 0)
        block.run()
        self.assertGreater(block.n_loops, expected)
        Block.reset()

  def test_t0_fail(self) -> None:
    """"""

    block = Block()

    with self.assertRaises(T0NotSetError):
      _ = block.t0

  def test_t0_success(self) -> None:
    """"""

    block = BlockFreq(10)
    Block.start_all(log_level=None)

    self.assertGreater(block.t0, 0)
