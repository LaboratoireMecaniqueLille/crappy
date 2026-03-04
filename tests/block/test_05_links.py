# coding: utf-8

from crappy import Block
from crappy._global import LinkDataError
from itertools import chain
from multiprocessing import Barrier, Event, Value, Queue

from .block_test_base import BlockTestBase, TestBlock, link


class TestBlockLink(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    super().loop()
    self.send({'a': 0})
    self.data_available()
    self.recv_data()
    self.send({'a': 0})
    self.recv_last_data()
    self.send({'a': 0})
    self.recv_all_data()
    self.send((0,))


class TestLinks(BlockTestBase):
  """"""

  def test_link_blocks(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()
    block_4 = TestBlock()

    link(block_1, block_2)
    link(block_2, block_2)
    link(block_2, block_3)
    link(block_2, block_4)

    self.assertEqual(len(block_1.inputs), 0)
    self.assertEqual(len(block_2.inputs), 2)
    self.assertEqual(len(block_3.inputs), 1)
    self.assertEqual(len(block_4.inputs), 1)

    self.assertEqual(len(block_1.outputs), 1)
    self.assertEqual(len(block_2.outputs), 3)
    self.assertEqual(len(block_3.outputs), 0)
    self.assertEqual(len(block_4.outputs), 0)

    Block.reset()

  def test_send(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_2)
    link(block_1, block_3)

    block_1.send(None)

    for link_ in block_1.outputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    with self.assertRaises(LinkDataError):
      block_1.send(0)

    block_1.send({'a': 0, 'b': 1})
    for link_ in block_1.outputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertTrue(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    for link_ in block_1.outputs:
      link_.sent.clear()

    block_1.labels = ['a', 'b']
    block_1.send((0, 1))
    for link_ in block_1.outputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertTrue(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    Block.reset()

  def test_data_available(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_2)
    link(block_1, block_3)

    for link_ in chain(block_2.inputs, block_3.inputs):
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self.assertFalse(block_2.data_available())
    self.assertFalse(block_3.data_available())

    for link_ in chain(block_2.inputs, block_3.inputs):
      with self.subTest(link=link_):
        self.assertTrue(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    block_1.send({'a': 0, 'b': 1})

    self.assertTrue(block_2.data_available())
    self.assertTrue(block_3.data_available())

    block_2.recv_data()
    block_3.recv_data()

    self.assertFalse(block_2.data_available())
    self.assertFalse(block_3.data_available())

    Block.reset()

  def test_recv_data(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_3)
    link(block_2, block_3)

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self.assertEqual(block_3.recv_data(), dict())

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertTrue(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    block_1.send({'a': 0})
    block_2.send({'b': 1})

    self.assertEqual(block_3.recv_data(), {'a': 0, 'b': 1})

    block_1.labels = ['a']
    block_2.labels = ['b']

    block_1.send((0,))
    block_2.send((1,))

    self.assertEqual(block_3.recv_data(), {'a': 0, 'b': 1})

    block_1.labels = ['a', 'c']

    block_1.send((0, 2))
    block_2.send((1,))

    self.assertEqual(block_3.recv_data(), {'a': 0, 'b': 1, 'c': 2})

    block_1.send((0,))
    block_2.send((1,))

    self.assertEqual(block_3.recv_data(), {'a': 0, 'b': 1})

    block_1.labels = ['a', 'c', 'd']

    block_1.send((0, 2))
    block_2.send((1,))

    self.assertEqual(block_3.recv_data(), {'a': 0, 'b': 1, 'c': 2})

    block_1.labels = ['b']

    block_1.send((0,))
    block_2.send((1,))

    self.assertEqual(block_3.recv_data(), {'b': 1})

    Block.reset()

  def test_recv_last_data(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_3)
    link(block_2, block_3)

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self.assertEqual(block_3.recv_last_data(False), dict())
    self.assertEqual(block_3.recv_last_data(True), dict())

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertTrue(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    block_1.send({'a': 0})
    block_2.send({'b': 1})

    self.assertEqual(block_3.recv_last_data(False), {'a': 0, 'b': 1})

    block_1.send({'a': 0})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_last_data(False), {'a': 0, 'b': 2})

    block_1.send({'a': 0})
    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_last_data(False), {'a': 0, 'b': 2})

    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_last_data(False), {'b': 2})

    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_last_data(True), {'a': 0, 'b': 2})

    block_1.send({'a': 1})
    block_3.recv_last_data(True)

    self.assertEqual(block_3.recv_last_data(True), {'a': 1, 'b': 2})

    Block.reset()

  def test_recv_all_data(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_3)
    link(block_2, block_3)

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self.assertEqual(block_3.recv_all_data(None), dict())
    self.assertEqual(block_3.recv_all_data(1.0), dict())

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertTrue(link_.received_chunk.is_set())

    block_1.send({'a': 0})
    block_2.send({'b': 1})

    self.assertEqual(block_3.recv_all_data(None), {'a': [0], 'b': [1]})

    block_1.send({'a': 0})
    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_all_data(None), {'a': [0], 'b': [1, 2]})

    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_all_data(None), {'b': [1, 2]})

    block_1.send({'a': 0})
    block_1.send({'b': 2})
    block_2.send({'a': 1})
    block_2.send({'b': 1})

    self.assertEqual(block_3.recv_all_data(None), {'a': [0, 1], 'b': [2, 1]})

    Block.reset()

  def test_recv_all_data_raw(self) -> None:
    """"""

    block_1 = TestBlock()
    block_2 = TestBlock()
    block_3 = TestBlock()

    link(block_1, block_3)
    link(block_2, block_3)

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self.assertEqual(block_3.recv_all_data_raw(None), [dict(), dict()])
    self.assertEqual(block_3.recv_all_data_raw(1.0), [dict(), dict()])

    for link_ in block_3.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertTrue(link_.received_chunk.is_set())

    block_1.send({'a': 0})
    block_2.send({'b': 1})

    self.assertEqual(block_3.recv_all_data_raw(None), [{'a': [0]}, {'b': [1]}])

    block_1.send({'a': 0})
    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_all_data_raw(None),
                     [{'a': [0]}, {'b': [1, 2]}])

    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_all_data_raw(None), [dict(), {'b': [1, 2]}])

    block_1.send({'a': 0})
    block_1.send({'b': 2})
    block_2.send({'a': 1})
    block_2.send({'b': 1})
    block_2.send({'b': 2})

    self.assertEqual(block_3.recv_all_data_raw(None),
                     [{'a': [0], 'b': [2]}, {'a': [1], 'b': [1, 2]}])

    Block.reset()

  def test_link_data_error(self) -> None:
    """"""

    self._block = TestBlockLink(stop=False)

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    link(self._block, self._block)

    for link_ in self._block.inputs:
      with self.subTest(link=link_):
        self.assertFalse(link_.polled.is_set())
        self.assertFalse(link_.sent.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
        self.assertFalse(link_.received_chunk.is_set())

    self._block.start()

    self._block.join(4.0)

    for link_ in self._block.inputs:
      with self.subTest(link=link_):
        self.assertTrue(link_.polled.is_set())
        self.assertTrue(link_.sent.is_set())
        self.assertTrue(link_.received.is_set())
        self.assertTrue(link_.received_last.is_set())
        self.assertTrue(link_.received_chunk.is_set())

    for link_ in self._block.outputs:
      with self.subTest(link=link_):
        self.assertTrue(link_.polled.is_set())
        self.assertTrue(link_.sent.is_set())
        self.assertTrue(link_.received.is_set())
        self.assertTrue(link_.received_last.is_set())
        self.assertTrue(link_.received_chunk.is_set())

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()
