# coding: utf-8

from crappy import Block, link
from crappy._global import LinkDataError
import unittest
from multiprocessing import Value, Array
from multiprocessing.sharedctypes import Synchronized, SynchronizedString
from time import sleep, time


class FakeSendSimple(Block):
  """"""

  def __init__(self, label: str, value: int) -> None:
    """"""

    super().__init__()

    self._value = value
    self._label = label

  def loop(self) -> None:
    """"""

    self.send({self._label: self._value})
    sleep(0.5)


class FakeSendDelay(Block):
  """"""

  def __init__(self, label: str, value: int) -> None:
    """"""

    super().__init__()

    self._value = value
    self._label = label

  def loop(self) -> None:
    """"""

    if time() - self.t0 < 0.1:
      self.send({self._label: 0})
    else:
      self.send({self._label: self._value})


class FakeSendMultiple(Block):
  """"""

  def __init__(self,
               label: str,
               value_1: int,
               value_2: int,
               reps: int) -> None:
    """"""

    super().__init__()

    self._value_1 = value_1
    self._value_2 = value_2
    self._reps = reps
    self._label = label

  def loop(self) -> None:
    """"""

    for _ in range(self._reps):
      self.send({self._label: self._value_1})
      self.send({self._label: self._value_2})
    sleep(0.5)


class FakeRecv(Block):
  """"""

  def __init__(self,
               value_1: Synchronized,
               value_2: Synchronized,
               label_1: str,
               label_2: str) -> None:
    """"""

    super().__init__()

    self._value_1 = value_1
    self._value_2 = value_2
    self._label_1 = label_1
    self._label_2 = label_2

  def loop(self) -> None:
    """"""

    sleep(0.2)
    data = self.recv_data()
    if self._label_1 in data:
      self._value_1.value = data[self._label_1]
    if self._label_2 in data:
      self._value_2.value = data[self._label_2]
    raise Exception


class FakeRecvLast(Block):
  """"""

  def __init__(self,
               value_1: Synchronized,
               value_2: Synchronized,
               label_1: str,
               label_2: str,
               fill_missing: bool) -> None:
    """"""

    super().__init__()

    self._value_1 = value_1
    self._value_2 = value_2
    self._label_1 = label_1
    self._label_2 = label_2

    self._fill_missing = fill_missing

  def loop(self) -> None:
    """"""

    sleep(0.2)
    data = self.recv_last_data(fill_missing=self._fill_missing)
    if self._label_1 in data:
      self._value_1.value = data[self._label_1]
    if self._label_2 in data:
      self._value_2.value = data[self._label_2]
    raise Exception


class FakeRecvLastFill(Block):
  """"""

  def __init__(self,
               value_1: Synchronized,
               value_2: Synchronized,
               label_1: str,
               label_2: str,
               fill_missing: bool) -> None:
    """"""

    super().__init__()

    self._value_1 = value_1
    self._value_2 = value_2
    self._label_1 = label_1
    self._label_2 = label_2

    self._fill_missing = fill_missing

  def loop(self) -> None:
    """"""

    sleep(0.05)
    data = self.recv_last_data(fill_missing=self._fill_missing)
    if self._label_1 in data:
      self._value_1.value = data[self._label_1]
    else:
      self._value_1.value = 0

    if self._label_2 in data:
      self._value_2.value = data[self._label_2]
    else:
      self._value_2.value = 0

    sleep(0.15)

    data = self.recv_last_data(fill_missing=self._fill_missing)
    if self._label_1 in data:
      self._value_1.value = data[self._label_1]
    else:
      self._value_1.value = 0

    if self._label_2 in data:
      self._value_2.value = data[self._label_2]
    else:
      self._value_2.value = 0
    raise Exception


class FakeRecvAll(Block):
  """"""

  def __init__(self,
               value_1: SynchronizedString,
               value_2: SynchronizedString,
               label_1: str,
               label_2: str) -> None:
    """"""

    super().__init__()

    self._value_1 = value_1
    self._value_2 = value_2
    self._label_1 = label_1
    self._label_2 = label_2

  def loop(self) -> None:
    """"""

    sleep(0.2)
    data = self.recv_all_data()
    if self._label_1 in data:
      self._value_1.value = str(data[self._label_1]).encode()
    if self._label_2 in data:
      self._value_2.value = str(data[self._label_2]).encode()
    raise Exception


class FakeRecvAllRaw(Block):
  """"""

  def __init__(self, value: SynchronizedString) -> None:
    """"""

    super().__init__()

    self._value = value

  def loop(self) -> None:
    """"""

    sleep(0.2)
    data = self.recv_all_data_raw()
    self._value.value = str(data).encode()
    raise Exception


class TestBlockLink(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    Block.reset()

  def test_send_list_fail(self) -> None:
    """"""

    block_send = Block()
    block_recv = Block()
    link(block_send, block_recv)

    block_send._log_level = None
    block_send._set_block_logger()

    with self.assertRaises(LinkDataError):
      block_send.send([1, 2, 3])

    with self.assertRaises(LinkDataError):
      block_send.send(0)

  def test_send_list_success(self) -> None:
    """"""

    block_send = Block()
    block_recv = Block()
    link(block_send, block_recv)

    block_send._log_level = None
    block_send._set_block_logger()
    block_send.labels = ['a', 'b', 'c']

    block_recv._log_level = None
    block_recv._set_block_logger()

    block_send.send([1, 2, 3])

    self.assertDictEqual(block_recv.recv_data(), {'a': 1, 'b': 2, 'c': 3})

  def test_recv(self) -> None:
    """"""

    value_1 = Value('i', 0)
    value_2 = Value('i', 0)
    block_send_1 = FakeSendSimple('label_1', 1)
    block_send_2 = FakeSendSimple('label_2', 2)
    block_recv = FakeRecv(value_1, value_2, 'label_1', 'label_2')
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value_1.value, 1)
    self.assertEqual(value_2.value, 2)

  def test_recv_last(self) -> None:
    """"""

    value_1 = Value('i', 0)
    value_2 = Value('i', 0)
    block_send_1 = FakeSendDelay('label_1', 1)
    block_send_2 = FakeSendDelay('label_2', 2)
    block_recv = FakeRecvLast(value_1, value_2, 'label_1', 'label_2', False)
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value_1.value, 1)
    self.assertEqual(value_2.value, 2)

  def test_recv_last_fill(self) -> None:
    """"""

    value_1 = Value('i', 0)
    value_2 = Value('i', 0)
    block_send_1 = FakeSendSimple('label_1', 1)
    block_send_2 = FakeSendDelay('label_2', 2)
    block_recv = FakeRecvLastFill(value_1, value_2, 'label_1', 'label_2', True)
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value_1.value, 1)
    self.assertEqual(value_2.value, 2)

  def test_recv_last_no_fill(self) -> None:
    """"""

    value_1 = Value('i', 0)
    value_2 = Value('i', 0)
    block_send_1 = FakeSendSimple('label_1', 1)
    block_send_2 = FakeSendDelay('label_2', 2)
    block_recv = FakeRecvLastFill(value_1, value_2, 'label_1', 'label_2',
                                  False)
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value_1.value, 0)
    self.assertEqual(value_2.value, 2)

  def test_recv_all(self) -> None:
    """"""

    value_1 = Array('c', 3)
    value_2 = Array('c', 18)
    block_send_1 = FakeSendSimple('label_1', 1)
    block_send_2 = FakeSendMultiple('label_2', 2, 3, 3)
    block_recv = FakeRecvAll(value_1, value_2, 'label_1', 'label_2')
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value_1.value, b'[1]')
    self.assertEqual(value_2.value, b'[2, 3, 2, 3, 2, 3]')

  def test_recv_all_raw(self) -> None:
    """"""

    value = Array('c', 51)
    block_send_1 = FakeSendSimple('label_1', 1)
    block_send_2 = FakeSendMultiple('label_2', 2, 3, 3)
    block_recv = FakeRecvAllRaw(value)
    link(block_send_1, block_recv)
    link(block_send_2, block_recv)
    Block.start_all(log_level=None)

    self.assertTrue(all(not block.is_alive() for block in Block.instances))
    self.assertEqual(value.value,
                     b"[{'label_1': [1]}, {'label_2': [2, 3, 2, 3, 2, 3]}]")
