# coding: utf-8

from crappy import Block
import unittest


class FakeException(Exception):
  """"""


class BlockFailInit(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

    raise FakeException


class BlockFailPrepare(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

  def prepare(self) -> None:
    """"""

    raise FakeException


class BlockFailBegin(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

  def begin(self) -> None:
    """"""

    raise FakeException


class BlockFailLoop(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

  def loop(self) -> None:
    """"""

    raise FakeException


class BlockStop2(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

  def loop(self) -> None:
    """"""

    self.stop()


class BlockFailFinish(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

  def loop(self) -> None:
    """"""

    raise FakeException

  def finish(self) -> None:
    """"""

    raise FakeException


class TestBlockException(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    Block.reset()

  def test_init_fail(self) -> None:
    """"""

    with self.assertRaises(FakeException):
      BlockFailInit()
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_prepare_fail(self) -> None:
    """"""

    _ = BlockFailPrepare()
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_begin_fail(self) -> None:
    """"""

    _ = BlockFailBegin()
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_loop_fail(self) -> None:
    """"""

    _ = BlockFailLoop()
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_loop_stop(self) -> None:
    """"""

    _ = BlockStop2()
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))

  def test_finish_fail(self) -> None:
    """"""

    _ = BlockFailFinish()
    Block.start_all(log_level=None)
    self.assertTrue(all(not block.is_alive() for block in Block.instances))
