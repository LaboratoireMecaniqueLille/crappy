# coding: utf-8

from crappy.blocks.sink import Sink

from ..block import BlockTestBase, TestBlock, link


class TestSink(BlockTestBase):
  """Unit tests for the Sink Block-specific behavior."""

  def test_sink_is_not_pausable(self) -> None:
    """Checks that Sink keeps draining data while Blocks are paused."""

    self.assertFalse(Sink().pausable)

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a Sink without input Links fails early."""

    sink = Sink()

    with self.assertRaises(IOError):
      sink.prepare()

  def test_prepare_accepts_input_links(self) -> None:
    """Checks that prepare accepts one or several incoming Links."""

    source_1 = TestBlock()
    source_2 = TestBlock()
    sink = Sink()

    link(source_1, sink)
    link(source_2, sink)

    sink.prepare()

  def test_loop_drops_all_received_data(self) -> None:
    """Checks that loop flushes every incoming Link."""

    source_1 = TestBlock()
    source_2 = TestBlock()
    sink = Sink()

    link(source_1, sink)
    link(source_2, sink)

    source_1.send({'a': 1})
    source_1.send({'a': 2, 'b': 3})
    source_2.send({'c': 4})

    self.assertTrue(any(link_.poll() for link_ in sink.inputs))

    sink.loop()

    self.assertFalse(any(link_.poll() for link_ in sink.inputs))
    for link_ in sink.inputs:
      with self.subTest(link=link_):
        self.assertTrue(link_.received_chunk.is_set())
        self.assertFalse(link_.received.is_set())
        self.assertFalse(link_.received_last.is_set())
