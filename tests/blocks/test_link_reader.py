# coding: utf-8

import logging
from crappy.blocks.link_reader import LinkReader

from ..block import BlockTestBase, TestBlock, link


class TestLinkReader(BlockTestBase):
  """Unit tests for the LinkReader Block-specific behavior."""

  @staticmethod
  def _capture_logs(reader: LinkReader) -> list[tuple[int, str]]:
    """Captures LinkReader log calls without relying on logging handlers."""

    logs = list()

    def log(level: int, msg: str) -> None:
      logs.append((level, msg))

    reader.log = log
    return logs

  def test_reader_name(self) -> None:
    """Checks that the display name can be explicitly set."""

    reader = LinkReader(name='Reader')

    self.assertEqual(reader._reader_name, 'Reader')

  def test_prepare_requires_input_link(self) -> None:
    """Checks that a LinkReader without input Links fails early."""

    reader = LinkReader(name='Reader')

    with self.assertRaises(IOError):
      reader.prepare()

  def test_prepare_accepts_input_link(self) -> None:
    """Checks that prepare accepts one incoming Link."""

    source = TestBlock()
    reader = LinkReader(name='Reader')
    link(source, reader)

    reader.prepare()

  def test_loop_does_nothing_without_available_data(self) -> None:
    """Checks that loop remains quiet when input Links have no data."""

    source = TestBlock()
    reader = LinkReader(name='Reader')
    link(source, reader)
    logs = self._capture_logs(reader)

    reader.loop()

    self.assertEqual(logs, [])

  def test_loop_logs_each_message_from_one_link(self) -> None:
    """Checks that variable-label messages are displayed individually."""

    source = TestBlock()
    reader = LinkReader(name='Reader')
    link(source, reader)
    logs = self._capture_logs(reader)

    source.send({'a': 1})
    source.send({'b': 2})
    source.send({'a': 3, 'b': 4})

    reader.loop()

    self.assertEqual(logs, [
      (logging.INFO, "Reader got: {'a': 1}"),
      (logging.INFO, "Reader got: {'b': 2}"),
      (logging.INFO, "Reader got: {'a': 3, 'b': 4}"),
    ])
    self.assertFalse(reader.inputs[0].poll())

  def test_loop_logs_each_input_link_separately(self) -> None:
    """Checks that messages from several input Links are not merged."""

    source_1 = TestBlock()
    source_2 = TestBlock()
    reader = LinkReader(name='Reader')
    link(source_1, reader)
    link(source_2, reader)
    logs = self._capture_logs(reader)

    source_1.send({'a': 1})
    source_1.send({'a': 2})
    source_2.send({'b': 10})

    reader.loop()

    self.assertEqual(logs, [
      (logging.INFO, "Reader got: {'a': 1}"),
      (logging.INFO, "Reader got: {'a': 2}"),
      (logging.INFO, "Reader got: {'b': 10}"),
    ])
    self.assertFalse(any(link_.poll() for link_ in reader.inputs))
