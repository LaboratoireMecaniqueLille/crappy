# coding: utf-8

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import logging

from crappy.blocks.schedulers.outputs import FromFile


class TestFromFile(TestCase):
  """File loading, interpolation, exhaustion and validation tests."""

  def setUp(self) -> None:
    """Give each test an isolated command file."""

    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.file = Path(directory.name) / 'command.csv'
    self.file.write_text('0,0\n1,10\n2,20\n', encoding='utf-8')

  def test_interpolates_and_stops_after_last_timestamp(self) -> None:
    """Values are clamped before the file and skipped after exhaustion."""

    output = FromFile('command', self.file)
    for dt, expected in ((-1, 0), (0, 0), (0.5, 5),
                         (1.5, 15), (2, 20)):
      with self.subTest(dt=dt):
        self.assertEqual(output(dt, {}), {'command': expected})
    with patch.object(output, 'log') as log:
      self.assertIsNone(output(2.1, {}))
      self.assertIsNone(output(3, {}))
    log.assert_called_once()
    self.assertEqual(log.call_args.args[0], logging.WARNING)

  def test_repeat_last_and_reset_warning(self) -> None:
    """The final value and one-shot warning reset on State entry."""

    output = FromFile('command', str(self.file), repeat_last=True)
    with patch.object(output, 'log') as log:
      self.assertEqual(output(3, {}), {'command': 20})
      self.assertEqual(output(4, {}), {'command': 20})
      self.assertEqual(log.call_count, 1)
      output.reset()
      self.assertEqual(output(5, {}), {'command': 20})
    self.assertEqual(log.call_count, 3)
    self.assertEqual(log.call_args.args[0], logging.WARNING)

  def test_custom_delimiter_and_single_row(self) -> None:
    """A one-row file remains a two-column array."""

    self.file.write_text('1;7\n', encoding='utf-8')
    output = FromFile('command', self.file, delimiter=';')
    self.assertEqual(output(0, {}), {'command': 7})
    self.assertEqual(output(1, {}), {'command': 7})

  def test_rejects_bad_arguments(self) -> None:
    """Labels, paths, delimiters and repeat flags are checked."""

    for kwargs, error in (({'label': ''}, ValueError),
                          ({'label': 1}, TypeError),
                          ({'file_name': ''}, ValueError),
                          ({'file_name': Path('')}, ValueError),
                          ({'file_name': 1}, TypeError),
                          ({'delimiter': ''}, ValueError),
                          ({'delimiter': 1}, TypeError),
                          ({'repeat_last': 1}, TypeError)):
      with self.subTest(kwargs=kwargs), self.assertRaises(error):
        FromFile(**{'label': 'command', 'file_name': self.file, **kwargs})

  def test_rejects_malformed_data(self) -> None:
    """Only two finite columns with increasing timestamps are accepted."""

    for contents in ('0\n1\n', '0,1,2\n', '0,nan\n', '0,inf\n',
                     '1,2\n0,3\n', '1,2\n1,3\n'):
      with self.subTest(contents=contents):
        self.file.write_text(contents, encoding='utf-8')
        with self.assertRaises(ValueError):
          FromFile('command', self.file)
