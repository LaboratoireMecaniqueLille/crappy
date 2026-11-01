# coding: utf-8

from tests.support.paths import PROJECT_ROOT

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest


class TestFailureDocumentation(unittest.TestCase):
  """Checks failure API documentation and the user-facing handling example."""

  _root = PROJECT_ROOT

  def test_group_handling_example_preserves_unhandled_failures(self) -> None:
    """The tutorial example handles errors and interrupts without hiding peers."""

    source = (self._root / 'docs' / 'source' / 'concepts' /
              'lifecycle_shutdown.rst').read_text(encoding='utf-8')
    snippet = source.split('.. code-block:: python\n\n', 1)[1]
    lines = []
    for line in snippet.splitlines():
      if line and not line.startswith('   '):
        break
      lines.append(line[3:] if line else '')
    code = compile('\n'.join(lines), '<failure handling example>', 'exec')
    unhandled = OSError('independent cleanup failure')

    def start() -> None:
      raise BaseExceptionGroup('run failures', [ValueError('invalid value'),
                                                KeyboardInterrupt(), unhandled])

    with redirect_stdout(StringIO()), self.assertRaises(ExceptionGroup) as caught:
      exec(code, {'crappy': SimpleNamespace(start=start)})
    self.assertEqual(caught.exception.exceptions, (unhandled,))

  def test_new_exception_types_are_in_the_reference(self) -> None:
    """Makes crash, timeout and remote replacement details easy to find."""

    source = (self._root / 'docs' / 'source' / 'crappy_docs' /
              'exceptions.rst').read_text(encoding='utf-8')
    for name in ('ProcessExitError', 'ShutdownTimeoutError',
                  'FailureTransportError', 'RemoteFailure',
                  'RemoteKeyboardInterrupt', 'RemoteSystemExit',
                  'RemoteBaseFailure'):
      with self.subTest(exception=name):
        self.assertIn(f'.. autoexception:: crappy._exceptions.{name}', source)

    for name in ('ExceptionNode', 'FailureReport', 'FailureManager'):
      with self.subTest(report=name):
        self.assertIn(f'.. autoclass:: crappy._failure.{name}', source)
