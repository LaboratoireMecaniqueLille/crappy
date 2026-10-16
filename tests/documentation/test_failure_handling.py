# coding: utf-8

import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest


class TestFailureDocumentation(unittest.TestCase):
  """Checks failure API documentation and the user-facing handling example."""

  _root = Path(__file__).resolve().parents[2]

  def test_failure_methods_document_arguments_and_results(self) -> None:
    """Keeps failure methods typed and their inputs and results explained."""

    for file_name in ('_failure.py', '_exceptions.py'):
      source = (self._root / 'src' / 'crappy' / file_name).read_text(
          encoding='utf-8')
      tree = ast.parse(source, feature_version=(3, 11))
      for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef):
          continue
        with self.subTest(file=file_name, method=method.name):
          doc = ast.get_docstring(method)
          self.assertTrue(doc)
          arguments = [*method.args.posonlyargs, *method.args.args,
                       *method.args.kwonlyargs]
          arguments += [arg for arg in (method.args.vararg,
                                        method.args.kwarg) if arg is not None]
          for argument in arguments:
            if argument.arg in ('self', 'cls'):
              continue
            self.assertIsNotNone(argument.annotation, argument.arg)
            self.assertIn(f'{argument.arg}:', doc)
          self.assertIsNotNone(method.returns)
          if ast.unparse(method.returns) != 'None':
            self.assertTrue('Returns:' in doc or 'Yields:' in doc)

  def test_report_constructors_document_each_field(self) -> None:
    """Checks the generated dataclass constructors have argument guidance."""

    from dataclasses import fields
    from crappy._failure import ExceptionNode, FailureReport

    for report_type in (ExceptionNode, FailureReport):
      with self.subTest(report=report_type.__name__):
        for field in fields(report_type):
          self.assertIn(f'{field.name}:', report_type.__doc__)
        self.assertIn('.. versionadded:: 2.2.0', report_type.__doc__)

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
        self.assertIn(f'.. autoexception:: crappy._exceptions.{name}\n', source)
