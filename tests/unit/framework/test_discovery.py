"""Collection stays complete, selective, and independent of GUI dependencies."""

from tests.support.processes import run_python
import os
import textwrap
import unittest

from tests.support.paths import PROJECT_ROOT


class TestDiscovery(unittest.TestCase):
  def run_script(self, script: str) -> None:
    environment = dict(os.environ, MPLBACKEND='Agg')
    environment.pop('CRAPPY_TEST_GUI', None)
    result = run_python(['-c', textwrap.dedent(script)], cwd=PROJECT_ROOT, env=environment, timeout=45)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_unselected_gui_groups_do_not_import_optional_dependencies(self):
    self.run_script('''
      import importlib
      import importlib.abc
      import os
      from pathlib import Path
      import sys
      import unittest

      forbidden = {'tkinter', '_tkinter', 'PyQt6', 'PySide6', 'cv2',
                   'matplotlib', 'pyqtgraph', 'numpy'}

      class RejectOptional(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.split('.')[0] in forbidden:
            raise AssertionError('Unselected GUI imported ' + fullname)

      sys.meta_path.insert(0, RejectOptional())
      modules = sorted(Path('tests/gui').rglob('test_*.py'))
      assert modules, 'No graphical tests were found'
      for path in modules:
        group = path.parts[2]
        other = 'pyqt' if group == 'tkinter' else 'tkinter'
        for selected in ('', other):
          os.environ['CRAPPY_TEST_GUI'] = selected
          try:
            importlib.import_module('.'.join(path.with_suffix('').parts))
          except unittest.SkipTest:
            pass
          else:
            raise AssertionError('Unselected graphical tests were imported')
      assert not forbidden.intersection(sys.modules)
    ''')

  def test_standard_discovery_collects_every_headless_case_once(self):
    self.run_script('''
      from pathlib import Path
      import unittest

      def cases(suite):
        for item in suite:
          if isinstance(item, unittest.TestSuite):
            yield from cases(item)
          else:
            yield item

      loader = unittest.TestLoader()
      root = list(cases(loader.discover('tests', top_level_dir='.')))
      assert not loader.errors, loader.errors
      ids = [case.id() for case in root]
      assert ids and len(ids) == len(set(ids)), 'Duplicate collection'

      expected = []
      for directory in ('unit', 'integration', 'documentation'):
        for path in sorted(Path('tests', directory).rglob('test_*.py')):
          module = '.'.join(path.with_suffix('').parts)
          try:
            selected = loader.loadTestsFromName(module)
          except unittest.SkipTest:
            continue
          expected.extend(case.id() for case in cases(selected)
                          if type(case).__module__.startswith('tests.'))
      assert not loader.errors, loader.errors
      assert len(expected) == len(set(expected)), 'Imported duplicate test classes'
      # Compare project cases without depending on unittest's representation
      # of modules skipped before import (including unselected GUI groups).
      actual = {case.id() for case in root
                if type(case).__module__.startswith('tests.')}
      assert actual == set(expected), (actual - set(expected), set(expected) - actual)
    ''')

  def test_focused_discovery_honors_the_filename_pattern(self):
    self.run_script('''
      import unittest

      def cases(suite):
        for item in suite:
          if isinstance(item, unittest.TestSuite):
            yield from cases(item)
          else:
            yield item

      loader = unittest.TestLoader()
      selected = list(cases(loader.discover(
          'tests/unit/modifier', pattern='test_mean.py', top_level_dir='.')))
      assert selected, 'Focused discovery collected no tests'
      assert not loader.errors, loader.errors
      assert {type(case).__module__ for case in selected} == {
          'tests.unit.modifier.test_mean'}
    ''')

  def test_optional_dependency_becomes_usable_after_installation(self):
    """An absent package can be skipped, then imported when it appears."""

    self.run_script('''
      import importlib
      from pathlib import Path
      import sys
      from tempfile import TemporaryDirectory
      from tests.support.dependencies import optional_import

      name = '_crappy_test_optional_dependency'
      assert optional_import(name) is None
      with TemporaryDirectory() as folder:
        sys.path.insert(0, folder)
        Path(folder, name + '.py').write_text('def value(): return 42\\n')
        importlib.invalidate_caches()
        assert optional_import(name).value() == 42
    ''')

  def test_broken_installed_dependencies_fail_collection(self):
    """A native-loader or transitive-import failure cannot become a skip."""

    for package, module in (('cv2', 'tests.integration.test_vision'),
                             ('tables', 'tests.integration.test_numeric')):
      for failure in ("ImportError()",
                       "ModuleNotFoundError(name='_missing_transitive_dependency')",
                       "RuntimeError()"):
        with self.subTest(package=package, failure=failure):
          self.run_script(f'''
            from pathlib import Path
            import sys
            from tempfile import TemporaryDirectory
            import unittest

            with TemporaryDirectory() as folder:
              Path(folder, {package!r} + '.py').write_text('raise ' + {failure!r})
              sys.path.insert(0, folder)
              loader = unittest.TestLoader()
              try:
                loader.loadTestsFromName({module!r})
              except RuntimeError:
                pass
              else:
                assert loader.errors, 'Broken dependency became a skip'
          ''')
