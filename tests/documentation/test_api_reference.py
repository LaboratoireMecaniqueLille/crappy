# coding: utf-8

import ast
from importlib import import_module
from pathlib import Path
import unittest

import crappy


class TestPublicApiReference(unittest.TestCase):
  """Keeps the intentional top-level API aligned with its reference pages."""

  _source_dir = (Path(__file__).resolve().parents[2] / 'docs' / 'source' /
                 'crappy_docs')

  # Each user-facing top-level export maps to a unique, searchable marker in
  # its API page. Aliases are included because they are the convenient public
  # spellings users encounter in scripts.
  _reference_entries = {
    'Actuator': ('aliases.rst', '.. autoclass:: crappy.Actuator'),
    'Block': ('aliases.rst', '.. autoclass:: crappy.Block'),
    'Camera': ('aliases.rst', '.. autoclass:: crappy.Camera'),
    'InOut': ('aliases.rst', '.. autoclass:: crappy.InOut'),
    'Modifier': ('aliases.rst', '.. autoclass:: crappy.Modifier'),
    'OptionalModule': ('exceptions.rst',
                       '.. autoclass:: crappy.OptionalModule'),
    'Path': ('aliases.rst', '.. autoclass:: crappy.Path'),
    'VisionBlock': ('aliases.rst', '.. autoclass:: crappy.VisionBlock'),
    'display_graph': ('aliases.rst', '.. autofunction:: crappy.display_graph'),
    'docs': ('aliases.rst', '.. autofunction:: crappy.docs'),
    'img_link': ('aliases.rst', '.. autofunction:: crappy.img_link'),
    'launch': ('aliases.rst', 'crappy.launch()'),
    'link': ('aliases.rst', '.. autofunction:: crappy.link'),
    'prepare': ('aliases.rst', 'crappy.prepare()'),
    'renice': ('aliases.rst', 'crappy.renice()'),
    'reset': ('aliases.rst', 'crappy.reset()'),
    'resources': ('aliases.rst', '.. autoclass:: crappy.resources'),
    'start': ('aliases.rst', 'crappy.start()'),
    'stop': ('aliases.rst', 'crappy.stop()'),
  }

  # These are intentionally importable conveniences, not standalone API
  # objects. Keeping reasons here makes additions deliberate and reviewable.
  _excluded_exports = {
    '__version__': 'package metadata is explained on the installation page',
    'actuator': 'module namespace documented through its category page',
    'blocks': 'module namespace documented through its category pages',
    'camera': 'module namespace documented through its category page',
    'inout': 'module namespace documented through its category page',
    'lamcube': 'module namespace retained for compatibility',
    'links': 'module namespace documented through the Links page',
    'modifier': 'module namespace documented through its category page',
    'tool': 'module namespace documented through the Tools page',
  }

  @staticmethod
  def _declared_exports() -> set[str]:
    """Finds names explicitly exposed by the package initializer.

    Importing a subpackage also adds it to ``vars(crappy)``, even when the
    initializer never exported it. Those incidental names are not part of the
    intentional top-level API checked here.
    """

    source = Path(crappy.__file__).read_text(encoding='utf-8')
    names = set()
    for statement in ast.parse(source).body:
      if isinstance(statement, ast.ImportFrom):
        for alias in statement.names:
          if alias.name == '*':
            raise AssertionError('Top-level star imports cannot be inventoried')
          names.add(alias.asname or alias.name)
      elif isinstance(statement, ast.Import):
        names.update(alias.asname or alias.name.split('.')[0]
                     for alias in statement.names)
      elif isinstance(statement, ast.Assign):
        names.update(target.id for target in statement.targets
                     if isinstance(target, ast.Name))
      elif isinstance(statement, ast.AnnAssign):
        if isinstance(statement.target, ast.Name):
          names.add(statement.target.id)
      elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
        names.add(statement.name)

    return {name for name in names
            if not name.startswith('_') or name == '__version__'}

  def test_top_level_export_inventory_is_intentional(self) -> None:
    """Fails when an export is added without documentation or a reason."""

    exports = self._declared_exports()
    classified = set(self._reference_entries) | set(self._excluded_exports)

    self.assertTrue(exports <= vars(crappy).keys())
    self.assertEqual(exports, classified)
    self.assertTrue(all(self._excluded_exports.values()))

  def test_imported_subpackage_does_not_expand_explicit_exports(self) -> None:
    """A later subpackage import cannot change the intended API inventory."""

    import_module('crappy.collection')
    self.assertIn('collection', vars(crappy))
    self.assertNotIn('collection', self._declared_exports())
    self.assertEqual(self._declared_exports(),
                     set(self._reference_entries) |
                     set(self._excluded_exports))

  def test_reference_entries_exist(self) -> None:
    """Checks every documented export still has its promised source entry."""

    for export, (file_name, marker) in self._reference_entries.items():
      with self.subTest(export=export):
        source = (self._source_dir / file_name).read_text(encoding='utf-8')
        self.assertIn(marker, source)

  def test_camera_configuration_families_are_documented(self) -> None:
    """Keeps the shared classes and both window backends in the reference."""

    source = (self._source_dir / 'tools.rst').read_text(encoding='utf-8')
    families = ('CameraConfig', 'CameraConfigBoxes', 'DICVEConfig',
                'DISCorrelConfig', 'VideoExtensoConfig')
    for backend, prefix in (('base', ''), ('tkinter', 'Tkinter'),
                            ('pyqt', 'PyQt')):
      module = import_module(f'crappy.tool.camera_config.{backend}')
      for family in families:
        with self.subTest(backend=backend, family=family):
          config_class = getattr(module, f'{prefix}{family}')
          marker = (f'.. autoclass:: {config_class.__module__}.'
                    f'{config_class.__name__}\n')
          self.assertEqual(source.count(marker), 1)

    self.assertEqual(source.count('.. autofunction:: '
                                  'crappy.tool.camera_config.factory.'
                                  'create_configurator\n'), 1)
