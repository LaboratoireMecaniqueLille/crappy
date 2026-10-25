# coding: utf-8

from importlib import import_module
from inspect import isfunction
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
    """Returns the package's intentional public export inventory.

    Importing a subpackage also adds it to ``vars(crappy)``, even when the
    initializer never exported it. Those incidental names are not part of the
    intentional top-level API checked here.
    """

    return set(crappy.__all__)

  def test_top_level_export_inventory_is_intentional(self) -> None:
    """Fails when an export is added without documentation or a reason."""

    exports = self._declared_exports()
    classified = set(self._reference_entries) | set(self._excluded_exports)

    self.assertEqual(exports, classified)
    for export in exports:
      with self.subTest(export=export):
        self.assertTrue(hasattr(crappy, export))
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

  def test_import_helpers_are_documented(self) -> None:
    """New import helpers have API entries without tutorial requirements."""

    functions = {
      'actuators.rst': ('crappy.actuator.actuator_names',
                        'crappy.actuator.resolve_actuator',
                        'crappy.actuator.is_reserved'),
      'cameras.rst': ('crappy.camera.camera_names',
                      'crappy.camera.resolve_camera',
                      'crappy.camera.is_reserved'),
      'inouts.rst': ('crappy.inout.inout_names',
                     'crappy.inout.resolve_inout',
                     'crappy.inout.is_reserved'),
      'block_base.rst': ('crappy.blocks.is_reserved',),
      'tools.rst': ('crappy.tool.camera_config.factory.resolve_configurator',),
    }
    for file_name, names in functions.items():
      source = (self._source_dir / file_name).read_text(encoding='utf-8')
      for name in names:
        with self.subTest(name=name):
          module, attribute = name.rsplit('.', 1)
          self.assertTrue(callable(getattr(import_module(module), attribute)))
          self.assertEqual(source.count(f'.. autofunction:: {name}\n'), 1)

    tools = (self._source_dir / 'tools.rst').read_text(encoding='utf-8')
    self.assertEqual(tools.count('.. autoclass:: '
                                'crappy.tool.camera_config.factory.'
                                'LazyConfiguratorMapping\n'), 1)
    optional = (self._source_dir / 'exceptions.rst').read_text(encoding='utf-8')
    self.assertIn(':members: resolve, is_available', optional)

  def test_vision_configuration_resolution_methods_are_documented(self) -> None:
    """Both source-side preparation methods remain in the API reference."""

    source = (self._source_dir / 'vision_blocks.rst').read_text(encoding='utf-8')
    for name in ('VisionBlock', 'crappy.blocks.vision.CameraSource'):
      entry = source.split(f'.. autoclass:: {name}\n', 1)[1].split('\n\n', 1)[0]
      self.assertIn('resolve_config_requests', entry)

  def test_validation_helpers_are_documented(self) -> None:
    """Keeps the validation reference aligned with the defined helpers."""

    module = import_module('crappy._validation')
    helpers = {name for name, value in vars(module).items()
               if isfunction(value) and value.__module__ == module.__name__}
    source = (self._source_dir / 'validation.rst').read_text(encoding='utf-8')
    self.assertIn('.. automodule:: crappy._validation\n', source)
    members = source.split(':members:', 1)[1].split('\n\n', 1)[0]
    self.assertEqual({name.strip() for name in members.split(',')}, helpers)

    api = (self._source_dir.parent / 'api.rst').read_text(encoding='utf-8')
    self.assertIn('crappy_docs/validation.rst', api)
    developers = (self._source_dir.parent / 'developers.rst').read_text(
      encoding='utf-8')
    self.assertIn('<crappy_docs/validation>', developers)
