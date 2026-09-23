# coding: utf-8

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

  def test_top_level_export_inventory_is_intentional(self) -> None:
    """Fails when an export is added without documentation or a reason."""

    exports = {
      name for name in vars(crappy)
      if not name.startswith('_') or name == '__version__'
    }
    classified = set(self._reference_entries) | set(self._excluded_exports)

    self.assertEqual(exports, classified)
    self.assertTrue(all(self._excluded_exports.values()))

  def test_reference_entries_exist(self) -> None:
    """Checks every documented export still has its promised source entry."""

    for export, (file_name, marker) in self._reference_entries.items():
      with self.subTest(export=export):
        source = (self._source_dir / file_name).read_text(encoding='utf-8')
        self.assertIn(marker, source)
