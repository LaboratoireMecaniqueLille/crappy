# coding: utf-8

"""Lazy access to bundled images and compatibility resource aliases."""

from tests.support.processes import run_python
import textwrap
from unittest import TestCase


class TestResources(TestCase):
  def _run_script(self, script: str) -> None:
    """Checks resource imports in a fresh interpreter."""

    result = run_python(['-B', '-c', textwrap.dedent(script)], timeout=30)
    if result.returncode == 77:
      self.skipTest('OpenCV is unavailable for decoding resources')
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_common_imports_and_discovery_do_not_load_resources(self):
    self._run_script('''
      import importlib.abc
      import sys

      class RejectResources(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname == 'crappy._resources':
            raise AssertionError('Resource implementation imported too early')

      sys.meta_path.insert(0, RejectResources())
      import crappy
      import crappy._global as common
      from crappy._global import OptionalModule, docs

      assert OptionalModule is crappy.OptionalModule
      assert docs is crappy.docs
      assert 'crappy._resources' not in sys.modules
      for module in (crappy, common):
        assert 'resources' not in vars(module)
        assert 'resources' in module.__all__
        assert 'resources' in dir(module)
        try:
          getattr(module, 'unknown_resource_export')
        except AttributeError as error:
          assert module.__name__ in str(error), error
        else:
          raise AssertionError('Unknown attribute was accepted')
      assert 'crappy._resources' not in sys.modules
    ''')

  def test_resource_aliases_load_once_from_either_namespace(self):
    for first_namespace in ('crappy', 'common'):
      with self.subTest(first_namespace=first_namespace):
        self._run_script(f'''
          import sys
          from unittest.mock import patch
          import crappy
          import crappy._global as common

          assert 'crappy._resources' not in sys.modules
          from tests.support.dependencies import optional_import
          cv2 = optional_import('cv2')
          if cv2 is None:
            raise SystemExit(77)

          with patch.object(cv2, 'imdecode', wraps=cv2.imdecode) as decode:
            first = {first_namespace}.resources
            from crappy import resources
            from crappy._global import resources as legacy
            from crappy._resources import resources as implementation

            assert isinstance(first, type)
            assert first is resources is legacy is implementation
            assert vars(crappy)['resources'] is first
            assert vars(common)['resources'] is first
            assert crappy.resources is common.resources is first
            assert decode.call_count == 3, decode.call_count
        ''')

  def test_decoded_images_and_paths_match_the_baseline(self):
    self._run_script('''
      import hashlib
      from pathlib import Path
      from tests.support.dependencies import optional_import
      cv2 = optional_import('cv2')
      if cv2 is None:
        raise SystemExit(77)
      from crappy import resources

      expected = {
        'speckle': ((512, 512),
          'c9f93c30108f458005452e8704cce999e6f5b5829940cbd41861f20a13626409'),
        've_markers': ((415, 632),
          '4cd531a10d497512b6b3b88b1a9d1909fb22ad1c89f1b04521d859ec7f9ab51e'),
        'pad': ((1039, 1418),
          '8cca4268f794cb13efd705e97841e8c3b77c569c7ad6219269ecfaf0effa168b'),
      }
      assert set(resources.paths) == set(expected)
      aliases = {
        'speckle': ('speckle_path', 'ref_speckle', 'path_speckle'),
        've_markers': ('ve_markers_path', 'ref_ve', 'path_ve'),
        'pad': ('pad_path', 'ref_pad', 'path_pad'),
      }
      for name, (shape, digest) in expected.items():
        image = getattr(resources, name)
        assert image.shape == shape, name
        assert str(image.dtype) == 'uint8', name
        assert hashlib.sha256(image.tobytes()).hexdigest() == digest, name
        path = Path(resources.paths[name])
        assert path.is_file(), path
        for alias in aliases[name]:
          assert Path(getattr(resources, alias)).samefile(path), alias
    ''')

  def test_wildcard_imports_include_the_same_resource_class(self):
    self._run_script('''
      import sys
      import crappy
      import crappy._global as common

      assert 'crappy._resources' not in sys.modules
      root_exports = {}
      exec('from crappy import *', root_exports)
      legacy_exports = {}
      exec('from crappy._global import *', legacy_exports)

      assert root_exports['resources'] is legacy_exports['resources']
      assert root_exports['resources'] is crappy.resources is common.resources
      assert legacy_exports['OptionalModule'] is crappy.OptionalModule
      assert legacy_exports['docs'] is crappy.docs
    ''')

  def test_missing_opencv_keeps_paths_and_the_original_import_failure(self):
    failures = (
      "ModuleNotFoundError('cv2 intentionally unavailable', name='cv2')",
      "ImportError('native loader failure', name='cv2', path='cv2.so')",
    )
    for failure in failures:
      with self.subTest(failure=failure):
        self._run_script(f'''
          import builtins
          import sys
          from pathlib import Path

          original_import = builtins.__import__
          cause = {failure}
          attempts = []
          def without_cv2(name, *args, **kwargs):
            if name == 'cv2' or name.startswith('cv2.'):
              attempts.append(name)
              raise cause
            return original_import(name, *args, **kwargs)
          builtins.__import__ = without_cv2

          import crappy
          from crappy._exceptions import OptionalDependencyError
          from crappy._global import OptionalModule
          assert 'crappy._resources' not in sys.modules
          from crappy import resources
          assert resources is crappy._global.resources
          assert set(resources.paths) == {{'pad', 'speckle', 've_markers'}}
          for path in resources.paths.values():
            assert Path(path).is_file(), path

          import_count = len(attempts)
          for name in ('pad', 'speckle', 've_markers'):
            placeholder = getattr(resources, name)
            assert isinstance(placeholder, OptionalModule), name
            assert placeholder.is_available() is False
            try:
              placeholder.resolve()
            except OptionalDependencyError as error:
              assert error.import_name == 'cv2', error
              assert error.install_name == 'opencv-python', error
              assert error.__cause__ is cause, error
            else:
              raise AssertionError('Missing OpenCV was silently accepted')
          assert len(attempts) == import_count
        ''')

  def test_decode_failure_propagates_and_resource_loading_can_be_retried(self):
    self._run_script('''
      import sys
      from unittest.mock import patch
      import crappy
      import crappy._global as common
      from tests.support.dependencies import optional_import
      cv2 = optional_import('cv2')
      if cv2 is None:
        raise SystemExit(77)

      failure = ValueError('resource decoding failed')
      with patch.object(cv2, 'imdecode', side_effect=failure):
        try:
          crappy.resources
        except ValueError as error:
          assert error is failure
        else:
          raise AssertionError('Decode failure was hidden')

      assert 'resources' not in vars(crappy)
      assert 'resources' not in vars(common)
      assert 'crappy._resources' not in sys.modules
      assert crappy.resources is common.resources
      assert crappy.resources.speckle.shape == (512, 512)
    ''')

  def test_legacy_pickle_reference_resolves_the_resource_class(self):
    self._run_script('''
      import pickle
      import sys
      import crappy

      assert 'crappy._resources' not in sys.modules
      legacy_reference = b'ccrappy._global\\nresources\\n.'
      assert pickle.loads(legacy_reference) is crappy.resources
      assert pickle.loads(pickle.dumps(crappy.resources)) is crappy.resources
    ''')
