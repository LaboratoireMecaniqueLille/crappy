# coding: utf-8

from dataclasses import FrozenInstanceError
from unittest import TestCase
from unittest.mock import patch

import crappy.collection.api as collection_api
import crappy._collection as collection_module
from crappy._collection import (CollectionEntry, CollectionRegistry,
                                CollectionUnavailableError,
                                collection_registry, load_collection_class)
from crappy.actuator._collection import moved_to_collection as moved_actuators
from crappy.camera._collection import moved_to_collection as moved_cameras
from crappy.collection.manifest import DRIVERS
from crappy.inout._collection import moved_to_collection as moved_inouts
from crappy.inout.meta_inout import InOut


class TestCollectionEntry(TestCase):
  """Unit tests for collection manifest entries."""

  def test_entry_is_an_immutable_value_object(self) -> None:
    """Checks stored metadata and the immutable public contract."""

    entry = CollectionEntry('Device', 'InOut', 'package.device')

    self.assertEqual(entry.name, 'Device')
    self.assertEqual(entry.kind, 'InOut')
    self.assertEqual(entry.module, 'package.device')
    with self.assertRaises(FrozenInstanceError):
      entry.name = 'Other'

  def test_invalid_metadata_is_rejected(self) -> None:
    """Checks validation of names, hardware kinds, and module types."""

    with self.assertRaises(ValueError):
      CollectionEntry('', 'InOut', 'package.device')
    with self.assertRaises(ValueError):
      CollectionEntry('Device', 'Unknown', 'package.device')
    with self.assertRaises(ValueError):
      CollectionEntry('Device', 'InOut', None)

  def test_empty_module_is_rejected(self) -> None:
    """Regression: a manifest entry must identify a module to import."""

    with self.assertRaises(ValueError):
      CollectionEntry('Device', 'InOut', '')


class TestCollectionRegistry(TestCase):
  """Unit tests for registering and discovering collection entries."""

  def setUp(self) -> None:
    """Creates an isolated registry and representative entries."""

    self.registry = CollectionRegistry()
    self.actuator = CollectionEntry('Shared', 'Actuator', 'pkg.actuator')
    self.camera = CollectionEntry('Shared', 'Camera', 'pkg.camera')
    self.inout = CollectionEntry('Input', 'InOut', 'pkg.inout')

  def test_register_get_and_find(self) -> None:
    """Checks lookup keys and identical names across hardware kinds."""

    self.registry.register(self.camera, self.actuator, self.inout)

    self.assertIs(self.registry.get('Actuator', 'Shared'), self.actuator)
    self.assertIsNone(self.registry.get('InOut', 'Missing'))
    self.assertCountEqual(self.registry.find('Shared'),
                          (self.camera, self.actuator))

  def test_registration_is_idempotent_but_rejects_conflicts(self) -> None:
    """Checks duplicate handling and registry type safety."""

    self.registry.register(self.actuator, self.actuator)
    self.assertEqual(self.registry.entries('Actuator'), (self.actuator,))

    conflict = CollectionEntry('Shared', 'Actuator', 'other.actuator')
    with self.assertRaises(RuntimeError):
      self.registry.register(conflict)
    with self.assertRaises(TypeError):
      self.registry.register('not an entry')

  def test_entries_can_be_filtered_and_are_sorted(self) -> None:
    """Checks deterministic listing for a requested hardware kind."""

    second = CollectionEntry('Zed', 'InOut', 'pkg.zed')
    self.registry.register(second, self.inout, self.camera)

    self.assertEqual(self.registry.entries('InOut'), (self.inout, second))
    self.assertEqual(self.registry.entries('Camera'), (self.camera,))

  def test_entries_without_kind_returns_every_entry(self) -> None:
    """Regression: omitting the optional kind should list all entries."""

    self.registry.register(self.inout, self.actuator, self.camera)

    self.assertEqual(self.registry.entries(),
                     (self.actuator, self.camera, self.inout))

  def test_lookup_arguments_are_validated(self) -> None:
    """Checks that malformed lookup arguments fail clearly."""

    with self.assertRaises(ValueError):
      self.registry.get('Unknown', 'Device')
    with self.assertRaises(ValueError):
      self.registry.get('InOut', '')
    with self.assertRaises(ValueError):
      self.registry.find('')
    with self.assertRaises(ValueError):
      self.registry.entries('Unknown')


class TestCollectionLoader(TestCase):
  """Unit tests for lazy import and class validation."""

  def setUp(self) -> None:
    """Creates a synthetic entry without importing optional dependencies."""

    self.entry = CollectionEntry('Device', 'InOut', 'package.device')

  def test_successful_import_returns_the_registered_class(self) -> None:
    """Checks the successful lazy-loading path."""

    cls = type('Device', (), {'__module__': self.entry.module})
    classes = {'Device': cls}

    with patch.object(collection_module, 'import_module') as mocked_import:
      self.assertIs(load_collection_class(self.entry, classes), cls)

    mocked_import.assert_called_once_with(self.entry.module)

  def test_import_failure_is_wrapped_with_the_original_cause(self) -> None:
    """Checks diagnostics when an optional driver cannot be imported."""

    cause = ImportError('missing dependency')
    with patch.object(collection_module, 'import_module', side_effect=cause):
      with self.assertRaises(CollectionUnavailableError) as raised:
        load_collection_class(self.entry, {})

    self.assertIs(raised.exception.entry, self.entry)
    self.assertIs(raised.exception.cause, cause)
    self.assertIs(raised.exception.__cause__, cause)
    self.assertIn('missing dependency', str(raised.exception))

  def test_missing_or_wrong_registration_is_rejected(self) -> None:
    """Checks post-import validation of the class registry."""

    wrong_cls = type('Device', (), {'__module__': 'other.module'})

    for classes, message in (({}, 'did not register'),
                             ({'Device': wrong_cls}, 'other.module')):
      with self.subTest(classes=classes):
        with patch.object(collection_module, 'import_module'):
          with self.assertRaises(CollectionUnavailableError) as raised:
            load_collection_class(self.entry, classes)
        self.assertIsInstance(raised.exception.cause, RuntimeError)
        self.assertIn(message, str(raised.exception.cause))


class TestCollectionApi(TestCase):
  """Unit tests for the dependency-independent public collection API."""

  def setUp(self) -> None:
    """Creates a compact synthetic manifest."""

    self.actuator = CollectionEntry('Motor', 'Actuator', 'pkg.motor')
    self.camera = CollectionEntry('Imager', 'Camera', 'pkg.imager')
    self.inout = CollectionEntry('Sensor', 'InOut', 'pkg.sensor')
    self.manifest = (self.actuator, self.camera, self.inout)

  def test_drivers_filters_without_loading_driver_modules(self) -> None:
    """Checks manifest discovery by hardware kind."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      self.assertEqual(collection_api.drivers('Camera'), (self.camera,))
      self.assertEqual(collection_api.drivers('InOut'), (self.inout,))
      with self.assertRaises(ValueError):
        collection_api.drivers('Unknown')

  def test_drivers_without_kind_lists_the_manifest(self) -> None:
    """Regression: the documented default should return every driver."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      self.assertEqual(collection_api.drivers(), self.manifest)

  def test_check_reports_success_and_unavailability(self) -> None:
    """Checks structured results for both lazy-loader outcomes."""

    with patch.object(collection_api, 'DRIVERS', (self.inout,)):
      with patch.object(collection_api, 'load_collection_class') as loader:
        result = collection_api.check('Sensor', 'InOut')

      self.assertTrue(result.available)
      self.assertIsNone(result.error_type)
      self.assertIsNone(result.error_message)
      loader.assert_called_once_with(self.inout, InOut.classes)

      cause = ModuleNotFoundError('optional_package')
      unavailable = CollectionUnavailableError(self.inout, cause)
      with patch.object(collection_api, 'load_collection_class',
                        side_effect=unavailable):
        result = collection_api.check('Sensor', 'InOut')

    self.assertFalse(result.available)
    self.assertEqual(result.error_type, 'ModuleNotFoundError')
    self.assertEqual(result.error_message, 'optional_package')

  def test_check_rejects_unknown_names_and_mismatched_kinds(self) -> None:
    """Checks explicit lookup failures before attempting an import."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      with self.assertRaises(ValueError):
        collection_api.check('Missing', 'Camera')
      with self.assertRaises(ValueError):
        collection_api.check('Sensor', 'Camera')

  def test_check_without_kind_finds_a_unique_driver(self) -> None:
    """Regression: kind is optional when a driver name is unambiguous."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      with patch.object(collection_api, 'load_collection_class'):
        result = collection_api.check('Sensor')

    self.assertTrue(result.available)
    self.assertEqual(result.kind, 'InOut')

  def test_check_without_kind_rejects_an_ambiguous_name(self) -> None:
    """Checks that callers must disambiguate names shared across kinds."""

    actuator = CollectionEntry('Shared', 'Actuator', 'pkg.actuator')
    camera = CollectionEntry('Shared', 'Camera', 'pkg.camera')
    with patch.object(collection_api, 'DRIVERS', (actuator, camera)):
      with self.assertRaisesRegex(ValueError, 'Specify the kind'):
        collection_api.check('Shared')

  def test_check_all_filters_and_preserves_manifest_order(self) -> None:
    """Checks batch lookup without importing real hardware drivers."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      with patch.object(collection_api, 'check',
                        side_effect=lambda name, kind: (name, kind)):
        results = collection_api.check_all('Actuator')

    self.assertEqual(results, (('Motor', 'Actuator'),))

  def test_check_all_without_kind_checks_the_entire_manifest(self) -> None:
    """Checks that the documented default covers every hardware kind."""

    with patch.object(collection_api, 'DRIVERS', self.manifest):
      with patch.object(collection_api, 'check',
                        side_effect=lambda name, kind: (name, kind)):
        results = collection_api.check_all()

    self.assertEqual(results, (('Motor', 'Actuator'),
                               ('Imager', 'Camera'),
                               ('Sensor', 'InOut')))


class TestCollectionManifest(TestCase):
  """Consistency checks for the real collection manifests."""

  def test_manifest_entries_are_unique_registered_and_match_moved_names(
      self) -> None:
    """Checks the hand-maintained manifest and migration lists stay aligned."""

    keys = tuple((entry.kind, entry.name) for entry in DRIVERS)
    self.assertEqual(len(keys), len(set(keys)))

    expected = {'Actuator': set(moved_actuators),
                'Camera': set(moved_cameras),
                'InOut': set(moved_inouts)}
    for kind, names in expected.items():
      with self.subTest(kind=kind):
        entries = tuple(entry for entry in DRIVERS if entry.kind == kind)
        self.assertEqual({entry.name for entry in entries}, names)
        for entry in entries:
          self.assertIs(collection_registry.get(kind, entry.name), entry)
