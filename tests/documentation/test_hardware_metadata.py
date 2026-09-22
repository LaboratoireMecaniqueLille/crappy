# coding: utf-8

from datetime import date
from pathlib import Path
from re import findall, MULTILINE
from runpy import run_path
import unittest


class TestHardwareMetadata(unittest.TestCase):
  """Keeps the hardware inventory complete and internally consistent."""

  _repository = Path(__file__).resolve().parents[2]
  _metadata_path = (_repository / 'docs' / 'source' / '_data' /
                    'hardware.py')
  _camera_api_path = (_repository / 'docs' / 'source' / 'crappy_docs' /
                      'cameras.rst')
  _inout_api_path = (_repository / 'docs' / 'source' / 'crappy_docs' /
                     'inouts.rst')
  _actuator_api_path = (_repository / 'docs' / 'source' / 'crappy_docs' /
                        'actuators.rst')
  _required_fields = {
    'name',
    'object',
    'kind',
    'distribution',
    'platforms',
    'transport',
    'dependencies',
    'backends',
    'maintenance',
    'verification',
    'verified_on',
    'verification_details',
    'example',
  }

  @classmethod
  def setUpClass(cls) -> None:
    cls._entries = run_path(str(cls._metadata_path))['HARDWARE']

  def test_schema_and_statuses(self) -> None:
    """Checks fields, statuses, dates, and identifiers."""

    names = set()
    objects = set()
    maintenance = {'core': 'maintained', 'collection': 'legacy'}

    for entry in self._entries:
      with self.subTest(driver=entry.get('name')):
        self.assertEqual(set(entry), self._required_fields)
        self.assertIn(entry['kind'], {'Camera', 'InOut', 'Actuator'})
        self.assertEqual(entry['maintenance'],
                         maintenance[entry['distribution']])
        self.assertIn(entry['verification'],
                      {'verified', 'unverified', 'software-only'})
        self.assertTrue(entry['verification_details'])
        if entry['verified_on'] is not None:
          self.assertEqual(entry['verification'], 'verified')
          date.fromisoformat(entry['verified_on'])
        else:
          self.assertNotEqual(entry['verification'], 'verified')

        name_key = (entry['kind'], entry['name'])
        self.assertNotIn(name_key, names)
        self.assertNotIn(entry['object'], objects)
        names.add(name_key)
        objects.add(entry['object'])

  def test_camera_api_inventory_is_complete(self) -> None:
    """Checks every Camera driver API entry has one metadata row."""

    source = self._camera_api_path.read_text(encoding='utf-8')
    driver_source = source.split('Parent Camera', maxsplit=1)[0]
    documented = set(findall(
      r'^\.\. autoclass:: (crappy\.(?:camera|collection\.camera)\.\S+)$',
      driver_source,
      flags=MULTILINE,
    ))
    inventoried = {
      entry['object'] for entry in self._entries if entry['kind'] == 'Camera'
    }

    self.assertEqual(documented, inventoried)

  def test_inout_api_inventory_is_complete(self) -> None:
    """Checks every InOut driver API entry has one metadata row."""

    source = self._inout_api_path.read_text(encoding='utf-8')
    driver_source = source.split('Parent In/Out', maxsplit=1)[0]
    documented = set(findall(
      r'^\.\. autoclass:: (crappy\.(?:inout|collection\.inout)\.\S+)$',
      driver_source,
      flags=MULTILINE,
    ))
    inventoried = {
      entry['object'] for entry in self._entries if entry['kind'] == 'InOut'
    }

    self.assertEqual(documented, inventoried)

  def test_actuator_api_inventory_is_complete(self) -> None:
    """Checks every Actuator driver API entry has one metadata row."""

    source = self._actuator_api_path.read_text(encoding='utf-8')
    driver_source = source.split('Parent Actuator', maxsplit=1)[0]
    documented = set(findall(
      r'^\.\. autoclass:: (crappy\.(?:actuator|collection\.actuator)\.\S+)$',
      driver_source,
      flags=MULTILINE,
    ))
    inventoried = {
      entry['object'] for entry in self._entries
      if entry['kind'] == 'Actuator'
    }

    self.assertEqual(documented, inventoried)

  def test_example_paths_exist(self) -> None:
    """Checks optional example links remain repository-relative and valid."""

    for entry in self._entries:
      if entry['example'] is None:
        continue
      with self.subTest(driver=entry['name']):
        self.assertTrue(entry['example'].startswith('examples/'))
        self.assertTrue((self._repository / entry['example']).is_file())
