# coding: utf-8

from crappy import Block
from crappy.links import link_graph
from unittest.mock import patch

from .block_test_base import BlockTestBase, TestBlock, link


class TestProperties(BlockTestBase):
  """Tests the validated public properties exposed by Block."""

  def setUp(self) -> None:
    """Creates one Block with all the default property values."""

    super().setUp()
    self._block = TestBlock()

  def test_defaults(self) -> None:
    """Checks the defaults exposed to every Block subclass."""

    self.assertEqual(self._block.niceness, 0)
    self.assertIsNone(self._block.labels)
    self.assertIsNone(self._block.freq)
    self.assertFalse(self._block.display_freq)
    self.assertEqual(self._block.name, 'crappy.TestBlock-1')
    self.assertTrue(self._block.pausable)
    self.assertFalse(self._block.is_vision_block)

  def test_valid_values(self) -> None:
    """Checks valid assignments and their backing values."""

    self._block.niceness = -20
    self._block.labels = ('time', 'value')
    self._block.freq = 25.0
    self._block.display_freq = True
    self._block.pausable = False
    self._block.is_vision_block = True

    self.assertEqual(self._block.niceness, -20)
    self.assertEqual(self._block.labels, ('time', 'value'))
    self.assertEqual(self._block.freq, 25.0)
    self.assertTrue(self._block.display_freq)
    self.assertFalse(self._block.pausable)
    self.assertTrue(self._block.is_vision_block)

    self._block.niceness = 19
    self._block.labels = None
    self._block.freq = True

    self.assertEqual(self._block.niceness, 19)
    self.assertIsNone(self._block.labels)
    self.assertEqual(self._block.freq, 1.0)

    self._block.freq = 50
    self.assertEqual(self._block.freq, 50.0)

    self._block.freq = None
    self.assertIsNone(self._block.freq)

  def test_niceness_validation(self) -> None:
    """Checks niceness type and platform range validation."""

    for value in (-21, 20):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          self._block.niceness = value

    for value in (-1.5, '0', None):
      with self.subTest(value=value):
        with self.assertRaises(TypeError):
          self._block.niceness = value

  def test_labels_validation(self) -> None:
    """Checks label container, element, and uniqueness validation."""

    for value in ('value', {'value'}, 1):
      with self.subTest(value=value):
        with self.assertRaises(TypeError):
          self._block.labels = value

    for value in (('time', 1), ('value', 'value')):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          self._block.labels = value

  def test_freq_validation(self) -> None:
    """Checks frequency type and positive-value validation."""

    for value in ('25', []):
      with self.subTest(value=value):
        with self.assertRaises(TypeError):
          self._block.freq = value

    for value in (0.0, -1.0):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          self._block.freq = value

  def test_boolean_property_validation(self) -> None:
    """Checks that the three boolean properties reject other values."""

    for attribute in ('display_freq', 'pausable', 'is_vision_block'):
      for value in (0, 1, None, 'yes'):
        with self.subTest(attribute=attribute, value=value):
          with self.assertRaises(TypeError):
            setattr(self._block, attribute, value)

  def test_name_updates_global_bookkeeping(self) -> None:
    """Checks renaming and uniqueness across Block instances."""

    other = TestBlock()
    old_name = self._block.name
    link(self._block, other, name='test-link')

    self._block.name = 'custom-name'

    self.assertEqual(self._block.name, 'custom-name')
    self.assertNotIn(old_name, Block.names)
    self.assertIn('custom-name', Block.names)
    self.assertNotIn(old_name, link_graph._nodes)
    self.assertIn('custom-name', link_graph._nodes)
    self.assertEqual(link_graph._edges['test-link'].source, 'custom-name')
    self.assertEqual(link_graph._edges['test-link'].target, other.name)

    with self.assertRaises(ValueError):
      self._block.name = other.name

    self.assertEqual(self._block.name, 'custom-name')
    self.assertEqual(len(Block.names), 2)
    self.assertEqual(link_graph._edges['test-link'].source, 'custom-name')

  def test_name_validation(self) -> None:
    """Checks that Block names are non-empty strings."""

    for value, exception in ((1, TypeError), (None, TypeError),
                             ('', ValueError)):
      with self.subTest(value=value):
        with self.assertRaises(exception):
          self._block.name = value

  def test_running_block_cannot_be_renamed(self) -> None:
    """Checks that rejected runtime renaming leaves registries unchanged."""

    old_name = self._block.name

    with patch.object(self._block, 'is_alive', return_value=True):
      with self.assertRaises(RuntimeError):
        self._block.name = 'runtime-name'

    self.assertEqual(self._block.name, old_name)
    self.assertIn(old_name, Block.names)
    self.assertNotIn('runtime-name', Block.names)
    self.assertIn(old_name, link_graph._nodes)
    self.assertNotIn('runtime-name', link_graph._nodes)
