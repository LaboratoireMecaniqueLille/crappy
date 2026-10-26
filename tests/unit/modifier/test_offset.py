# coding: utf-8

from unittest import TestCase

from crappy.modifier import Offset


class TestOffset(TestCase):
  """Unit tests for the Offset Modifier."""

  def test_single_label_and_scalar_offset_are_supported(self) -> None:
    """Checks compensation from the first received value."""

    modifier = Offset('x', 6)
    first = {'x': 3, 'other': 1}
    second = {'x': 10, 'other': 2}

    self.assertIs(modifier(first), first)
    self.assertEqual(first, {'x': 6, 'other': 1})
    self.assertEqual(modifier._compensations, {'x': 3})
    self.assertTrue(modifier._compensated)

    self.assertEqual(modifier(second), {'x': 13, 'other': 2})

  def test_multiple_labels_are_offset_independently(self) -> None:
    """Checks compensation for several labels."""

    modifier = Offset(('x', 'y'), (0, 10))

    self.assertEqual(modifier({'x': 3, 'y': 8, 'z': 1}),
                     {'x': 0, 'y': 10, 'z': 1})
    self.assertEqual(modifier({'x': 5, 'y': 9, 'z': 2}),
                     {'x': 2, 'y': 11, 'z': 2})

  def test_mismatched_label_and_offset_counts_are_rejected(self) -> None:
    """Checks constructor validation."""

    with self.assertRaises(ValueError):
      Offset(('x', 'y'), (0,))

  def test_missing_label_raises_key_error_before_compensation(self) -> None:
    """Checks that configured labels must be present."""

    modifier = Offset('x', 0)

    with self.assertRaises(KeyError):
      modifier({'y': 1})

    self.assertFalse(modifier._compensated)
