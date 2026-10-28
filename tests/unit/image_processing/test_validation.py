"""Public validation and protected state for this feature."""

from unittest import TestCase
import numpy as np
from crappy.tool import ApplyStrainToImage


class TestStrainValidation(TestCase):
  def test_strain_rejects_singular_inputs(self):
    with self.assertRaises((TypeError, ValueError)):
      ApplyStrainToImage(np.ones((1, 2)))
    strain = ApplyStrainToImage(np.ones((2, 2, 1), dtype=np.uint8))
    with self.assertRaises((TypeError, ValueError)):
      strain(-100, 0)
