"""Public validation and protected state for this feature."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import crappy.blocks.generator_path.ramp as ramp_module
import warnings
import numpy as np
from crappy.blocks.generator_path import Constant, Cyclic, CyclicRamp, Integrator, Ramp, Sine
from crappy.blocks.generator_path.custom import Custom
from tests.support.generator_paths import GeneratorPathTestBase


class TestPathValidation(GeneratorPathTestBase):
  def test_paths_reject_nonfinite_numbers_and_invalid_conditions(self):
    constructors = (lambda value: Constant(None, value),
                    lambda value: Ramp(None, value, init_value=0),
                    lambda value: Sine(None, value, 1),
                    lambda value: Cyclic(None, None, 0, 1, cycles=value),
                    lambda value: Integrator(None, value, 'x', init_value=0))
    for constructor in constructors:
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(constructor=constructor, value=value):
          with self.assertRaises((TypeError, ValueError)):
            constructor(value)
    for condition, error in ((object(), TypeError), ('x>nan', ValueError),
                              ('x<inf', ValueError), ('delay=-1', ValueError)):
      with self.subTest(condition=condition):
        with self.assertRaises(error):
          Constant(condition, 0)

  def test_signed_paths_and_half_cycles_remain_supported(self):
    ramp = Ramp(None, -1, init_value=-2)
    with patch.object(ramp_module, 'time', return_value=self.path_t0 + 2):
      self.assertEqual(ramp.get_cmd({}), -4)
    integral = Integrator(None, -1, 'x', init_value=0)
    self.assertEqual(integral.get_cmd({'t(s)': [0, 1], 'x': [2, 2]}), -2)
    self.assertEqual(Cyclic(None, None, -1, 1, cycles=0.5).get_cmd({}), -1)
    with patch('crappy.blocks.generator_path.cyclic_ramp.time',
               return_value=self.path_t0):
      self.assertEqual(CyclicRamp(None, None, -1, 1, cycles=-1,
                                  init_value=0).get_cmd({}), 0)
    with self.assertRaises((TypeError, ValueError)):
      Cyclic(None, None, 0, 1, cycles=0.25)

  def test_custom_paths_reject_empty_or_nonfinite_data(self):
    with TemporaryDirectory() as folder:
      filename = Path(folder) / 'path.csv'
      for contents in ('', '0,0\n1,nan\n'):
        filename.write_text(contents)
        with self.subTest(contents=contents), warnings.catch_warnings(record=True):
          with self.assertRaises((TypeError, ValueError)):
            Custom(filename)

  def test_filenames_require_a_name_for_both_strings_and_path_objects(self):
    root = Path(Path.cwd().anchor)
    for value in ('', ' ', '.', './', str(root), Path(), root, Path(' '),
                  1, b'data', object()):
      with self.subTest(value=value):
        with self.assertRaises((TypeError, ValueError)):
          Custom(value)

  def test_input_filenames_reject_directories(self):
    with TemporaryDirectory() as folder:
      for value in (folder, Path(folder)):
        with self.subTest(value=value):
          with self.assertRaises(OSError):
            Custom(value)
