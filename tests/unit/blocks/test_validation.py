"""Public validation and protected state for this feature."""

from pathlib import Path
import numpy as np
from crappy.blocks import FakeMachine, Generator, HDFRecorder, Multiplexer, PID, Synchronizer
from tests.support.block_test_case import BlockTestBase


class TestBlockValidation(BlockTestBase):
  def test_generator_validates_flags_and_path_type_names(self):
    path = [{'type': 'Constant', 'value': 0, 'condition': None}]
    for key in ('repeat', 'spam', 'safe_start'):
      with self.subTest(key=key):
        with self.assertRaises((TypeError, ValueError)):
          Generator(path, **{key: 1})
    with self.assertRaises((TypeError, ValueError)):
      Generator(path, end_delay=np.nan)
    with self.assertRaises((TypeError, ValueError)):
      Generator([{'type': []}])

  def test_pid_keeps_infinite_limits_but_rejects_nan(self):
    pid = PID(-1, out_min=np.inf, out_max=-np.inf,
              i_limit=(np.inf, -np.inf))
    self.assertIsNotNone(pid)
    for key in ('kp', 'ki', 'kd', 'out_min', 'out_max'):
      options = {'kp': 1, key: np.nan}
      with self.subTest(key=key):
        with self.assertRaises((TypeError, ValueError)):
          PID(**options)
    with self.assertRaises((TypeError, ValueError)):
      PID(1, i_limit=(1,))

  def test_numeric_blocks_reject_invalid_configuration(self):
    cases = ((Multiplexer, {'interp_freq': np.inf}),
             (Synchronizer, {'reference_label': ''}),
             (FakeMachine, {'l0': 0}),
             (FakeMachine, {'sigma': {'F(N)': -1}}))
    for constructor, options in cases:
      with self.subTest(constructor=constructor, options=options):
        with self.assertRaises((TypeError, ValueError)):
          constructor(**options)

  def test_filenames_require_a_name_for_both_strings_and_path_objects(self):
    root = Path(Path.cwd().anchor)
    for value in ('', ' ', '.', './', str(root), Path(), root, Path(' '),
                  1, b'data', object()):
      with self.subTest(value=value):
        with self.assertRaises((TypeError, ValueError)):
          HDFRecorder(value)
