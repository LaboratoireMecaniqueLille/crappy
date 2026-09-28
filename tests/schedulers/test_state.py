# coding: utf-8

from unittest import TestCase

from crappy import blocks
from crappy.blocks import Scheduler, schedulers
from crappy.blocks.schedulers import State, conditions, outputs


def output(_dt, _data):
  """Small valid State output."""

  return {'command': 1}


def condition(_dt, _data):
  """Small valid State condition."""

  return False


class TestState(TestCase):
  """Validation and public API tests for Scheduler States."""

  def test_accepts_callables_and_normalizes_sequences(self) -> None:
    """The outer output and transition sequences become ordered tuples."""

    state = State('drive', [output], [[condition, 'End']])

    self.assertEqual(state.id, 'drive')
    self.assertEqual(state.outputs, (output,))
    self.assertEqual(state.stop_conditions, ([condition, 'End'],))
    self.assertEqual(State('idle', [], []).outputs, ())

  def test_rejects_invalid_ids_outputs_and_transitions(self) -> None:
    """Malformed state declarations fail before the Block runs."""

    for id_, error in (('', ValueError), ('  ', ValueError), (1, TypeError)):
      with self.subTest(id_=id_), self.assertRaises(error):
        State(id_, [], [])

    for invalid in (None, 4, 'output', [output, None]):
      with self.subTest(outputs=invalid), self.assertRaises(TypeError):
        State('drive', invalid, [])

    for invalid in (None, 4, ['End'], [(condition,)],
                    [(condition, 'End', 'extra')], [(None, 'End')],
                    [(condition, '')], [(condition, 1)]):
      with self.subTest(conditions=invalid), self.assertRaises(TypeError):
        State('drive', [], invalid)

  def test_public_exports(self) -> None:
    """The package makes the State and every helper class available."""

    self.assertIs(schedulers.State, State)
    self.assertIs(blocks.Scheduler, Scheduler)
    for package, names in (
        (conditions, ('Condition', 'Delay', 'AllLabel', 'AnyLabel',
                      'AllCondition', 'AnyCondition', 'Compare', 'Crossing')),
        (outputs, ('Output', 'Constant', 'Ramp', 'Triangle', 'Square', 'Sine',
                   'FromFile'))):
      for name in names:
        with self.subTest(name=name):
          self.assertIs(getattr(schedulers, name), getattr(package, name))
