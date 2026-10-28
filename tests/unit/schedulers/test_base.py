# coding: utf-8

from unittest import TestCase
import logging

from crappy.blocks.schedulers.conditions import Condition
from crappy.blocks.schedulers.outputs import Output


class ProbeOutput(Output):
  """Concrete Output for checking inherited hooks."""

  def __call__(self, dt, data):
    return {'value': dt}


class ProbeCondition(Condition):
  """Concrete Condition for checking inherited hooks."""

  def __call__(self, dt, data):
    return bool(data)


class TestBaseHelpers(TestCase):
  """Checks the contracts shared by all output and condition helpers."""

  def test_base_classes_are_abstract(self) -> None:
    """Users must provide a callable implementation."""

    for cls in (Output, Condition):
      with self.subTest(cls=cls), self.assertRaises(TypeError):
        cls()

  def test_reset_and_log_forwarding_remain_usable(self) -> None:
    """Reset keeps the helper usable; log calls preserve user messages/levels."""

    for cls in (ProbeOutput, ProbeCondition):
      with self.subTest(cls=cls):
        helper = cls()
        expected = helper(2, {'value': [1]})
        helper.reset()
        self.assertEqual(helper(2, {'value': [1]}), expected)
        with self.assertLogs(level='INFO') as captured:
          helper.log(logging.INFO, 'first')
          helper.log(logging.WARNING, 'second')
        messages = [(record.levelno, record.getMessage()) for record in captured.records]
        self.assertIn((logging.INFO, 'first'), messages)
        self.assertIn((logging.WARNING, 'second'), messages)
