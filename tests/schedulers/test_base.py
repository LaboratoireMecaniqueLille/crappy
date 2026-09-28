# coding: utf-8

from unittest import TestCase
from unittest.mock import Mock, patch
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

  def test_default_reset_and_lazy_logging(self) -> None:
    """The base reset is safe and the logger is created only when used."""

    for cls in (ProbeOutput, ProbeCondition):
      with self.subTest(cls=cls):
        helper = cls()
        self.assertIsNone(helper._logger)
        helper.reset()
        logger = Mock()
        with patch('logging.getLogger', return_value=logger) as get_logger:
          helper.log(logging.INFO, 'first')
          helper.log(logging.DEBUG, 'second')
        get_logger.assert_called_once()
        self.assertTrue(get_logger.call_args.args[0].endswith(cls.__name__))
        self.assertEqual(logger.log.call_count, 2)
