# coding: utf-8

import logging
from multiprocessing import current_process
from unittest import TestCase

import crappy.modifier as modifier_package
from crappy._global import DefinitionError
from crappy.modifier import Modifier


class TestModifier(TestCase):
  """Unit tests for the Modifier base class and package registry."""

  def tearDown(self) -> None:
    """Removes dynamic classes registered during tests."""

    for name in ('TestRegisteredModifier', 'TestDuplicateModifier',
                 'TestEvaluateModifier', 'TestPassthroughModifier'):
      Modifier.classes.pop(name, None)

  def test_base_modifier_is_abstract(self) -> None:
    """Checks that Modifier cannot be instantiated directly."""

    with self.assertRaises(TypeError):
      Modifier()

  def test_subclasses_are_registered_by_name(self) -> None:
    """Checks that new Modifier subclasses are added to the registry."""

    Modifier.classes.pop('TestRegisteredModifier', None)

    cls = type('TestRegisteredModifier',
               (Modifier,),
               {'__call__': lambda self, data: data})

    self.assertIs(Modifier.classes['TestRegisteredModifier'], cls)

  def test_duplicate_subclass_names_are_rejected(self) -> None:
    """Checks that two Modifiers cannot share a class name."""

    Modifier.classes.pop('TestDuplicateModifier', None)
    type('TestDuplicateModifier',
         (Modifier,),
         {'__call__': lambda self, data: data})

    with self.assertRaisesRegex(DefinitionError, 'A Modifier'):
      type('TestDuplicateModifier',
           (Modifier,),
           {'__call__': lambda self, data: data})

  def test_deprecated_evaluate_method_is_rejected(self) -> None:
    """Checks that legacy evaluate methods cannot define a Modifier."""

    with self.assertRaisesRegex(DefinitionError, 'evaluate method'):
      type('TestEvaluateModifier',
           (Modifier,),
           {
             'evaluate': lambda self, data: data,
             '__call__': lambda self, data: data,
           })

  def test_init_accepts_unused_args_and_sets_logger(self) -> None:
    """Checks base initialization with ignored arguments."""

    cls = type('TestPassthroughModifier',
               (Modifier,),
               {'__call__': lambda self, data: data})

    modifier = cls('unused', option='ignored')

    self.assertIsNone(modifier._logger)

  def test_log_initializes_process_scoped_logger(self) -> None:
    """Checks logger naming and lazy initialization."""

    cls = type('TestPassthroughModifier',
               (Modifier,),
               {'__call__': lambda self, data: data})
    modifier = cls()
    logger_name = f"{current_process().name}.TestPassthroughModifier"

    with self.assertLogs(logger_name, level='INFO') as captured:
      modifier.log(logging.INFO, "message")

    self.assertIs(modifier._logger, logging.getLogger(logger_name))
    self.assertEqual(captured.output, [f"INFO:{logger_name}:message"])

  def test_base_call_can_be_reused_by_subclasses(self) -> None:
    """Checks the fallback implementation available through super."""

    def passthrough(self, data):
      return super(type(self), self).__call__(data)

    cls = type('TestPassthroughModifier',
               (Modifier,),
               {'__call__': passthrough})
    modifier = cls()
    logs = list()
    modifier.log = lambda level, msg: logs.append((level, msg))
    data = {'x': 1}

    self.assertIs(modifier(data), data)
    self.assertEqual([level for level, _ in logs],
                     [logging.DEBUG, logging.WARNING, logging.DEBUG])
    self.assertIn('Received', logs[0][1])
    self.assertIn('__call__ method is not defined', logs[1][1])
    self.assertIn('Sending', logs[2][1])

  def test_package_modifier_dict_is_base_registry(self) -> None:
    """Checks package-level registry exports."""

    self.assertIs(modifier_package.modifier_dict, Modifier.classes)
    for name in ('Demux', 'Diff', 'DownSampler', 'Integrate', 'Mean',
                 'Median', 'MovingAvg', 'MovingMed', 'Offset',
                 'TrigOnChange', 'TrigOnValue'):
      self.assertIs(modifier_package.modifier_dict[name],
                    getattr(modifier_package, name))
