# coding: utf-8

from .._global import DefinitionError


class MetaActuator(type):
  """This metaclass will just check if the actuator are defined properly.

  They must have at least an ``open``, a ``stop``, a ``close`` and either a
  ``set_speed`` or a ``set_position`` method.
  """

  classes = {}
  needed_methods = ["open", "stop", ('set_speed', 'set_position'), 'close']

  def __new__(mcs, name, bases, dict_):
    return type.__new__(mcs, name, bases, dict_)

  def __init__(cls, name, bases, dict_):
    type.__init__(cls, name, bases, dict_)  # This is the important line
    if name in MetaActuator.classes:
      raise DefinitionError("Cannot redefine " + name + " class")

    if name == "Actuator":
      return
    for m in MetaActuator.needed_methods:
      if isinstance(m, tuple):
        ok = False
        for n in m:
          if n in dict_:
            ok = True
            break
        if not ok:
          raise DefinitionError(
              name + " class needs at least one of these methods: " + str(m))
      else:
        if m not in dict_:
          raise DefinitionError(name + " class needs the method " + str(m))

    MetaActuator.classes[name] = cls


class Actuator(object, metaclass=MetaActuator):
  pass
