"""Collection implementations loaded by name, then reconstructed by workers."""

from tests.unit.framework.lazy_startup_fixtures import (StartupActuator, StartupCamera,
                                                 StartupInOut)


class CollectionStartupActuator(StartupActuator):
  pass


class CollectionStartupCamera(StartupCamera):
  pass


class CollectionStartupInOut(StartupInOut):
  pass
