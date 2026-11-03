"""Isolated public timing/command context for Generator Path tests."""

import logging
from unittest import TestCase

from crappy.blocks.generator_path.meta_path import Path


class GeneratorPathTestBase(TestCase):
  path_t0 = 10

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(setattr, Path, 't0', Path.t0)
    self.addCleanup(setattr, Path, 'last_cmd', Path.last_cmd)
    self.addCleanup(logging.disable, logging.root.manager.disable)
    Path.t0, Path.last_cmd = self.path_t0, None
    logging.disable(logging.NOTSET)
