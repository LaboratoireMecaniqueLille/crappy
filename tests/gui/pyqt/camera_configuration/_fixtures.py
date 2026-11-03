# coding: utf-8

"""Resources shared by the optional PyQt6 window tests."""

from tests.support.gui import require_gui_group

require_gui_group('pyqt')

import importlib.util
import logging
import unittest
from multiprocessing import Queue
from unittest.mock import Mock
import numpy as np

from crappy.tool.camera_config.pyqt import PyQtCameraConfig
from crappy._failure import FailureManager
from tests.support.cameras import DummyCamera
from tests.support.resources import close_queue


class PreviewCamera(DummyCamera):
  """Provide a fixed frame for Qt pixel and selection-coordinate checks."""

  def __init__(self) -> None:
    super().__init__()
    self.frame = ({}, np.arange(10000, dtype=np.uint8).reshape(100, 100))


@unittest.skipUnless(importlib.util.find_spec('PyQt6'), 'PyQt6 not installed')
class PyQtConfigTestCase(unittest.TestCase):
  """Create optional Qt windows with fail-safe per-test resource cleanup."""

  def setUp(self) -> None:
    super().setUp()
    previous_disabled = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    self.addCleanup(logging.disable, previous_disabled)
    self.camera = PreviewCamera()
    self.log_queue = Queue()
    self.addCleanup(self._close_log_queue)
    self.configs = []

  def make_config(self, configurator=PyQtCameraConfig, *args,
                  histogram_process: bool = False):
    """Construct a window directly; backend selection belongs to Block tests."""

    manager = FailureManager('test configuration', 'configuration')
    config = configurator(self.camera, self.log_queue, None, 30, None,
                          manager, *args)
    self.addCleanup(config.stop)
    if not histogram_process:
      # Only the end-to-end histogram test needs to spawn an actual worker
      # Spec the class to avoid reading an unstarted process's sentinel property
      process = Mock(spec=type(config._histogram_process))
      process.is_alive.return_value = False
      config._histogram_process = process
      config._lifecycle._histogram_process = process
    self.configs.append(config)
    return config

  def _close_log_queue(self):
    close_queue(self.log_queue)
