# coding: utf-8

from multiprocessing import get_start_method
import logging
import logging.handlers

from .camera_process_test_base import CameraProcessTestBase, TestCameraProcess


class TestLogging(CameraProcessTestBase):
  """Tests CameraProcess logging helpers."""

  def test_log_no_logger(self) -> None:
    """Tests that log is a no-op before logger initialization."""

    self._process = TestCameraProcess()

    self._process.log(logging.INFO, "no logger yet")

  def test_set_logger(self) -> None:
    """Tests that _set_logger creates the process logger."""

    self._process = TestCameraProcess()
    shared = self.make_shared(log_level=logging.ERROR)

    logger = logging.getLogger(self._process.name)
    logger.handlers.clear()

    self._process._set_logger()

    self.assertIs(self._process._logger, logger)
    self.assertEqual(logger.level, logging.ERROR)

    if get_start_method() == "spawn":
      self.assertTrue(any(isinstance(handler, logging.handlers.QueueHandler)
                          for handler in logger.handlers))

    self.assertIs(self._process._log_queue, shared.log_queue)

  def test_disable_logging(self) -> None:
    """Tests that passing None disables logging globally."""

    self._process = TestCameraProcess()
    self.make_shared(log_level=None)

    self._process._set_logger()

    self.assertIsNotNone(self._process._logger)
    self.assertGreaterEqual(logging.root.manager.disable, logging.CRITICAL)
