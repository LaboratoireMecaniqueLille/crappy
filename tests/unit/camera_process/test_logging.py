# coding: utf-8

from multiprocessing import get_start_method
import logging
import logging.handlers

from tests.support.camera_processes import CameraProcessTestBase, TestCameraProcess


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

    self._process._set_logger()

    self.assertIs(self._process._logger, logger)
    self.assertEqual(logger.level, logging.ERROR)

    if get_start_method() != 'fork':
      self.assertTrue(any(isinstance(handler, logging.handlers.QueueHandler)
                          for handler in logger.handlers))

    self.assertIs(self._process._log_queue, shared.log_queue)

  def test_disable_logging(self) -> None:
    """Tests that passing None only disables this process's logger."""

    self._process = TestCameraProcess()
    self.make_shared(log_level=None)

    disable = logging.root.manager.disable
    self._process._set_logger()

    self.assertIsNotNone(self._process._logger)
    self.assertTrue(self._process._logger.disabled)
    self.assertEqual(logging.root.manager.disable, disable)
