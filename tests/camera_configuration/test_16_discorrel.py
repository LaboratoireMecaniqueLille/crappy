# coding: utf-8

from copy import deepcopy
from time import sleep
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.dis_correl_config import DISCorrelConfig
from crappy.tool.camera_config import Box


class TestDISCorrel(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def customSetUp(self) -> None:
    """"""

    self._config = DISCorrelConfig(self._camera, self._log_queue,
                                   self._log_level, self._freq, Box())

    self._config._testing = True
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def test_discorrel(self) -> None:
    """"""

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=50, y=50)
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Release the mouse button to complete the box
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=50, y=50)
    self._config._upd_sched()

    # The end point should now be defined and the box is complete
    self.assertFalse(self._config.box.no_points())
    self.assertIsNotNone(self._config.box.x_start)
    self.assertIsNotNone(self._config.box.y_start)
    self.assertIsNotNone(self._config.box.x_end)
    self.assertIsNotNone(self._config.box.y_end)

    box = deepcopy(self._config.box)

    # Reset the box
    self._config.box.reset()
    self._config._upd_sched()

    # The box should now have been reset
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # Release the mouse button at the same location
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=20, y=20)
    self._config._upd_sched()
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # The box should be empty
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # Draw a box with no pixels inside
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=20, y=50)
    self._config._upd_sched()
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=20, y=50)
    self._config._upd_sched()

    # The box should be empty
    self.assertTrue(self._config.box.no_points())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._correl_box = box
