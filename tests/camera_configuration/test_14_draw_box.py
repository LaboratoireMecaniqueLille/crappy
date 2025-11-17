# coding: utf-8

from time import sleep
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.camera_config_boxes import CameraConfigBoxes


class TestDrawBox(ConfigurationWindowTestBase):
  """Class for testing the 
  :class:`~crappy.tool.camera_config_boxes.CameraConfigBoxes` class.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface and for
    adding bindings, otherwise the test wouldn't work."""

    self._config = CameraConfigBoxes(self._camera, self._log_queue,
                                     self._log_level, self._freq)

    self._config._img_canvas.bind('<ButtonPress-1>', self._config._start_box)
    self._config._img_canvas.bind('<B1-Motion>', self._config._extend_box)

    self._config._testing = True
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def test_draw_box(self) -> None:
    """Tests whether the selection box is correctly displayed on the image when
    drawing it."""

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # There should be no points for now
    self.assertTrue(self._config._select_box.no_points())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config._upd_sched()

    # The start point should not have been counted
    self.assertTrue(self._config._select_box.no_points())

    # Start drawing a box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # The start point should have been counted but not the end point
    # The box should still be considered as not complete
    self.assertTrue(self._config._select_box.no_points())
    self.assertIsNotNone(self._config._select_box.x_start)
    self.assertIsNotNone(self._config._select_box.y_start)
    self.assertIsNone(self._config._select_box.x_end)
    self.assertIsNone(self._config._select_box.y_end)

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=50, y=50)
    self._config._upd_sched()

    # The end point should now be defined and the box is complete
    self.assertFalse(self._config._select_box.no_points())
    self.assertIsNotNone(self._config._select_box.x_start)
    self.assertIsNotNone(self._config._select_box.y_start)
    self.assertIsNotNone(self._config._select_box.x_end)
    self.assertIsNotNone(self._config._select_box.y_end)

    # Reset the box
    self._config._select_box.reset()
    self._config._upd_sched()

    # The box should now have been reset
    self.assertTrue(self._config._select_box.no_points())
