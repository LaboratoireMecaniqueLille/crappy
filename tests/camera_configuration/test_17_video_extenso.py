# coding: utf-8

from copy import deepcopy
from time import sleep
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSpots)
from crappy.tool.camera_config.video_extenso_config import VideoExtensoConfig
from crappy.tool.camera_config import SpotsDetector, Box


class TestVideoExtenso(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSpots(), **kwargs)

  def customSetUp(self) -> None:
    """"""

    self._config = VideoExtensoConfig(self._camera, self._log_queue,
                                      self._log_level, self._freq,
                                      SpotsDetector())

    self._config._testing = True
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def customTearDown(self) -> None:
    """"""

    self._config._detector.spots.spot_1 = Box(0, 100, 0, 100)

  def test_video_extenso(self) -> None:
    """"""

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Get the width of the canvas
    height = self._config._img_canvas.winfo_height()

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now",
        x=-int(0.02 * height), y=-int(0.02 * height))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    print(self._config._select_box)

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now",
        x=int(0.08 * height), y=int(0.08 * height))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    print(self._config._select_box)

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=int(0.1 * height), y=int(0.1 * height))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    print(self._config._select_box)

    # Move the mouse iteratively in case a border is hit
    for i in range(int(0.1 * height), int(0.9 * height), int(0.1 * height)):
      self._config._img_canvas.event_generate(
          '<B1-Motion>', when="now", x=i, y=i)
      self._config._upd_sched()

      print(self._config._select_box)

    # Release the mouse button to complete the box
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now",
        x=int(0.9 * height), y=int(0.9 * height))
    self._config._upd_sched()

    print(self._config._select_box)

    # The spots should have been populated now
    self.assertFalse(self._config._spots.empty())
    self.assertIsInstance(self._config._spots.spot_1, Box)
    self.assertIsInstance(self._config._spots.spot_2, Box)
    self.assertIsInstance(self._config._spots.spot_3, Box)
    self.assertIsInstance(self._config._spots.spot_4, Box)

    # Check that the initial lengths have not been set
    self.assertIsNone(self._config._spots.x_l0)
    self.assertIsNone(self._config._spots.y_l0)

    # Click on the save L0 button
    self._config._update_button.invoke()

    # Check that the initial lengths have been set
    self.assertIsNotNone(self._config._spots.x_l0)
    self.assertIsNotNone(self._config._spots.y_l0)

    spots = deepcopy(self._config._detector.spots)

    # Reset the box
    self._config._detector.spots.reset()
    self._config._upd_sched()

    # The box should now have been reset
    self.assertTrue(self._config._detector.spots.empty())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._detector.spots = spots
