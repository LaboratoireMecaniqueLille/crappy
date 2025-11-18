# coding: utf-8

from copy import deepcopy
from time import sleep
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSpots)
from crappy.tool.camera_config.video_extenso_config import VideoExtensoConfig
from crappy.tool.camera_config import SpotsDetector, Box


class TestVideoExtenso(ConfigurationWindowTestBase):
  """Class for testing the 
  :class:`~crappy.tool.video_extenso_config.VideoExtensoConfig` class.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args, camera=FakeTestCameraSpots(), **kwargs)

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface."""

    self._config = VideoExtensoConfig(self._camera, self._log_queue,
                                      self._log_level, self._freq,
                                      SpotsDetector())

    self._config._testing = True
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def customTearDown(self) -> None:
    """Used for ensuring at least one spot is defined, so that the interface
    can exit even in case of a bug."""

    self._config._detector.spots.spot_1 = Box(0, 100, 0, 100)

  def test_video_extenso(self) -> None:
    """Tests whether the spots are correctly detected in different
    scenarios."""

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Get the width of the canvas
    width = self._config._img_canvas.winfo_width()
    height = self._config._img_canvas.winfo_height()

    # Get the ratio of the canvas and the image
    can_ratio = width / height
    img_ratio = 320 / 240

    # Determine the position of the image on the canvas from the ratios
    if can_ratio > img_ratio:
      x0 = int(0.5 * height * (can_ratio - img_ratio))
      y0 = 0
      width_eff =  width - 2 * x0
      height_eff = height
    else:
      x0 = 0
      y0 = int(0.5 * width * (1 / can_ratio - 1 / img_ratio))
      width_eff = width
      height_eff = height - 2 * y0

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now",
        x=int(x0 - 0.02 * width_eff), y=int(y0 - 0.02 * height_eff))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now",
        x=int(x0 + 0.08 * width_eff), y=int(y0 + 0.08 * height_eff))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now",
        x=int(x0 + 0.1 * width_eff), y=int(x0 + 0.1 * height_eff))
    self._config._upd_sched()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Move the mouse iteratively in case a border is hit
    for i in range(10, 90, 10):
      self._config._img_canvas.event_generate(
          '<B1-Motion>', when="now",
          x=int(x0 + i * width_eff / 100), y=int(y0 + i * height_eff / 100))
      self._config._upd_sched()

    # Release the mouse button to complete the box
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now",
        x=int(x0 + 0.9 * height_eff), y=int(y0 + 0.9 * height_eff))
    self._config._upd_sched()

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
