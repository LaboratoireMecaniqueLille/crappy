# coding: utf-8

from copy import deepcopy
from time import sleep
from platform import system

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)
from crappy.tool.camera_config.dic_ve_config import DICVEConfig
from crappy.tool.camera_config import SpotsBoxes, Box


class TestDICVE(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)
    self._exit = False

  def customSetUp(self) -> None:
    """"""

    self._config = DICVEConfig(self._camera, self._log_queue,
                               self._log_level, self._freq, SpotsBoxes())

    self._config._testing = True
    self._config._patch_size.value = 20
    self._config.start()

    # Allow some time for the HistogramProcess to start on Windows
    if system() == 'Windows':
      sleep(3)

  def customTearDown(self) -> None:
    """"""

    self._config._spots.spot_1 = Box(0, 100, 0, 100)

  def test_dicve(self) -> None:
    """"""

    # Sleeping to avoid zero division error on Windows
    sleep(0.05)
    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # There should be no points for now
    self.assertTrue(self._config._spots.empty())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config._upd_sched()

    # The patch should be empty
    self.assertTrue(self._config._spots.empty())

    # Start drawing a box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config._upd_sched()

    # The patch should still be empty
    self.assertTrue(self._config._spots.empty())

    # Move the mouse with the button pressed to make a small selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=40, y=40)
    self._config._upd_sched()

    # The patch should still be empty
    self.assertTrue(self._config._spots.empty())

    # Move the mouse iteratively in case a border is hit
    for i in range(40, 200, 20):
      self._config._img_canvas.event_generate(
          '<B1-Motion>', when="now", x=i, y=i)
      self._config._upd_sched()

    # Move the mouse with the button pressed to make a large selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=200, y=200)
    self._config._upd_sched()

    # The spots should have been populated now
    self.assertFalse(self._config._spots.empty())
    self.assertIsInstance(self._config._spots.spot_1, Box)
    self.assertIsInstance(self._config._spots.spot_2, Box)
    self.assertIsInstance(self._config._spots.spot_3, Box)
    self.assertIsInstance(self._config._spots.spot_4, Box)

    # The initial lengths should still be unset
    self.assertIsNone(self._config._spots.x_l0)
    self.assertIsNone(self._config._spots.y_l0)

    spots = deepcopy(self._config._spots)

    # Reset the box
    self._config._spots.reset()
    self._config._upd_sched()

    # The box should now have been reset
    self.assertTrue(self._config._spots.empty())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._spots = spots

    # Delete the configuration window
    self._config.finish()

    # Check that the initial lengths have been set
    self.assertIsNotNone(self._config._spots.x_l0)
    self.assertIsNotNone(self._config._spots.y_l0)
