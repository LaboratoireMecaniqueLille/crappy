# coding: utf-8

"""Tk box gestures, local settings, and specialized configuration handoff."""

from copy import deepcopy
from importlib.util import find_spec
import unittest
from unittest.mock import patch

from ._fixtures import (TkinterConfigTestCase, FakeTestCameraSimple,
                        FakeTestCameraSpots)
from crappy.tool.camera_config.tkinter import (TkinterCameraConfigBoxes,
                                                TkinterDICVEConfig,
                                                TkinterDISCorrelConfig,
                                                TkinterVideoExtensoConfig)
from crappy.tool.camera_config import Box, SpotsBoxes


class TestDrawBox(TkinterConfigTestCase):
  """Class for testing the
  :class:`~crappy.tool.camera_config_boxes.CameraConfigBoxes` class.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface and for
    adding bindings, otherwise the test wouldn't work."""

    self._config = TkinterCameraConfigBoxes(self._camera, self._log_queue,
                                            self._log_level, self._freq, None)

    self._config._img_canvas.bind('<ButtonPress-1>', self._config._start_box)
    self._config._img_canvas.bind('<B1-Motion>', self._config._extend_box)

    self._config._testing = True
    self.start_configuration()

  def test_draw_box(self) -> None:
    """Tests whether the selection box is correctly displayed on the image when
    drawing it."""

    self.run_config_cycle()

    # There should be no points for now
    self.assertTrue(self._config._select_box.no_points())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config.update_idletasks()

    # The start point should not have been counted
    self.assertTrue(self._config._select_box.no_points())

    # Start drawing a box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

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
    self._config.update_idletasks()

    # The end point should now be defined and the box is complete
    self.assertFalse(self._config._select_box.no_points())
    self.assertIsNotNone(self._config._select_box.x_start)
    self.assertIsNotNone(self._config._select_box.y_start)
    self.assertIsNotNone(self._config._select_box.x_end)
    self.assertIsNotNone(self._config._select_box.y_end)

    # Reset the box
    self._config._select_box.reset()
    self._config.update_idletasks()

    # The box should now have been reset
    self.assertTrue(self._config._select_box.no_points())


class TestDICVE(TkinterConfigTestCase):
  """Class for testing the :class:`~crappy.tool.dic_ve_config.DICVEConfig`
  class.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface and for
    setting a smaller patch size value for the tests."""

    self._config = TkinterDICVEConfig(self._camera, self._log_queue,
                                      self._log_level, self._freq, None,
                                      SpotsBoxes())

    self._config._testing = True
    self._config._patch_size.value = 20
    self._config._sync_setting_controls()
    self.start_configuration()


  def test_patch_size_uses_common_apply_path(self) -> None:
    """A local setting is applied before camera settings by the manager."""

    patch_size = self._config._patch_size
    self.assertIs(self._config._setting_manager.settings[0], patch_size)
    observed_patch_sizes = []
    self._camera.add_bool_setting(
      'capture_patch_size', setter=lambda _: observed_patch_sizes.append(
        patch_size.value))
    camera_setting = self._camera.settings['capture_patch_size']
    self._config._add_bool_setting(camera_setting)
    self._config._setting_controls[patch_size].widget.set(30)
    self._config._setting_controls[camera_setting].widget.invoke()

    self._config._update_button.invoke()

    self.assertEqual(patch_size.value, 30)
    self.assertEqual(observed_patch_sizes, [30])

  def test_dicve(self) -> None:
    """Tests whether the patches are correctly defined in several scenarios."""

    self.run_config_cycle()

    # There should be no points for now
    self.assertTrue(self._config._spots.empty())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config.update_idletasks()

    # The patch should be empty
    self.assertTrue(self._config._spots.empty())

    # Start drawing a box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

    # The patch should still be empty
    self.assertTrue(self._config._spots.empty())

    # Move the mouse with the button pressed to make a small selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=40, y=40)
    self._config.update_idletasks()

    # The patch should still be empty
    self.assertTrue(self._config._spots.empty())

    # Move the mouse iteratively in case a border is hit
    for i in range(40, 200, 20):
      self._config._img_canvas.event_generate(
          '<B1-Motion>', when="now", x=i, y=i)
      self._config.update_idletasks()

    # Move the mouse with the button pressed to make a large selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=200, y=200)
    self._config.update_idletasks()

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
    self._config.update_idletasks()

    # The box should now have been reset
    self.assertTrue(self._config._spots.empty())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._spots = spots

    # Delete the configuration window
    self._config.finish()

    # Check that the initial lengths have been set
    self.assertIsNotNone(self._config._spots.x_l0)
    self.assertIsNotNone(self._config._spots.y_l0)

    configured_spots, = self._config.get_config()
    self.assertIs(configured_spots, self._config._spots)


class TestDISCorrel(TkinterConfigTestCase):
  """Class for testing the
  :class:`~crappy.tool.dis_correl_config.DISCorrelConfig` class.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface."""

    self._config = TkinterDISCorrelConfig(self._camera, self._log_queue,
                                          self._log_level, self._freq, None,
                                          Box())

    self._config._testing = True
    self.start_configuration()


  def test_discorrel(self) -> None:
    """Tests whether the patch is correctly defined in several scenarios."""

    self.run_config_cycle()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Start drawing a box outside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=-20, y=-20)
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=50, y=50)
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config.box.no_points())

    # Release the mouse button to complete the box
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=50, y=50)
    self._config.update_idletasks()

    # The end point should now be defined and the box is complete
    self.assertFalse(self._config.box.no_points())
    self.assertIsNotNone(self._config.box.x_start)
    self.assertIsNotNone(self._config.box.y_start)
    self.assertIsNotNone(self._config.box.x_end)
    self.assertIsNotNone(self._config.box.y_end)

    box = deepcopy(self._config.box)

    # Reset the box
    self._config.box.reset()
    self._config.update_idletasks()

    # The box should now have been reset
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

    # Release the mouse button at the same location
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=20, y=20)
    self._config.update_idletasks()
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

    # The box should be empty
    self.assertTrue(self._config.box.no_points())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now", x=20, y=20)
    self._config.update_idletasks()

    # Draw a box with no pixels inside
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now", x=20, y=50)
    self._config.update_idletasks()
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now", x=20, y=50)
    self._config.update_idletasks()

    # The box should be empty
    self.assertTrue(self._config.box.no_points())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._correl_box = box

    configured_box, = self._config.get_config()
    self.assertIs(configured_box, self._config._correl_box)


@unittest.skipUnless(
    find_spec('cv2') is not None and find_spec('skimage') is not None,
    "opencv-python and scikit-image are required for video extensometry tests")
class TestVideoExtenso(TkinterConfigTestCase):
  """Class for testing the
  :class:`~crappy.tool.video_extenso_config.VideoExtensoConfig` class.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSpots:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSpots()

  def customSetUp(self) -> None:
    """Used for instantiating the special configuration interface."""

    self._config = TkinterVideoExtensoConfig(self._camera, self._log_queue,
                                             self._log_level, self._freq,
                                             None,
                                             white_spots=False,
                                             num_spots=None,
                                             min_area=150,
                                             blur=5,
                                             update_thresh=False,
                                             safe_mode=False,
                                             border=5)

    self._config._testing = True
    self.start_configuration()


  def test_actions_and_invalid_close(self) -> None:
    """Save L0 stays available when auto-apply disables only Apply."""

    self.assertIs(self._config._update_button, self._config._apply_button)
    save_button = self._config._action_buttons['save_l0']
    self.assertIsNot(save_button, self._config._apply_button)
    self._config._auto_apply_var.set(True)
    self._config._on_auto_apply_toggle()
    self.assertEqual(self._config._apply_button['state'], 'disabled')
    self.assertEqual(save_button['state'], 'normal')

    with patch('crappy.tool.camera_config.tkinter.camera_config.showerror') as dialog:
      self._config.finish()
    dialog.assert_called_once()
    self.assertTrue(self._config.winfo_exists())
    self.assertFalse(self._config._stop_event.is_set())
    self.assertIsNone(self._config._detector.spots.x_l0)

  def test_video_extenso(self) -> None:
    """Tests whether the spots are correctly detected in different
    scenarios."""

    self.run_config_cycle()

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
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Start drawing the selection box inside the image
    self._config._img_canvas.event_generate(
        '<ButtonPress-1>', when="now",
        x=int(x0 + 0.08 * width_eff), y=int(y0 + 0.08 * height_eff))
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Move the mouse with the button pressed to complete the selection box
    self._config._img_canvas.event_generate(
        '<B1-Motion>', when="now",
        x=int(x0 + 0.1 * width_eff), y=int(x0 + 0.1 * height_eff))
    self._config.update_idletasks()

    # The box should not be set for now
    self.assertTrue(self._config._detector.spots.empty())

    # Move the mouse iteratively in case a border is hit
    for i in range(10, 90, 10):
      self._config._img_canvas.event_generate(
          '<B1-Motion>', when="now",
          x=int(x0 + i * width_eff / 100), y=int(y0 + i * height_eff / 100))
      self._config.update_idletasks()

    # Release the mouse button to complete the box
    self._config._img_canvas.event_generate(
        '<ButtonRelease-1>', when="now",
        x=int(x0 + 0.9 * height_eff), y=int(y0 + 0.9 * height_eff))
    self._config.update_idletasks()

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
    self._config._action_buttons['save_l0'].invoke()

    # Check that the initial lengths have been set
    self.assertIsNotNone(self._config._spots.x_l0)
    self.assertIsNotNone(self._config._spots.y_l0)

    spots = deepcopy(self._config._detector.spots)

    # Reset the box
    self._config._detector.spots.reset()
    self._config.update_idletasks()

    # The box should now have been reset
    self.assertTrue(self._config._detector.spots.empty())

    # Re-populate the spots to avoid the interface crashing at exit
    self._config._detector.spots = spots

    configured_spots, threshold = self._config.get_config()
    self.assertIs(configured_spots, self._config._detector.spots)
    self.assertEqual(threshold, self._config._detector.thresh)
