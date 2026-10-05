# coding: utf-8

"""Tk preview indicators and acquisition scheduling."""

from itertools import product
from math import log2, ceil
from platform import system
from unittest.mock import patch

from ._fixtures import TkinterConfigTestCase, FakeTestCameraSimple
import crappy.tool.camera_config.tkinter.camera_config as camera_config_module


class TestFPS(TkinterConfigTestCase):
  """Class for testing if the FPS are correctly handled in the configuration
  window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple()

  def test_fps(self) -> None:
    """Tests whether the FPS are correctly calculated, displayed, and if the
    maximum FPs value is enforced."""

    # FPS-related variables should be initialized to their default values
    self.assertEqual(self._config._display_state.fps, 0.)
    self.assertEqual(self._config._fps_txt.get(),
                     f'fps = 0.00\n(might be lower in this GUI than actual)')

    current_time = [0.0]
    self._config._last_upd_t = current_time[0]

    def fake_time() -> float:
      return current_time[0]

    def acquire_image() -> None:
      self._config._n_loops += 1

    with patch.object(camera_config_module, 'time', side_effect=fake_time), \
         patch.object(camera_config_module, 'monotonic',
                      side_effect=fake_time), \
         patch.object(self._config, '_update_img', side_effect=acquire_image):
      # Ten evenly-spaced frames are enough to verify each frequency exactly;
      # there is no need to wait through the equivalent wall-clock duration.
      for fps in (1, 2, 3, 4, 5, 10, 15, 20):
        with self.subTest(fps=fps):
          self._config._max_freq = fps
          self._config._next_acq_t = -float('inf')

          for _ in range(10):
            current_time[0] += 1 / fps
            self._config._img_acq_sched()
          self._config._upd_var_sched()

          self.assertAlmostEqual(self._config._display_state.fps, fps)
          self.assertEqual(
              self._config._fps_txt.get(),
              f'fps = {self._config._display_state.fps:.2f}\n(might be lower in '
              f'this GUI than actual)')

      # Free-looping should not impose the configured 20 FPS ceiling.
      self._config._max_freq = None
      for _ in range(25):
        current_time[0] += 1 / 25
        self._config._img_acq_sched()
      self._config._upd_var_sched()

    self.assertAlmostEqual(self._config._display_state.fps, 25.0)

  def test_acquisition_schedules_deadline_instead_of_polling(self) -> None:
    """The frame timer sleeps until the next limited acquisition is due."""

    self._config._max_freq = 20
    self._config._testing = False

    with (patch.object(camera_config_module, 'monotonic', return_value=100.0),
          patch.object(self._config, '_update_img') as acquire,
          patch.object(self._config, '_sync_setting_controls') as sync,
          patch.object(self._config, 'after', return_value='scheduled') as after):
      self._config._img_acq_sched()
      self._config._img_acq_sched()

    acquire.assert_called_once_with()
    sync.assert_called_once_with()
    self.assertEqual(self._config._next_acq_t, 100.05)
    self.assertEqual(after.call_count, 2)
    self.assertEqual([call.args[0] for call in after.call_args_list], [50, 50])
    self._config._testing = True


class TestIndicators(TkinterConfigTestCase):
  """Class for testing the display of the status indicators in the
  configuration window.

  .. versionadded:: 2.0.8
  """

  def make_camera(self) -> FakeTestCameraSimple:
    """Create the deterministic Camera used by this test."""

    return FakeTestCameraSimple(min_val=0, max_val=255)

  def test_indicators(self) -> None:
    """Tests whether the status indicators are correctly determined and
    displayed in the interface."""

    # Monitoring variables should be initialized to their default values
    self.assertEqual(self._config._display_state.detected_bits, 0)
    self.assertEqual(self._config._display_state.max_pixel, 0)
    self.assertEqual(self._config._display_state.min_pixel, 0)
    self.assertEqual(self._config._display_state.reticle_value, 0)
    self.assertEqual(self._config._display_state.reticle_x, 0)
    self.assertEqual(self._config._display_state.reticle_y, 0)
    self.assertEqual(self._config._display_state.zoom_percent, 100.0)

    # Displayed texts should be initialized to their default values
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 0')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 0, max: 0')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 0')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    self.run_config_cycle()

    # The monitoring variables should change because of the acquired image
    self.assertEqual(self._config._display_state.detected_bits, 8)
    self.assertEqual(self._config._display_state.min_pixel, 0)
    self.assertEqual(self._config._display_state.max_pixel, 255)
    self.assertEqual(self._config._display_state.reticle_value, 0)
    self.assertEqual(self._config._display_state.reticle_x, 0)
    self.assertEqual(self._config._display_state.reticle_y, 0)
    self.assertEqual(self._config._display_state.zoom_percent, 100.0)

    # These displayed texts should change because of the acquired image
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 0, max: 255')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 0')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # Exercise representative ranges, especially around bit-depth boundaries.
    ranges = ((0, 1), (0, 2), (1, 15), (3, 16), (20, 31),
              (20, 32), (100, 127), (100, 128), (240, 255))
    for min_, max_ in ranges:
      with self.subTest(min=min_, max=max_):
        # Updating the min and max values of the camera object
        self._camera._min = min_
        self._camera._max = max_

        self.run_config_cycle()

        # Checking that the indicators have the right values
        self.assertEqual(self._config._display_state.detected_bits,
                         ceil(log2(max_ + 1)))
        self.assertEqual(self._config._display_state.min_pixel, min_)
        self.assertEqual(self._config._display_state.max_pixel, max_)
        self.assertEqual(self._config._display_state.reticle_value, min_)

        # Checking that the correct text is displayed
        self.assertEqual(self._config._bits_txt.get(),
                         f'Detected bits: {ceil(log2(max_ + 1))}')
        self.assertEqual(self._config._min_max_pix_txt.get(),
                         f'min: {min_}, max: {max_}')
        self.assertEqual(self._config._reticle_txt.get(),
                         f'X: 0, Y: 0, V: {min_}')

    # Reset the min and max image values
    self._camera._min = 0
    self._camera._max = 255

    # Checking if the zoom level is updated correctly when zooming in at the
    # center of the image
    for i in range(self._config._max_zoom_step):
      with self.subTest(zoom_step=i):
        if system() == "Linux":
          self._config._img_canvas.event_generate(
              '<4>', when="now",
              x=self._config._img_canvas.winfo_width() // 2,
              y=self._config._img_canvas.winfo_height() // 2)
        else:
          self._config._img_canvas.event_generate(
              '<MouseWheel>', when="now",
              x=self._config._img_canvas.winfo_width() // 2,
              y=self._config._img_canvas.winfo_height() // 2, delta=1)

        self.assertEqual(self._config._display_state.zoom_percent,
                         100 * (1 / self._config._zoom_ratio) **
                         self._config._zoom_step)
        self.assertEqual(
          self._config._zoom_txt.get(),
          f'Zoom: {self._config._display_state.zoom_percent:.1f}%')

    # Checking if the zoom level is updated correctly when zooming out at the
    # center of the image
    for i in range(self._config._max_zoom_step):
      with self.subTest(zoom_step=self._config._max_zoom_step - i):
        if system() == "Linux":
          self._config._img_canvas.event_generate(
              '<5>', when="now",
              x=self._config._img_canvas.winfo_width() // 2,
              y=self._config._img_canvas.winfo_height() // 2)
        else:
          self._config._img_canvas.event_generate(
              '<MouseWheel>', when="now",
              x=self._config._img_canvas.winfo_width() // 2,
              y=self._config._img_canvas.winfo_height() // 2, delta=-1)

        self.assertEqual(self._config._display_state.zoom_percent,
                         100 * (1 / self._config._zoom_ratio) **
                         self._config._zoom_step)
        self.assertEqual(
          self._config._zoom_txt.get(),
          f'Zoom: {self._config._display_state.zoom_percent:.1f}%')

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
      width_eff = width - 2 * x0
      height_eff = height
    else:
      x0 = 0
      y0 = int(0.5 * width * (1 / can_ratio - 1 / img_ratio))
      width_eff = width
      height_eff = height - 2 * y0

    # Checking if the position and reticle values are updated correctly when
    # moving the mouse around
    x_positions = tuple(x0 + round(fraction * width_eff)
                        for fraction in (0.05, 0.25, 0.5, 0.75, 0.95))
    y_positions = tuple(y0 + round(fraction * height_eff)
                        for fraction in (0.05, 0.25, 0.5, 0.75, 0.95))
    for x, y in product(x_positions, y_positions):
      with self.subTest(x=x, y=y):

        # Moving the mouse over the displayed image
        self._config._img_canvas.event_generate('<Motion>', when="now",
                                                x=x, y=y)

        self.run_config_cycle()

        # Checking that the indicators are correctly updated
        reticle = int(((x - x0) + (y - y0)) / (width_eff + height_eff) * 255)
        x_pos = int((x - x0) / width_eff * 320)
        y_pos = int((y - y0) / height_eff * 240)
        self.assertAlmostEqual(self._config._display_state.reticle_value,
                               reticle, delta=2)
        self.assertAlmostEqual(self._config._display_state.reticle_x,
                               x_pos, delta=1)
        self.assertAlmostEqual(self._config._display_state.reticle_y,
                               y_pos, delta=1)
        self.assertEqual(self._config._reticle_txt.get(),
                         f'X: {self._config._display_state.reticle_x}, '
                         f'Y: {self._config._display_state.reticle_y}, '
                         f'V: {self._config._display_state.reticle_value}')
