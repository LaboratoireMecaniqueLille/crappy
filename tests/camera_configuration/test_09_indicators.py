# coding: utf-8

from itertools import product
from math import log2, ceil
from platform import system
from time import sleep

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestIndicators(ConfigurationWindowTestBase):
  """Class for testing the display of the status indicators in the 
  configuration window.

  .. versionadded:: 2.0.8
  """

  def __init__(self, *args, **kwargs) -> None:
    """Used to instantiate a Camera that actually generates images."""

    super().__init__(*args,
                     camera=FakeTestCameraSimple(min_val=0, max_val=255),
                     **kwargs)

  def test_indicators(self) -> None:
    """Tests whether the status indicators are correctly determined and
    displayed in the interface."""

    # Monitoring variables should be initialized to their default values
    self.assertEqual(self._config._nb_bits.get(), 0)
    self.assertEqual(self._config._max_pixel.get(), 0)
    self.assertEqual(self._config._min_pixel.get(), 0)
    self.assertEqual(self._config._reticle_val.get(), 0)
    self.assertEqual(self._config._x_pos.get(), 0)
    self.assertEqual(self._config._y_pos.get(), 0)
    self.assertEqual(self._config._zoom_level.get(), 100.0)

    # Displayed texts should be initialized to their default values
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 0')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 0, max: 0')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 0')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # The monitoring variables should change because of the acquired image
    self.assertEqual(self._config._nb_bits.get(), 8)
    self.assertEqual(self._config._min_pixel.get(), 0)
    self.assertEqual(self._config._max_pixel.get(), 255)
    self.assertEqual(self._config._reticle_val.get(), 0)
    self.assertEqual(self._config._x_pos.get(), 0)
    self.assertEqual(self._config._y_pos.get(), 0)
    self.assertEqual(self._config._zoom_level.get(), 100.0)

    # These displayed texts should change because of the acquired image
    self.assertEqual(self._config._bits_txt.get(), 'Detected bits: 8')
    self.assertEqual(self._config._min_max_pix_txt.get(), 'min: 0, max: 255')
    self.assertEqual(self._config._reticle_txt.get(), 'X: 0, Y: 0, V: 0')
    self.assertEqual(self._config._zoom_txt.get(), 'Zoom: 100.0%')

    # Checking if the min, max, and bits indicators are working correctly
    for min_, max_ in product(range(0, 256, 20), repeat=2):
      with self.subTest(min=min_, max=max_):

        # Ensuring that the max is greater than the min
        if min_ >= max_:
          continue

        # Updating the min and max values of the camera object
        self._camera._min = min_
        self._camera._max = max_

        # Sleeping to avoid zero division error on Windows
        sleep(0.05)
        # Looping to update the image
        self._config._img_acq_sched()
        self._config._upd_var_sched()
        self._config._upd_sched()

        # Checking that the indicators have the right values
        self.assertEqual(self._config._nb_bits.get(), ceil(log2(max_ + 1)))
        self.assertEqual(self._config._min_pixel.get(), min_)
        self.assertEqual(self._config._max_pixel.get(), max_)
        self.assertEqual(self._config._reticle_val.get(), min_)

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

        self.assertEqual(self._config._zoom_level.get(),
                         100 * (1 / self._config._zoom_ratio) **
                         self._config._zoom_step)
        self.assertEqual(self._config._zoom_txt.get(),
                         f'Zoom: {self._config._zoom_level.get():.1f}%')

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

        self.assertEqual(self._config._zoom_level.get(),
                         100 * (1 / self._config._zoom_ratio) **
                         self._config._zoom_step)
        self.assertEqual(self._config._zoom_txt.get(),
                         f'Zoom: {self._config._zoom_level.get():.1f}%')

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
    for x, y in product(range(x0 + 1, x0 + width_eff, width_eff // 10),
                        range(y0 + 1, y0 + height_eff, height_eff // 10)):
      with self.subTest(x=x, y=y):

        # Moving the mouse over the displayed image
        self._config._img_canvas.event_generate('<Motion>', when="now",
                                                x=x, y=y)

        # Sleeping to avoid zero division error on Windows
        sleep(0.05)
        # Looping to update the image
        self._config._img_acq_sched()
        self._config._upd_var_sched()
        self._config._upd_sched()

        # Checking that the indicators are correctly updated
        reticle = int(((x - x0) + (y - y0)) / (width_eff + height_eff) * 255)
        x_pos = int((x - x0) / width_eff * 320)
        y_pos = int((y - y0) / height_eff * 240)
        self.assertAlmostEqual(self._config._reticle_val.get(), reticle,
                               delta=2)
        self.assertAlmostEqual(self._config._x_pos.get(), x_pos, delta=1)
        self.assertAlmostEqual(self._config._y_pos.get(), y_pos, delta=1)
        self.assertEqual(self._config._reticle_txt.get(),
                         f'X: {self._config._x_pos.get()}, '
                         f'Y: {self._config._y_pos.get()}, '
                         f'V: {self._config._reticle_val.get()}')
