# coding: utf-8

from time import sleep
from re import fullmatch

from .camera_configuration_test_base import (ConfigurationWindowTestBase,
                                             FakeTestCameraSimple)


class TestResize(ConfigurationWindowTestBase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, camera=FakeTestCameraSimple(), **kwargs)

  def test_resize(self) -> None:
    """"""

    # Calling the first loop
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # Leave some time to the histogram calculation
    sleep(1)

    # Call a second loop to display the histogram
    self._config._img_acq_sched()
    self._config._upd_var_sched()
    self._config._upd_sched()

    # Read the current sizes of the graphical objects
    hist_canvas_width = self._config._hist_canvas.winfo_width()
    hist_canvas_height = self._config._hist_canvas.winfo_height()
    hist_size_w, hist_size_h = self._config._pil_hist.size
    img_canvas_width = self._config._img_canvas.winfo_width()
    img_canvas_height = self._config._img_canvas.winfo_height()
    img_size_w, img_size_h = self._config._pil_img.size

    # Read the current geometry and reduce it
    w, h, x, y = map(int, fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)',
                                    self._config.winfo_geometry()).groups())
    self._config.geometry(f"{int(0.6 * w)}x{int(0.8 * h)}+{x}+{y}")

    # Call new loops to apply the changes
    for _ in range(2):
      # Sleeping to avoid zero division error on Windows
      sleep(0.05)
      self._config._img_acq_sched()
      self._config._upd_var_sched()
      self._config._upd_sched()
      sleep(1)

    # All dimensions should be smaller
    self.assertGreater(hist_canvas_width,
                       self._config._hist_canvas.winfo_width())
    self.assertGreater(hist_canvas_height,
                       self._config._hist_canvas.winfo_height())
    self.assertGreater(hist_size_w, self._config._pil_hist.size[0])
    self.assertGreater(hist_size_h, self._config._pil_hist.size[1])
    self.assertGreater(img_canvas_width,
                       self._config._img_canvas.winfo_width())
    self.assertGreater(img_canvas_height,
                       self._config._img_canvas.winfo_height())
    self.assertGreater(img_size_w, self._config._pil_img.size[0])
    self.assertGreater(img_size_h, self._config._pil_img.size[1])

    # Read the current geometry and extend it
    w, h, x, y = map(int, fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)',
                                    self._config.winfo_geometry()).groups())
    self._config.geometry(f"{int(2.5 * w)}x{int(1.5 * h)}+{x}+{y}")

    # Call new loops to apply the changes
    for _ in range(2):
      # Sleeping to avoid zero division error on Windows
      sleep(0.05)
      self._config._img_acq_sched()
      self._config._upd_var_sched()
      self._config._upd_sched()
      sleep(1)

    # All dimensions should be greater
    self.assertGreater(self._config._hist_canvas.winfo_width(),
                       hist_canvas_width)
    self.assertGreater(self._config._hist_canvas.winfo_height(),
                       hist_canvas_height)
    self.assertGreater(self._config._pil_hist.size[0], hist_size_w)
    self.assertGreater(self._config._pil_hist.size[1], hist_size_h)
    self.assertGreater(self._config._img_canvas.winfo_width(),
                       img_canvas_width)
    self.assertGreater(self._config._img_canvas.winfo_height(),
                       img_canvas_height)
    self.assertGreater(self._config._pil_img.size[0], img_size_w)
    self.assertGreater(self._config._pil_img.size[1], img_size_h)
