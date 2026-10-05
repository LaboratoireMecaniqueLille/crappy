# coding: utf-8

"""
This example demonstrates how to customize a GUI backend without changing the
shared Camera configuration behavior. It does not require hardware, but
requires PyQt6, Pillow, matplotlib, and Tk support in the Python installation.

The custom PyQt6 window has a laboratory-specific title and an instruction
banner above the settings panel. It also adds a Ctrl+Return shortcut to the
existing Apply Settings button. Camera settings, preview acquisition, and
acceptance rules remain those of the parent configuration class.

After starting this script, change a FakeCamera setting and press Ctrl+Return
to apply it. Close the configuration window when done. The acquired images
are then displayed for ten seconds before the test stops automatically. No
images are saved.

Change config_backend below to 'tkinter' to compare with the unmodified
Tkinter window. PyQt6 remains required because this script imports its widgets
explicitly.
"""

import crappy
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QLabel
from crappy.tool.camera_config.pyqt import PyQtCameraConfig
from crappy.tool.camera_config.tkinter import TkinterCameraConfig


class LaboratoryPyQtConfig(PyQtCameraConfig):
  """Customize the PyQt6 interface without changing configuration logic."""

  def _set_layout(self) -> None:
    """Build the standard interface, then add the banner and shortcut."""

    # Let the backend create its controls before accessing any Qt widgets
    super()._set_layout()
    self.setWindowTitle('Laboratory camera setup')

    # Use the existing button so keyboard and mouse follow the same Apply path
    self._apply_button.setShortcut(QKeySequence('Ctrl+Return'))
    self._apply_button.setToolTip('Apply pending settings with Ctrl+Return')

    # Insert the banner in the button's side panel, not over the preview image
    banner = QLabel('Laboratory camera setup\n'
                    'Ctrl+Return applies pending settings.')
    banner.setWordWrap(True)
    font = banner.font()
    font.setBold(True)
    banner.setFont(font)
    panel_layout = self._apply_button.parentWidget().layout()
    panel_layout.insertWidget(0, banner)


class LaboratoryCameraSource(crappy.blocks.vision.CameraSource):
  """Replace only the PyQt6 window used by this CameraSource subclass."""

  configurator = {'tkinter': TkinterCameraConfig,
                  'pyqt': LaboratoryPyQtConfig}


if __name__ == '__main__':

  # The backend-specific custom window is created during preparation, not here
  # Change config_backend to 'tkinter' to keep the original interface instead
  camera = LaboratoryCameraSource(
      'FakeCamera',
      config_backend='pyqt',
      width=320,
      height=240,
      speed=40,
      fps=15,
      freq=30,
  )

  # This separate display window is unchanged by the configurator subclass
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='Laboratory camera images',
      backend='mpl',
      framerate=15,
      freq=30,
  )

  # Stop automatically ten seconds after the configuration has been accepted
  stop = crappy.blocks.StopBlock('t(s) > 10')

  crappy.img_link(camera, displayer)
  crappy.link(camera, stop)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
