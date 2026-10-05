# coding: utf-8

"""
This example demonstrates how to customize the shared Camera configuration
behavior without writing GUI code. It does not require hardware, but requires
Pillow, PyQt6, and matplotlib. The default configuration window uses PyQt6.
To try Tkinter instead, pass 'tkinter' as config_backend and ensure Tk support
is available in the Python installation.

The custom configuration adds a Confirm setup checkbox and refuses to close
until that setting has been applied. The checkbox belongs to the configuration
window, not to the Camera, so the FakeCamera driver does not need any changes.
The same confirmation rule is combined with both GUI backends, and a custom
CameraSource Block selects the appropriate class through its configurator
attribute.

After starting this script, try closing the window without confirming the
setup. Then check Confirm setup, click Apply Settings, and close the window
again. The terminal reports Camera setup confirmed, and the acquired images
are displayed for ten seconds before the test stops automatically. The
configuration window itself has no timeout, and no images are saved.
"""

from abc import ABC
import logging
import crappy
from crappy.camera.meta_camera.camera_setting import (
    CameraBoolSetting, CameraSetting)
from crappy.tool.camera_config.base import CameraConfig
from crappy.tool.camera_config.pyqt import PyQtCameraConfig
from crappy.tool.camera_config.tkinter import TkinterCameraConfig


class ConfirmedConfig(CameraConfig, ABC):
  """Add the same confirmation setting and acceptance rule to both backends."""

  def _create_local_settings(self) -> tuple[CameraSetting, ...]:
    """Create the checkbox without adding a setting to the Camera driver."""

    # Keep the parent's settings when extending another configuration class
    self._confirmation = CameraBoolSetting('Confirm setup', default=False)
    return *super()._create_local_settings(), self._confirmation

  def _validate_close(self) -> str | None:
    """Keep the window open until the confirmation has been applied."""

    # Preserve any acceptance requirements defined by the parent class
    reason = super()._validate_close()
    if reason is not None:
      return reason

    # Read the applied setting, not a backend-specific widget's pending value
    if not self._confirmation.value:
      return 'Check Confirm setup and apply the settings before closing.'
    return None

  def _on_valid_close(self) -> None:
    """Report successful confirmation before the window releases resources."""

    super()._on_valid_close()
    self.log(logging.INFO, 'Camera setup confirmed')


# Put the shared extension first so its hooks take precedence over the backend
class TkinterConfirmedConfig(ConfirmedConfig, TkinterCameraConfig):
  """Combine the shared rule with the existing Tkinter interface."""


class PyQtConfirmedConfig(ConfirmedConfig, PyQtCameraConfig):
  """Combine the shared rule with the existing PyQt6 interface."""


class ConfirmedCameraSource(crappy.blocks.vision.CameraSource):
  """Select the custom windows without changing the CameraSource arguments."""

  # Select explicit classes on the Block rather than creating GUI instances
  configurator = {'tkinter': TkinterConfirmedConfig,
                  'pyqt': PyQtConfirmedConfig}


if __name__ == '__main__':

  # This custom CameraSource opens one of the windows defined above during
  # preparation using the default PyQt6 backend.
  camera = ConfirmedCameraSource(
      'FakeCamera',
      width=320,
      height=240,
      speed=40,
      fps=15,
      freq=30,
  )

  # Image display is independent of the configuration window's GUI backend
  displayer = crappy.blocks.vision.ImageDisplayer(
      title='Confirmed camera setup',
      backend='mpl',
      framerate=15,
      freq=30,
  )

  # This timeout starts after configuration, when the experiment begins
  stop = crappy.blocks.StopBlock('t(s) > 10')

  # Images use an ImageLink; the stop condition only needs regular timestamps
  crappy.img_link(camera, displayer)
  crappy.link(camera, stop)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
