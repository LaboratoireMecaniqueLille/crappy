# coding: utf-8

"""
This example demonstrates how to add a trigger setting to a custom Camera
object in Crappy. It builds on custom_camera_basic.py, which should be read
first. It does not require any hardware, but requires the Pillow and
opencv-python Python modules.

Crappy makes it easy to implement a trigger setting in Camera objects. In the
configuration window, the user can choose free-run mode, hardware-trigger
mode, or hardware-trigger mode after configuration. The last option lets the
user configure the camera in free-run mode and switch to hardware-trigger mode
for the test. Adding a trigger setting requires only one method call, as
demonstrated in this example.

Here, a very simple Camera object is instantiated and driven by a Camera Block
that displays the acquired images. The Camera object features a trigger
setting that lets the user select the trigger mode in the configuration
window. Because there's no actual hardware involved, the hardware trigger mode
is emulated by only allowing one image acquisition per second. The goal is
to show how to implement a trigger setting in Camera objects.

After starting this script, a configuration window appears in which you can see
the generated images. You can only tune the trigger mode setting. Select one
and close the configuration window to start the test and observe its effect.
In Free run mode, the image rate should be close to 30 FPS in both the
configuration and displayer windows because no trigger is applied. In Hardware
trigger mode, both windows update only once per second. In Hdw after config
mode, the configuration window runs normally at about 30 FPS, but the displayer
updates at about 1 Hz. Free run remains active until the configuration window
closes, after which the Camera switches to Hardware trigger mode. To end this
demo, click the stop button.
"""

import crappy
import numpy as np
import numpy.random as rd
from time import time, sleep


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy with a trigger setting.

  In the open method, add_trigger_setting adds a trigger setting to the Camera
  object. By tuning this setting in the configuration window, the user can
  choose to let the camera run in free run mode, to switch it to hardware
  trigger mode after closing the configuration window, or to directly switch it
  to hardware trigger mode.

  This class is based on the one defined in custom_camera_basic.py. Refer to
  that example for more information.
  """

  def __init__(self) -> None:
    """Almost the same as in custom_camera_basic.py.

    Here, a buffer and a flag are defined to emulate a hardware trigger
    setting.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    self._trigger_mode: str = 'Free run'
    self._run: bool = True

  def open(self, **kwargs) -> None:
    """Compared to the custom_camera_basic.py example, we define here a
    trigger setting using the add_trigger_setting method.

    Unlike the settings defined in the custom_camera_settings.py example, the
    trigger setting is handled internally and in a standardized way.

    The camera can run in free-run mode, hardware-trigger mode, or switch to
    hardware-trigger mode after the configuration window closes. The last
    option makes it easier to adjust settings when the trigger frequency is low
    or when Crappy generates the trigger only after configuration.

    Since this example is designed to run without hardware, the hardware
    trigger is replaced by a delay of 1 s to simulate a hardware-trigger signal
    running at 1 Hz.
    """

    # Adding a trigger setting, which Crappy handles differently from other
    # types of settings
    self.add_trigger_setting(getter=self._get_trigger_mode,
                             setter=self._set_trigger_mode)

    # This line is mandatory to apply the initial trigger setting
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, np.ndarray]:
    """Compared with custom_camera_basic.py, this method returns the
    same images but with a delay if in Hardware trigger mode.

    Since this example runs without any hardware, it emulates the hardware
    trigger by only allowing one image per second in Hardware trigger mode.
    """

    # Outside free-run mode, assume a hardware trigger is issued about once per
    # second
    if not self._run:
      sleep(1)

    return time(), rd.randint(low=0, high=256, size=(480, 640), dtype='uint8')

  def close(self) -> None:
    """Same as in custom_camera_basic.py, nothing to do here."""

    pass

  def _set_trigger_mode(self, mode: str) -> None:
    """This method sets the current trigger mode.

    Normally it would set this parameter directly on hardware, but this demo
    was designed to run completely virtually.
    """

    # Setting the flag telling whether the camera should return images
    if mode in ('Free run', 'Hdw after config'):
      self._run = True
    else:
      self._run = False

    # Storing the selected mode
    self._trigger_mode = mode

  def _get_trigger_mode(self) -> str:
    """This method returns the current trigger mode.

    Normally it would read this parameter directly from hardware, but this demo
    was designed to run completely virtually.
    """

    return self._trigger_mode


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It acquires images and displays them in a dedicated Displayer window. The
  # user can choose in which trigger mode the Camera runs.
  cam = crappy.blocks.Camera(
      'CustomCam',  # The name of the custom Camera that was just written
      config=True,  # Easier to set to True when possible
      display_images=True,  # Displaying the images to show how they look
      displayer_framerate=30,  # Matching the acquisition frame rate
      freq=30,  # Lowering the frequency because it's just a demo
      save_images=False,  # No need to record images in this example

      # Sticking to default for the other arguments
  )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
