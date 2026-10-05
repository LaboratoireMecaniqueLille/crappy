# coding: utf-8

"""
This example demonstrates how to add a software ROI setting to a custom Camera
object in Crappy. It builds on custom_camera_basic.py, which should be read
first. It does not require any hardware, but requires the Pillow and
opencv-python Python modules.

Crappy makes it easy to implement a software ROI setting in Camera objects.
Using this setting, the user can select in the configuration window which part
of the image (region of interest) to keep for processing, display, and
recording. The ROI is always rectangular, and the user can tune its x and y
offsets, height, and width. Adding a software ROI setting requires only two
method calls, as demonstrated in this example.

Here, a very simple Camera object is instantiated and driven by a Camera Block
that displays the acquired images. The Camera object features a software ROI
setting that lets the user select the ROI in the configuration window. The
goal here is to show how to implement a software ROI in Camera objects.

After starting this script, a configuration window appears in which you can see
the generated images. Four settings control the software ROI. Change their
values, click the Apply Settings button, and observe how the image changes.
When you're done, close the configuration window and observe how the settings
affect the images displayed during the test. To end this demo, click on the
stop button that appears.
"""

import crappy
import numpy as np
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy with a software ROI setting.

  In the open method, add_software_roi adds a software ROI setting to the
  Camera object. By tuning this setting in the
  configuration window, the user can choose to keep only a part of the acquired
  image for processing, display, and recording.

  This class is based on the one defined in custom_camera_basic.py. Refer to
  that example for more information.
  """

  def __init__(self) -> None:
    """Almost the same as in custom_camera_basic.py.

    Here, we define a static image that is returned by the get_image method,
    potentially altered by the ROI. This way, the effect of the ROI is clearly
    visible.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Instantiating the base static image
    x, y = np.meshgrid(range(640), range(480))
    self._img = (x * y / 306081 * 255).astype('uint8')

  def open(self, **kwargs) -> None:
    """Compared to the custom_camera_basic.py example, we define here a
    software ROI setting using the add_software_roi method.

    Unlike the settings defined in the custom_camera_settings.py example, the
    software ROI setting is handled internally and in a standardized way.

    It is possible to update the width and height limits of the software ROI
    setting by calling the reload_software_roi method, but this advanced use is
    not demonstrated in the examples.
    """

    # Adding a setting for driving the software ROI
    self.add_software_roi(width=self._img.shape[1], height=self._img.shape[0])

    # This line is mandatory to apply the initial ROI parameters
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, np.ndarray]:
    """Compared with custom_camera_basic.py, this method returns the
    static image cropped by the selected software ROI.

    If the apply_soft_roi method is omitted, the ROI parameters are ignored and
    the full image is always returned.
    """

    # Applying the software ROI to the static image and returning it
    return time(), self.apply_soft_roi(self._img)

  def close(self) -> None:
    """Same as in custom_camera_basic.py, nothing to do here."""

    pass


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It acquires images and displays them in a dedicated Displayer window. It
  # features a setting for applying a software ROI to the images
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
