# coding: utf-8

"""
This example demonstrates the instantiation of a custom Camera object in
Crappy, with the integration of a software ROI setting. This example is based
on the custom_camera_basic.py, that should first be read for a better
understanding. It does not require any hardware to run, but necessitates the
Pillow and opencv-python Python modules to be installed.

In Camera objects, Crappy offers the possibility to easily implement a software
ROI setting. Using this setting, the user can select in the configuration
window which part of the image (Region of Interest) to keep for processing,
display, and recording. The ROI is always rectangular, and the user can tune
its x and y offset as well as it height and width. The addition of a software
ROI setting only requires calling two methods, as demonstrated in this example.

Here, a very simple Camera object is instantiated, and driven by a Camera Block
that displays the acquired images. The Camera object features a software ROI
setting, that lets the user select the ROI in the configuration window. The
goal here is to show how to implement a software ROI in Camera objects. Note
that in addition, A StopButton Block allows stopping the script properly
without using CTRL+C by clicking on a button.

After starting this script, a configuration window appears in which you can see
the generated images. There are four settings that you can tune, that
correspond to software ROI settings. Change their values, click on the Apply
Setting button, and see how the image is impacted. When you're done, close the
configuration window and see how the settings values are applied to the
displayed images during the test. To end this demo, click on the stop button
that appears. You can also hit CTRL+C, but it is not a clean way to stop
Crappy.
"""

import crappy
import numpy as np
from typing import Tuple
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy, with the instantiation of a software ROI setting.

  In the open method, the add_software_roi method is called for adding a
  software ROI setting to the Camera object. By tuning this setting in the
  configuration window, the user can choose to keep only a part of the acquired
  image for processing, display, and recording.

  This class is based on the one defined in custom_camera_basic.py, please
  refer to that example for more information.
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

    # This line is mandatory here for first applying the parameters of the ROI
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, np.ndarray]:
    """Compared to the one in custom_camera_basic.py, this method returns the
    static image cropped by the selected software ROI.

    If the apply_soft_roi method is omitted, the parameters of the ROi are just
    ignored and the full image is always returned.
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
      config=True,  # easier to set it to True when possible
      display_images=True,  # Displaying the images to show how they look
      displayer_framerate=30,  # Setting same framerate as acquisition
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
