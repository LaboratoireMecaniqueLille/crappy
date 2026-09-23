# coding: utf-8

"""
This example demonstrates how to add several settings to a custom Camera object
in Crappy. It builds on custom_camera_basic.py, which should be read first. It
does not require any hardware, but requires the Pillow and opencv-python Python
modules.

Crappy Camera objects can implement settings that usually correspond to
controls available on the hardware. Several setting types are available, each
with its own behavior. All settings can be adjusted interactively in the
configuration window before the test starts. This example shows the methods and
syntax for creating and accessing them.

Here, a Camera object is instantiated and driven by a Camera Block that
displays the acquired images. The Camera object generates random images and
features several settings for tuning image generation. The goal is to
show how to add and access camera settings in Camera objects.

After starting this script, a configuration window appears in which you can see
the generated images. Four settings can be tuned, and they affect the generated
images. Modify their values, click the Apply Settings button, and observe how
the image changes. When you're done, close the configuration window and observe
how the settings affect the images displayed during the test. To end this demo,
click on the stop button that appears.
"""

import crappy
import numpy as np
import numpy.random as rd
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy with various settings.

  The settings are instantiated in the open method, and can be of different
  types. They are then used in the get_image method to control the random image
  generation. These settings can be tuned interactively by the user in the
  configuration window before the test starts.

  This class is based on the one defined in custom_camera_basic.py. Refer to
  that example for more information.
  """

  def __init__(self) -> None:
    """Almost the same as in custom_camera_basic.py.

    Here, the _width attribute is initialized.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Defining attribute
    self._width: int = 640

  def open(self, **kwargs) -> None:
    """In this method, four settings are instantiated.

    They tune parameters of the random image generation and are used
    in the get_image method.

    Settings of the three main types are instantiated to give an overview of
    what the Camera class supports.
    """

    # Adding a Boolean setting that can only be True or False
    # We do not add getter or setter methods, so the setting value is stored
    # internally in a buffer and is unrelated to the hardware
    self.add_bool_setting(name='color',
                          getter=None,
                          setter=None,
                          default=False)
    # Adding a scale setting that can take only integer values in a given
    # range and appears as a slider in the configuration window
    # Since a getter and a setter are given, this setting would normally be
    # read and set directly on hardware (this example is hardware-free though)
    self.add_scale_setting(name='width',
                           lowest=2,
                           highest=640,
                           getter=self._get_width,
                           setter=self._set_width,
                           default=640)
    # This other scale setting does not have getter or setter methods, so it is
    # unrelated to hardware.
    self.add_scale_setting(name='height',
                           lowest=2,
                           highest=480,
                           getter=None,
                           setter=None,
                           default=480)
    # Adding a choice setting that can take a value from a predefined set
    self.add_choice_setting(name='filter',
                            choices=('None', 'Square', 'Binary'),
                            getter=None,
                            setter=None,
                            default='None')

    # This line is mandatory to apply the initial settings
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, np.ndarray]:
    """This method is an extension of the custom_camera_basic.py one.

    Instead of always generating a random image in the same way, this method
    uses adjustable image parameters. These parameters are instantiated in the
    open method and can be tuned in the configuration window.

    The instantiated settings can be accessed directly by calling
    self.<setting_name>, which simplifies their integration and avoids the need
    to use buffers such as self._width in every case.
    """

    # Getting the size of the image to generate, based on the settings values
    # Note how the height setting is directly accessible by calling self.height
    # The width setting can also be accessed this way, but here we access it
    # through the underlying self._width buffer
    size = (self.height, self._width)
    # Adding 3 color channels if the color image option is checked
    # Note how the color setting is directly accessible by calling self.color
    if self.color:
      size = (*size, 3)

    # Generating the random image
    img = rd.randint(low=0, high=256, size=size, dtype='uint8')

    # Filtering the data if requested by the filter setting
    # Note how the filter setting is directly accessible by calling self.filter
    if self.filter == 'Square':
      # Squaring the image and clamping it back between 0 and 255
      img **= 2
      img = ((img - np.min(img)) /
             (np.max(img) - np.min(img)) * 255).astype('uint8')
    elif self.filter == 'Binary':
      # Forcing each pixel value either at 0 or at 255
      img = np.where(img > 128, 255, 0)

    return time(), img

  def close(self) -> None:
    """Same as in custom_camera_basic.py, nothing to do here."""

    pass

  def _set_width(self, width: int) -> None:
    """This method is a setter setting the value of the width setting.

    Normally it would set this parameter directly on hardware, but this demo
    was designed to run completely virtually.
    """

    self._width = width

  def _get_width(self) -> int:
    """This method is a getter returning the value of the width setting.

    Normally this value would be read from hardware, but this demo was designed
    to run completely virtually.
    """

    return self._width


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It acquires images and displays them in a dedicated Displayer window. It
  # features a few settings that the user can tune in the configuration window.
  cam = crappy.blocks.Camera(
      'CustomCam',  # The name of the custom Camera that was just written
      config=True,  # Prefer enabling configuration when possible
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
