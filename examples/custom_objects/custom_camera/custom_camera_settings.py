# coding: utf-8

"""
This example demonstrates the instantiation of a custom Camera object in
Crappy, with the integration of several camera settings. This example is based
on the custom_camera_basic.py, that should first be read for a better
understanding. It does not require any hardware to run, but necessitates the
Pillow and opencv-python Python modules to be installed.

In camera objects, Crappy offers the possibility of implementing camera
settings that usually correspond to a setting tunable on the hardware. Several
types of settings are available, with each their specificities. All the
settings can be tuned interactively in the configuration window before starting
the test. The instantiation and access to the camera settings uses specific
methods and syntax, that are shown in this example.

Here, a Camera object is instantiated, and driven by a Camera Block that
displays the acquired images. The Camera object generates random images, and
features several settings for tuning the image generation. The goal here is to
show how to add and access camera setting in Camera objects. Note that in
addition, A StopButton Block allows stopping the script properly without using
CTRL+C by clicking on a button.

After starting this script, a configuration window appears in which you can see
the generated images. There are four settings that you can tune, and that
affect the generated images. You can modify their values, click on the Apply
Setting button, and see how the image is impacted. When you're done, close the
configuration window and see how the settings values are applied to the
displayed images during the test. To end this demo, click on the stop button
that appears. You can also hit CTRL+C, but it is not a clean way to stop
Crappy.
"""

import crappy
import numpy as np
import numpy.random as rd
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy, with the instantiation of various settings.

  The settings are instantiated in the open method, and can be of different
  types. They are then used in the get_image method to control the random image
  generation. These settings can be tuned interactively by the user in the
  configuration window before the test starts.

  This class is based on the one defined in custom_camera_basic.py, please
  refer to that example for more information.
  """

  def __init__(self) -> None:
    """Almost the same as in custom_camera_basic.py.

    Here, the _width attributes is instantiated.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Defining attribute
    self._width: int = 640

  def open(self, **kwargs) -> None:
    """In this method, 4 settings are instantiated.

    They allow to tune parameters of the random image generation, and are used
    in the get_image method.

    Settings of the three main types are instantiated, to give an overview of
    what is implemented in the Camera class.
    """

    # Adding a bool setting, that can only take the True or False values.
    # We don't add a getter and a setter method, so the value of the setting is
    # stored internally in a buffer and is unrelated to the hardware
    self.add_bool_setting(name='color',
                          getter=None,
                          setter=None,
                          default=False)
    # Adding a scale setting, that can take only integer values in a given
    # range and appears as a slider in the configuration window
    # Since a getter and a setter are given, this setting would normally be
    # read and set directly on hardware (this example is hardware-free though)
    self.add_scale_setting(name='width',
                           lowest=2,
                           highest=640,
                           getter=self._get_width,
                           setter=self._set_width,
                           default=640)
    # This other scale setting does not have a getter and a setter, so it is
    # unrelated to hardware.
    self.add_scale_setting(name='height',
                           lowest=2,
                           highest=480,
                           getter=None,
                           setter=None,
                           default=480)
    # Adding a choice setting that can take any value in a predefined set of
    # possible choices.
    self.add_choice_setting(name='filter',
                            choices=('None', 'Square', 'Binary'),
                            getter=None,
                            setter=None,
                            default='None')

    # This line is mandatory here for first applying the instantiated settings
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, np.ndarray]:
    """This method is an extension of the custom_camera_basic.py one.

    Instead of just generating the random image always in a same way, it is
    here possible to tune parameters of the image. These parameters are the
    ones that were instantiated in the open method, and they can be tuned in
    the configuration window.

    The instantiated settings can be accessed directly by calling
    self.<setting_name>, which greatly simplifies their integration in the code
    and doesn't make it necessary to always use buffers like self._width.
    """

    # Getting the size of the image to generate, based on the settings values
    # Note how the height setting is directly accessible by calling self.height
    # The width setting can also be accessed this way, but here we access it
    # through the underlying self._weight buffer
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
  # features a few settings tha the user can tune in the configuration window.
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
