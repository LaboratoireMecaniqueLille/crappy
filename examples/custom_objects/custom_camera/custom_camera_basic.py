# coding: utf-8

"""
This example demonstrates the instantiation of a custom Camera object in
Crappy. The example presented here shows only the basic steps for creating a
Camera object. It does not require any hardware to run, but necessitates the
Pillow and opencv-python Python modules to be installed.

In Crappy, users can define their own Camera objects and use them along with
the Camera Block and the other image-processing Blocks like VideoExtenso. This
way, users can interface with their own hardware without having to integrate it
in the distributed version of Crappy.

Here, a very simple Camera object is instantiated, and driven by a Camera Block
that displays the acquired images. The Camera object generates images randomly,
and does not feature any setting. The goal here is to show the basic methods to
use for creating a custom Camera object. Note that in addition, A StopButton
Block allows stopping the script properly without using CTRL+C by clicking on a
button.

After starting this script, a configuration window appears in which you can see
the generated images. There is no setting to tune. Close this window to start
the test, the actual smaller displayer window should then appear. You can see
how the code written in this example translates to a usable Camera object. To
end this demo, click on the stop button that appears. You can also hit CTRL+C,
but it is not a clean way to stop Crappy.
"""

import crappy
import numpy as np
import numpy.random as rd
from typing import Tuple
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy.

  It is fully recognized by Crappy as a Camera, and can be used by the Camera
  Block and the other image-processing Blocks.

  Each Camera class must be a child of crappy.camera.Camera, otherwise it is
  not recognized as such.
  """

  def __init__(self) -> None:
    """In this method you can initialize all the Python objects necessary for
    driving the camera.

    Also, don't forget to initialize the parent class or your Camera object
    won't be recognized by Crappy.

    Not that this method takes no argument, the arguments passed to the Camera
    objects are given as kwargs in the open method.

    There is nothing to perform here in this simple demo.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

  def open(self, **kwargs) -> None:
    """In this method you would perform any action needed to connect to the
    camera and start the image acquisition.

    There is no action to perform in this simple demo though, except calling
    the set_all method which is a good practice.

    The arguments passed to the Camera object are given here as kwargs, and not
    in the __init__ method.
    """

    # This line is strongly recommended at the end of the open method,
    # otherwise the settings of the camera are not set at all
    # Here the Camera does not include settings though
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, np.ndarray]:
    """This method must return the current timestamp as well as an acquired
    image.

    In this simple demo the image is generated randomly and there's nothing
    more to do.
    """

    return time(), rd.randint(low=0, high=256, size=(480, 640), dtype='uint8')

  def close(self) -> None:
    """In this method you would perform any action needed to disconnect from
    the camera and release the resources.

    There is no action to perform in this simple demo though.
    """

    pass


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It simply acquires images and displays them in a dedicated Displayer window
  cam = crappy.blocks.Camera(
      'CustomCam',  # The name of the custom Camera that was just written
      config=True,  # easier to set it to True when possible
      display_images=True,  # Displaying the images to show how they look
      displayer_framerate=30,  # Setting a nice framerate on the display
      freq=30,  # Lowering the default frequency because it's just a demo
      save_images=False,  # No need to record the images in this demo

      # Sticking to default for the other arguments
  )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
