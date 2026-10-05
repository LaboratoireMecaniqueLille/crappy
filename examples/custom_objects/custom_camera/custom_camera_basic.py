# coding: utf-8

"""
This example demonstrates how to instantiate a custom Camera object in Crappy.
It shows the basic steps for creating a Camera object. It does not require any
hardware, but requires the Pillow and opencv-python Python modules.

In Crappy, users can define their own Camera objects and use them with
the Camera Block and the other image-processing Blocks like VideoExtenso. This
lets users interface with their own hardware without integrating it into the
distributed version of Crappy.

Here, a very simple Camera object is instantiated and driven by a Camera Block
that displays the acquired images. The Camera object generates random images
and does not feature any settings. The goal is to show the basic methods for
creating a custom Camera object.

After starting this script, a configuration window appears in which you can see
the generated images. There are no settings to tune. Close this window to start
the test; the smaller displayer window should then appear. You can see how the
code written in this example translates to a usable Camera object. To
end this demo, click on the stop button that appears.
"""

import crappy
import numpy as np
import numpy.random as rd
from time import time


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy.

  It is fully recognized by Crappy as a Camera, and can be used by the Camera
  Block and the other image-processing Blocks.

  Each Camera class must inherit from crappy.camera.Camera, otherwise, Crappy
  does not recognize it as a Camera.
  """

  def __init__(self) -> None:
    """In this method you can initialize all the Python objects necessary for
    driving the camera.

    Remember to initialize the parent class so that Crappy recognizes the
    Camera object.

    Note that this method takes no arguments. Arguments passed to Camera
    objects are given as keyword arguments to the open method.

    There is nothing to perform here in this simple demo.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

  def open(self, **kwargs) -> None:
    """In this method you would perform any action needed to connect to the
    camera and start the image acquisition.

    This simple demo only calls the set_all method, as recommended.

    Arguments passed to the Camera object are given here as keyword arguments,
    not in the __init__ method.
    """

    # This line is strongly recommended at the end of the open method,
    # otherwise the settings of the camera are not set at all
    # Here the Camera does not include settings though
    self.set_all(**kwargs)

  def get_image(self) -> tuple[float, np.ndarray]:
    """This method must return the current timestamp as well as an acquired
    image.

    In this simple demo, the image is generated randomly.
    """

    return time(), rd.randint(low=0, high=256, size=(480, 640), dtype='uint8')

  def close(self) -> None:
    """In this method you would perform any action needed to disconnect from
    the camera and release the resources.

    There is no action to perform in this simple demo.
    """

    pass


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It simply acquires images and displays them in a dedicated Displayer window
  cam = crappy.blocks.Camera(
      'CustomCam',  # The name of the custom Camera that was just written
      config=True,  # Easier to set to True when possible
      display_images=True,  # Displaying the images to show how they look
      displayer_framerate=30,  # Displaying up to 30 frames per second
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
