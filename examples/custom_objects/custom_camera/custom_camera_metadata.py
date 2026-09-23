# coding: utf-8

"""
This example demonstrates custom metadata handling in a custom Crappy Camera
object. It builds on custom_camera_basic.py, which should be read first. It
does not require any hardware, but requires the Pillow and opencv-python Python
modules.

In Camera objects, it is possible for the get_image method to return an entire
metadata dictionary instead of just a timestamp. This way, metadata from the
camera can be recorded along with the images in a metadata.csv file. The
inclusion of custom metadata is straightforward, but there are certain rules
to follow, as shown in this example.

Here, a very simple Camera object is instantiated and driven by a Camera Block
that displays the acquired images. The Camera object generates random images
and does not feature any settings. One out of every three images is recorded in
a newly created demo_metadata folder, along with a metadata.csv file containing
the metadata of all recorded images. The goal is to show how to implement
custom metadata acquisition in Camera objects.

After starting this script, a configuration window appears in which you can see
the generated images. There are no settings to tune. Close this window to start
the test, the smaller displayer window should then display the acquired images.
After a few seconds, click the stop button to end the demo. A demo_metadata
folder should have been created in the current working directory. It contains
the recorded images and a metadata.csv file. You can open this file and check
that the metadata returned by the get_image method was saved correctly.
"""

import crappy
import numpy as np
import numpy.random as rd
from typing import Any
from time import time, strftime, gmtime


class CustomCam(crappy.camera.Camera):
  """This class demonstrates the instantiation of a custom Camera object in
  Crappy, with custom metadata handling.

  For each image, a custom metadata dictionary is returned by the get_image
  method. Some returned fields use valid EXIF tags so that they can be
  embedded in the recorded images.

  This class is based on the one defined in custom_camera_basic.py. Refer to
  that example for more information.
  """

  def __init__(self) -> None:
    """Almost the same as in custom_camera_basic.py.

    Here, a frame counter is added.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    self._count: int = 0

  def open(self, **kwargs) -> None:
    """Same as in custom_camera_basic.py, not much to do here."""

    # This line is strongly recommended at the end of the open method,
    # otherwise the settings of the camera are not set at all
    # Here the Camera does not include settings though
    self.set_all(**kwargs)

  def get_image(self) -> tuple[dict[str, Any], np.ndarray]:
    """Compared to the custom_camera_basic.py example, this method returns a
    complete dictionary of metadata instead of just a timestamp.

    In the returned metadata, it is mandatory to include the 't(s)' and
    'ImageUniqueID' fields. For the other fields, users are free to choose the
    field names and values. Using valid EXIF tags is preferable.

    In the custom_camera_basic.py example, a similar (although less extensive)
    metadata dictionary is generated automatically from the returned timestamp.
    You can enable image recording and check the metadata file to verify.
    """

    # Grabbing the timestamp
    t = time()
    # Generating a random image
    img = rd.randint(low=0, high=256, size=(480, 640), dtype='uint8')
    # Generating some metadata
    meta = {'t(s)': t,  # Mandatory
            # The current date in the valid EXIF format
            'DateTimeOriginal': strftime("%Y:%m:%d %H:%M:%S", gmtime(t)),
            # The decimal part of the timestamp
            'SubsecTimeOriginal': f'{t % 1:.6f}',
            'ImageUniqueID': self._count,  # Mandatory
            'ImageWidth': img.shape[1],  # The width of the image in pixels
            'ImageHeight': img.shape[0],  # The height of the image in pixels
            'Orientation': 1}  # Regular horizontal orientation
    # Updating the image counter
    self._count += 1

    return meta, img

  def close(self) -> None:
    """Same as in custom_camera_basic.py, nothing to do here."""

    pass


if __name__ == '__main__':

  # This Camera Block drives the CustomCam Camera object that we just created.
  # It acquires images, displays them in a dedicated Displayer window, and also
  # records them along with their metadata
  cam = crappy.blocks.Camera(
      'CustomCam',  # The name of the custom Camera that was just written
      config=True,  # Easier to set to True when possible
      display_images=True,  # Displaying the images to show how they look
      displayer_framerate=30,  # Setting same framerate as acquisition
      freq=30,  # Lowering the frequency because it's just a demo
      save_images=True,  # The images are recorded to show that the metadata
      # is recorded
      save_folder='demo_metadata',  # The images are saved in this newly
      # created folder in the current working directory
      save_period=3,  # Saving one out of three images to limit disk usage

      # Sticking to default for the other arguments
  )

  # This Block allows the user to properly exit the script
  stop = crappy.blocks.StopButton(
      # No specific argument to give for this Block
  )

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
