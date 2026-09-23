# coding: utf-8

"""
This example demonstrates the instantiation of a custom Camera Block subclass
in Crappy. It also shows how to instantiate a CameraProcess, which is required
for implementing a custom Camera Block, and an optional Overlay object. This
example requires a camera compatible with OpenCV and the Pillow and
opencv-python Python modules.

Advanced Crappy users can define Camera Block subclasses that perform custom
image processing. This lets users adapt Crappy's image-management features to
their own needs. Of Crappy's custom objects, this is among the most complex to
create.

Here, a new Camera Block is instantiated that performs eye detection on images
acquired from the Webcam Camera. The eye detection is performed by a separate
CameraProcess object, also defined in the script. During the test, the acquired
images are displayed in a dedicated Displayer window. In this window, a third
custom-defined Overlay class draws ellipses outlining the detected eyes,
providing real-time feedback on the processing. The coordinates of the
detected eyes are sent to a LinkReader Block for display in the terminal. The
goal of this script is to demonstrate the steps for creating a custom Camera
Block object.

After starting this script and closing the configuration window, you should
film a face with the camera and see in the Displayer window how the eyes are
detected. A StopButton Block stops this script cleanly when you're done using
it. It might appear under the Displayer window.
"""

import crappy
import cv2
import numpy as np
from collections.abc import Callable
from pathlib import Path
from typing import Literal


class Ellipse(crappy.tool.camera_config.Overlay):
  """This class demonstrates the instantiation of a custom Overlay object in
  Crappy.

  It draws an ellipse as an overlay of the images displayed by a
  Camera Block whose display_images argument is set to True.

  It is passed to the send_to_draw method of a CameraProcess and then sent to
  the Displayer Process that displays the acquired images.
  """

  def __init__(self,
               center_x: int,
               center_y: int,
               x_axis: int,
               y_axis: int) -> None:
    """This method should initialize the Python objects used in this class and
    handle the provided arguments.

    Args:
      center_x: The x coordinate of the center of the ellipse to draw, as an
        integer.
      center_y: The y coordinate of the center of the ellipse to draw, as an
        integer.
      x_axis: The half of the major axis of the ellipse in the x direction.
      y_axis: The half of the major axis of the ellipse in the y direction.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Storing the given arguments as attributes
    self._center_x = center_x
    self._center_y = center_y
    self._x_axis = x_axis
    self._y_axis = y_axis

  def draw(self, img: np.ndarray) -> None:
    """This method draws the overlay on the image to display.

    Args:
      img: The Python object containing the image to display, usually as a
        numpy array.
    """

    # Adjusting the thickness of the line to the size of the image
    thickness = max(img.shape[0] // 480, img.shape[1] // 640, 1) + 1

    # Actually drawing the ellipse
    # Refer to OpenCV's documentation for more information on the significance
    # of each positional argument
    cv2.ellipse(img,
                (self._center_x, self._center_y),
                (self._x_axis, self._y_axis),
                0., 0., 360., (0.,), thickness)


class CustomCameraProcess(crappy.blocks.camera_processes.CameraProcess):
  """This class demonstrates the instantiation of a custom CameraProcess object
  in Crappy.

  A Camera Block subclass uses it to process acquired images in parallel.
  Users implementing their own image processing in Crappy must define a
  CameraProcess object.

  CameraProcess objects can send Overlay objects via the send_to_draw method to
  draw overlays on top of the displayed images if the display_images
  argument of the Camera Block is set to True.

  Here, this class performs human eye detection on the images it receives. It
  then sends the coordinates of the detected eyes to downstream Blocks, and
  sends the outline of the detected eyes for display via Ellipse Overlay
  objects.
  """

  def __init__(self,
               scale_factor: float = 1.2,
               min_neighbors: int = 3) -> None:
    """This method should initialize the Python objects used in this class and
    handle the provided arguments.

    Perform as little work as possible in this method, because objects created
    here may not behave correctly when used in later methods.

    Args:
      scale_factor: Parameter specifying how much the image size is reduced at
        each image scale iteration of the CascadeClassifier.
      min_neighbors: Parameter specifying how many neighbors each candidate
        detected shape should have to retain it.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    super().__init__()

    # Reserving an attribute for the image processing object
    # Not setting it though, this should be done in a later method
    self._eye_cascade = None

    # Storing the given arguments as attributes
    self._scale_factor = scale_factor
    self._min_neighbors = min_neighbors

  def init(self) -> None:
    """This method should initialize the Python objects that will be used for
    image processing.

    It is fine not to define this method if there is nothing specific to
    perform here.
    """

    # A CascadeClassifier is used for performing the eye detection
    self._eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')

  def loop(self) -> None:
    """This method should perform the main image-processing task and send the
    result to downstream Blocks.

    Here, Overlay objects can also be sent to the Displayer Process to add
    overlays to the displayed images.
    """

    # This line performs the eye detection. The self.img attribute contains the
    # latest received image, and the loop method is only called if self.img is
    # updated. This way, you can be sure that the image contained in self.img
    # is always a different one.
    eyes = self._eye_cascade.detectMultiScale(self.img,
                                              scaleFactor=self._scale_factor,
                                              minNeighbors=self._min_neighbors)

    # Instantiating an Ellipse Overlay object for each detected eye
    # The coordinate system for the detected eyes and the ellipses are
    # different, hence the conversion
    to_draw = list()
    for (x, y, width, height) in eyes:
      to_draw.append(Ellipse(int(x + width / 2), int(y + height / 2),
                             int(width / 2), int(height / 2)))

    # Sending the Overlays to draw to the Process in charge of displaying the
    # acquired images
    self.send_to_draw(to_draw)
    # Sending the coordinates of the detected eyes to the downstream Blocks.
    # The self.metadata attribute contains the metadata corresponding to the
    # image contained in self.img, and in particular its timestamp.
    self.send({'t(s)': self.metadata['t(s)'], 'eyes': eyes})

  def finish(self) -> None:
    """This method should de-initialize the Python objects that were used for
    image processing.

    It is fine not to define this method if there is nothing specific to
    perform here.
    """

    ...


class CustomCameraBlock(crappy.blocks.Camera):
  """Demonstrate the instantiation of a custom Camera Block subclass.

  It mainly indicates which CameraProcess object to use for image processing.
  It is also the object that users ultimately instantiate in their Crappy
  scripts.

  Here, the CustomCameraProcess defined above is given as the Process to use
  for image processing.
  """

  def __init__(self,
               camera: str,
               transform: Callable[[np.ndarray], np.ndarray] | None = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Literal['cv2', 'mpl'] | None = None,
               displayer_framerate: float = 5,
               software_trig_label: str | None = None,
               display_freq: bool = False,
               freq: float | None = 200,
               debug: bool | None = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: str | Path | None = None,
               save_period: int = 1,
               save_backend: Literal['sitk', 'pil',
                                     'cv2', 'npy'] | None = None,
               image_generator: Callable[[float, float],
                                         np.ndarray] | None = None,
               img_shape: tuple[int, int] | tuple[int, int, int] | None = None,
               img_dtype: str | None = None,
               scale_factor: float = 1.2,
               min_neighbors: int = 3,
               **kwargs) -> None:
    """This method should initialize the Python objects used in this class and
    handle the provided arguments.

    It also initializes the parent Camera Block, and provides it with all its
    possible arguments. Note that only camera is a mandatory argument, so all
    the other ones could be left to default. It was chosen to include them here
    so that subclasses of the Camera Block continue to expose them.
    """

    # Mandatory line usually at the very beginning of the __init__ method
    # Most of the arguments of this Block are actually intended for the parent
    # class
    super().__init__(camera=camera,
                     transform=transform,
                     config=config,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     display_freq=display_freq,
                     freq=freq,
                     debug=debug,
                     save_images=save_images,
                     img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     img_shape=img_shape,
                     img_dtype=img_dtype,
                     **kwargs)

    # Storing the other given arguments as attributes
    self._scale_factor = scale_factor
    self._min_neighbors = min_neighbors

  def prepare(self) -> None:
    """In this method, the CameraProcess to use for image processing should be
    set as the self.process_proc argument.

    This method should also run the prepare method of the parent Block.

    These are the only mandatory actions that this method has to perform,
    although it can perform any other action that would be required for your
    specific needs.

    Here, it sets the CustomCameraProcess defined above as the image processor.
    """

    # Setting the CameraProcess to use
    self.process_proc = CustomCameraProcess(self._scale_factor,
                                            self._min_neighbors)
    # It is mandatory to run the prepare method of the parent class at the end
    # of this method
    super().prepare()


if __name__ == '__main__':

  # This CustomCameraBlock acquires images via a Webcam Camera object, and
  # processes them using a CustomCameraProcess. The output of the processing is
  # sent to the LinkReader Block for display in the terminal. In addition, the
  # acquired images are displayed in a displayer window, with overlay shapes
  # drawn on top to provide real-time feedback on how the processing performs.
  cam = CustomCameraBlock(
      'Webcam',  # The name of the Camera to acquire images from. Here, a
      # camera readable by OpenCV must be used, typically a webcam will do
      display_images=True,  # The acquired images will be displayed in a
      # dedicated window
      save_images=False,  # The acquired images will not be recorded
      freq=20,  # The maximum allowed acquisition frequency of the camera
      displayer_framerate=20,  # The displayer window can display up
      # to 20 images per second
      scale_factor=1.2,  # Argument passed to the CustomCameraProcess object
      min_neighbors=6,  # Argument passed to the CustomCameraProcess object

      # Sticking to default for the other arguments
  )

  # This Block provides a clean way to stop the test
  stop = crappy.blocks.StopButton()

  # This LinkReader Block displays all the data it receives from the
  # CustomCameraBlock in the terminal. This way, you can have a clear overview
  # of what the custom Camera Block sends to downstream Blocks.
  reader = crappy.blocks.LinkReader()

  # Linking the Block so that the information is correctly sent and received
  crappy.link(cam, reader)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
