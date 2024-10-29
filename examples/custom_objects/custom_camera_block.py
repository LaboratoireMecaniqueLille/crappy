# coding: utf-8

"""
This example demonstrates the instantiation of a custom child of the Camera
Block in Crappy. The example presented here also shows the instantiation of a
CameraProcess object, necessary for implementing a custom Camera Block, as well
as an Overlay object, optional to use with the CameraProcess. This example
necessitates a camera compatible with OpenCV torun, and requires the Pillow and
opencv-python Python modules to be installed.

For advanced users of Crappy, it is possible to define their own children of
the Camera Block for performing custom image processing. This way, users can
adapt the good performance of Crappy for image management to their own needs.
Compared to the other custom objects of Crappy, this one is however among the
most complex to create.

Here, a new Camera Block is instantiated that performs eye detection on images
acquired from the Webcam Camera. The eye detection is performed by a separate
CameraProcess object, also defined in the script. During the test, the acquired
images are displayed in a dedicated Displayer window. In this window, a third
custom-defined Overlay class allows to draw ellipses outlining the detected
eyes, for a real-time follow-up of the processing. The coordinates of the
detected eyes are sent to a LinkReader Block, for display in the terminal. The
goal of this script is to demonstrate the steps for creating a custom Camera
Block object.

After starting this script and closing the configuration window, you should
film a face with the camera and see in the Displayer window how the eyes are
detected. A StopButton Block allows to stop this script when you're done using
it. It might appear under the Displayer window. Alternatively, you can also hit
CTRL+C to stop Crappy, but it is not a clean way to do it.
"""

import crappy
import cv2
import numpy as np
from typing import Optional, Callable, Union
from pathlib import Path


class Ellipse(crappy.tool.camera_config.Overlay):
  """This class demonstrates the instantiation of a custom Overlay object in
  Crappy.

  It allows to draw an ellipse as an overlay of the images displayed by a
  Camera Block whose display_images argument is set to True.

  It is given as an argument of the send_to_draw method of a CameraProcess, to
  be sent to the Displayer Process in charge of displaying the acquired images.
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
                0, 0, 360, 0, thickness)


class CustomCameraProcess(crappy.blocks.camera_processes.CameraProcess):
  """This class demonstrates the instantiation of a custom CameraProcess object
  in Crappy.

  It is used by a child of the Camera block for performing image processing on
  the acquired images in a parallelized way. It is mandatory to define a
  CameraProcess object for users wishing to implement their own image
  processing in Crappy.

  The CameraProcess objects can send Overlay objects via the send_to_draw
  method, to draw overlays on top of the displayed images if the display_images
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

    It is strongly recommended to perform as little as possible in this method,
    because objects instantiated already here might get buggy when used in
    later methods.

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
    """This method should perform the main image processing task, and send the
     result to downstream Blocks.

     Here, Overlay objects can also be sent to the Displayer Process for adding
     overlays on top of the displayed images.
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
  """This class demonstrates the instantiation of a custom child of the Camera
  Block in Crappy.

  It mainly indicates which CameraProcess object to use for image processing.
  It is also the object that the user ultimately uses in its Crappy script.

  Here, the CustomCameraProcess defined above is given as the Process to use
  for image processing.
  """

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               display_freq: bool = False,
               freq: Optional[float] = 200,
               debug: Optional[bool] = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
               scale_factor: float = 1.2,
               min_neighbors: int = 3,
               **kwargs) -> None:
    """This method should initialize the Python objects used in this class and
    handle the provided arguments.
    
    It also initializes the parent Camera Block, and provides it with all its
    possible arguments. Note that only camera is a mandatory argument, so all
    the other ones could be left to default. It was chosen to include them here
    to remind users that is it better if children of the Camera Block still 
    have them accessible.
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

    Here, it just sets the CustomCameraProcess defined above as the one to use
    for performing the image processing.
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
      displayer_framerate=20,  # The displayer window is allowed to display up
      # to 20 images per second
      scale_factor=1.2,  # Argument passed to the CustomCameraProcess object
      min_neighbors=6,  # Argument passed to the CustomCameraProcess object

      # Sticking to default for the other arguments
  )

  # This StopBlock checks if the data from the Button Block satisfies any of
  # its stop criteria, in which case it stops the test
  stop = crappy.blocks.StopButton()

  # This LinkReader Block displays all the data it receives from the
  # CustomCameraBlock in the terminal. This way, you can have a clear overview
  # of what the custom Camera Block sends to downstream Blocks.
  reader = crappy.blocks.LinkReader()

  # Linking the Block so that the information is correctly sent and received
  crappy.link(cam, reader)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
