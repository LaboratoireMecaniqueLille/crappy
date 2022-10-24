# coding: utf-8

from typing import Callable, Union, Optional, List
import numpy as np
from pathlib import Path

from ..tool.videoextenso import LostSpotError, VideoExtenso as Ve
from ..tool.videoextensoConfig import VE_config
from .camera import Camera
from .displayer import Displayer


class Video_extenso(Camera):
  """This block can detect and track spots on images, and compute a strain
  value based on the displacement of the spots.

  It is meant to be used for following spots drawn on a sample during a tensile
  test, so that the local strain values can be deduced from the displacement of
  the spots. The spots are selected interactively in a GUI before the test
  starts.

  The timestamp, strain values, as well as the position of the detected spots
  are sent to downstream blocks for each received image.
  """

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               verbose: bool = False,
               freq: float = 200,
               save_images: bool = False,
               img_name: str = "{self._n_loops:6d}_{t-self.t0:.6f}.tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               labels: List[str] = None,
               raise_on_lost_spot: bool = True,
               white_spots: bool = False,
               update_thresh: bool = False,
               num_spots: Optional[int] = None,
               safe_mode: bool = False,
               border: int = 5,
               min_area: int = 150,
               blur: int = 5,
               **kwargs) -> None:
    """Sets the args and initializes the camera.

    Args:
      camera: The name of the camera to control. See :ref:`Cameras` for an
        exhaustive list of available cameras.
      transform: A function taking an image as an argument and returning a
        transformed image. The original image is discarded and only the
        transformed one is kept for processing, display and saving.
      config: If :obj:`True`, a config window is shown before the test starts
        for interactively tuning the camera settings. It also allows selecting
        the spots to track.
      display_images: If :obj:`True`, a window displays the acquired images
        in low resolution during the test. This display is mainly intended for
        debugging and visual follow-up, but not for displaying high-quality
        images.
      displayer_backend: If ``display_images`` is :obj:`True`, the backend to
        use for the display window. Should be one of :
        ::

          'cv2', 'mpl'

        If not given, OpenCV will be used if available.
      displayer_framerate: If ``display_images`` is :obj:`True`, sets the
        maximum framerate for updating the display window. This setting allows
        limiting the resources used by the displayer. Note that the actual
        achieved framerate might differ, this is just the maximum limit.
      verbose: If :obj:`True`, the achieved framerate will be displayed in the
        console during the test.
      freq: If given, the block will try to loop at this frequency. If it is
        lower than the framerate of the camera, frames will be dropped. This
        argument can be used for limiting the achieved framerate when the
        camera doesn't support framerate control.
      save_images: If :obj:`True`, the acquired images are saved on the
        computer during the test. Note that saving images uses CPU, so the
        achieved performance might drop when this feature is in use.
      img_name: If ``save_images`` is :obj:`True`, the template for naming the
        recorded images. It is evaluated as an `f-string`, and must contain the
        file extension at the end. For building the `f-string`, the
        ``self._n_loops`` attribute holds the loop number, and ``t-self.t0``
        holds the current timestamp.
      save_folder: If ``save_images`` is :obj:`True`, the directory to save
        images to. If it doesn't exist, it will be created. If not given, the
        images are saved in a folder named `Crappy_images` and created next to
        the file being run.
      save_period: If ``save_images`` is :obj:`True`, only one every this
        number of images will be saved.
      save_backend: The backend to use for saving the images. Should be one
        of :
        ::

          'sitk', 'pil', 'cv2'

        If not specified, SimpleITK will be used if available, then OpenCV as a
        second choice, and finally Pillow if none of the others was available.
      image_generator: A function taking two floats as arguments, and returning
        an image. It is only used for demonstration without camera in the
        examples, and isn't meant to be used in an actual test.
      labels: A :obj:`list` containing the labels to send to downstream blocks,
        carrying the information about the position of the tracked spots and
        the strain values. If not given, the default labels are :
        ::

          ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)']

      raise_on_lost_spot: If :obj:`True`, an exception is raised as soon as
        the block is losing track of a spot, what causes the test to stop.
        Otherwise, the block simply stops processing incoming images but
        doesn't raise any exception.
      white_spots: If :obj:`True`, detects white objects on a black background,
        else black objects on a white background.
      update_thresh: If :obj:`True`, the threshold for detecting the spots is
        re-calculated for each new image. Otherwise, the first calculated
        threshold is kept for the entire test. The spots are less likely to be
        lost with adaptive threshold, but the measurement will be more noisy.
        Adaptive threshold may also yield inconsistent results when spots are
        lost.
      num_spots: The number of spots to detect, between 1 and 4. The class will
        then try to detect this exact number of spots, and won't work if not
        enough spots can be found. If this argument is not given, at most 4
        spots can be detected but the class will work with any number of
        detected spots between 1 and 4.
      safe_mode: If :obj:`True`, the class will stop and raise an exception as
        soon as overlapping is detected. Otherwise, it will first try to reduce
        the detection window to get rid of overlapping. This argument should be
        used when inconsistency in the results may have critical consequences.
      border: When searching for the new position of a spot, the class will
        search in the last known bounding box of this spot plus a few
        additional pixels in each direction. This argument sets the number of
        additional pixels to use. It should be greater than the expected
        "speed" of the spots, in pixels / frame. But if it's too big, noise or
        other spots might hinder the detection.
      min_area: The minimum area an object should have to be potentially
        detected as a spot. The value is given in pixels, as a surface unit.
        It must of course be adapted depending on the resolution of camera and
        the size of the spots to detect.
      blur: The size in pixels of the kernel to use for applying a median blur
        to the image before the spot detection. If not given, no blurring is
        performed. A slight blur improves the spot detection by smoothening the
        noise, but also takes a bit more time compared to no blurring.
      **kwargs: Any additional argument to pass to the camera.
    """

    super().__init__(camera=camera,
                     transform=transform,
                     config=False,
                     display_images=False,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=None,
                     verbose=verbose,
                     freq=freq,
                     save_images=save_images,
                     img_name=img_name,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     **kwargs)

    self._ve = Ve(white_spots=white_spots,
                  update_thresh=update_thresh,
                  num_spots=num_spots,
                  safe_mode=safe_mode,
                  border=border,
                  min_area=min_area,
                  blur=blur)

    self.labels = ['t(s)', 'Coord(px)',
                   'Eyy(%)', 'Exx(%)'] if labels is None else labels

    # Setting the args
    self._display_images = display_images
    self._raise_on_lost_spot = raise_on_lost_spot
    self._config_ve = config
    self._lost_spots = False

    # Instantiating the displayer
    if display_images:
      self._displayer_ve = Displayer(f"Displayer {camera} "
                                     f"{Camera.cam_count[self._camera_name]}",
                                     displayer_framerate,
                                     displayer_backend)
    else:
      self._displayer_ve = None

  def prepare(self) -> None:
    """Opening the camera, starting the Video Extenso config and the tracker
    processes."""

    super().prepare()

    if self._config_ve:
      config = VE_config(self._camera, self._ve)
      config.main()

    self._ve.start_tracking()

    if self._displayer_ve is not None:
      self._displayer_ve.prepare()

  def finish(self) -> None:
    """Stopping the tracker processes, the displayer and the camera."""

    self._ve.stop_tracking()

    if self._displayer_ve is not None:
      self._displayer_ve.finish()

    super().finish()

  def _additional_loop(self, t: float, img: np.ndarray) -> None:
    """Gets the data from the latest image, sends it, and updates the
    display."""

    # Trying to process the incoming images
    if not self._lost_spots:
      try:
        data = self._ve.get_data(img)
        if data is not None:
          self.send([t - self.t0, *data])
      except LostSpotError:
        self._ve.stop_tracking()
        # Raising if specified by the user
        if self._raise_on_lost_spot:
          raise
        # Otherwise, simply setting a flag so that no additional processing is
        # performed
        else:
          self._lost_spots = True
          print("[VideoExtenso] Spots lost, not processing data anymore !")

    # Updating the display
    if self._displayer_ve is not None:

      # Adding the contours on the image only if the spots were not lost
      if not self._lost_spots:
        for spot in self._ve.spots:
          self._draw_box(img, spot)

      # Updating the display in all cases
      self._displayer_ve.update(img)
