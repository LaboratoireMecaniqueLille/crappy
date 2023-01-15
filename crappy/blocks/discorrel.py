# coding: utf-8

import numpy as np
from typing import Callable, Optional, Union, List, Dict, Any
from pathlib import Path

from ..tool import DISCorrel as Dis
from ..tool import DiscorrelConfig, Box
from .camera import Camera
from .displayer import Displayer


class DISCorrel(Camera):
  """This block performs Dense Inverse Search on an image using OpenCV's
  DISOpticalFlow.

  First, a region of interest has to be selected in a GUI. At each new frame,
  the displacement field between the frame and the reference image is projected
  on the desired base of fields, and the returned results are then the
  average values of each field in the selected region of interest.

  This block is mainly intended for calculating the average displacement and/or
  the average strain in the region of interest, but other fields can also be
  computed.
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
               debug: bool = False,
               save_images: bool = False,
               img_name: str = "{self._n_loops:6d}_{t-self.t0:.6f}.tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               fields: List[str] = None,
               labels: List[str] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,
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
      fields: A :obj:`list` of :obj:`str` representing the base of fields on
        which the image will be projected during correlation. The possible
        fields are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

        If not given, the default fields are :
        ::

          ["x", "y", "exx", "eyy"]

      labels: A :obj:`list` containing the labels to send to downstream blocks.
        The first label should be time, the second the metadata, and there
        should then be one label per field. If not given, the default labels
        are :
        ::

          ['t(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)']

      alpha: Weight of the smoothness term in DisFlow.
      delta: Weight of the color constancy term in DisFlow.
      gamma: Weight of the gradient constancy term in DisFlow.
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DisFlow (`0` means full scale).
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DisFlow.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DisFlow.
      init: If :obj:`True`, the new optical flow is at each loop initialized
        using the previous optical flow.
      patch_size: Size of an image patch for matching in DisFlow
        (in pixels).
      patch_stride: Stride between neighbor patches in DisFlow. Must be
        less than patch size.
      residual: If :obj:`True`, the residuals are computed and sent under the
        label ``'res'``. This label shouldn't be included if custom labels are
        given.
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
                     debug=debug,
                     save_images=save_images,
                     img_name=img_name,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     **kwargs)

    # Managing the fields and labels lists
    fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    self.labels = ['t(s)', 'meta', 'x(pix)', 'y(pix)',
                   'Exx(%)', 'Eyy(%)'] if labels is None else labels
    if residual:
      self.labels.append('res')

    # Making sure a coherent number of labels and fields was given
    if 2 + len(fields) + int(residual) != len(self.labels):
      raise ValueError(
        "The number of fields is inconsistent with the number "
        "of labels !\nMake sure that the time label was given")

    self._dis = Dis(fields=fields,
                    alpha=alpha,
                    delta=delta,
                    gamma=gamma,
                    finest_scale=finest_scale,
                    init=init,
                    iterations=iterations,
                    gradient_iterations=gradient_iterations,
                    patch_size=patch_size,
                    patch_stride=patch_stride)

    self._residual = residual
    self._config_dis = config
    self._bbox = Box()

    # Instantiating the displayer
    if display_images:
      self._displayer_dis = Displayer(f"Displayer {camera} "
                                      f"{Camera.cam_count[self._camera_name]}",
                                      displayer_framerate,
                                      displayer_backend)
    else:
      self._displayer_dis = None

  def prepare(self) -> None:
    """Opening the camera, displaying the config window and setting the region
    of interest."""

    super().prepare()

    if self._config_dis:
      conf = DiscorrelConfig(self._camera)
      conf.main()
      self._bbox = conf.box

    if self._bbox.no_points():
      raise AttributeError("The region of interest wasn't properly selected in"
                           " the config window !")

    self._dis.set_box(self._bbox)

    if self._displayer_dis is not None:
      self._displayer_dis.prepare()

  def begin(self) -> None:
    """Capturing a first image and setting it as a reference for the
    correlation."""

    _, img = self._camera.get_image()
    self._dis.set_img0(img)

  def finish(self) -> None:
    """Closing the displayer and the camera."""

    if self._displayer_dis is not None:
      self._displayer_dis.finish()

    super().finish()

  def _additional_loop(self, meta: Dict[str, Any], img: np.ndarray) -> None:
    """Getting the updated values from the correlation and updating the
    displayer."""

    data = self._dis.get_data(img, self._residual)
    self.send([meta['t(s)'], meta] + data)

    if self._displayer_dis is not None:
      self._draw_box(img, self._bbox)
      self._displayer_dis.update(img)
