# coding: utf-8

from typing import List, Tuple, Optional, Callable, Union, Dict, Any
import numpy as np
from pathlib import Path
from ..tool import DISVE as VE, DISVE_config, Spot_boxes
from .camera import Camera
from .displayer import Displayer


class DISVE(Camera):
  """This block tracks the motion of regions of an image (patches), taking the
  first image as a reference.

  It relies on cross-correlation, and is thus well-suited for tracking speckled
  patches. The displacement output may then be used to compute strains, and so
  to perform video-extensometry.

  The images may be acquired by a camera, or be sent from another block.
  Several algorithms are available for tracking the patches, with different
  characteristics. All the computations are done by the :ref:`Disve tool`.
  """

  def __init__(self,
               camera: str,
               patches: List[Tuple[int, int, int, int]],
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
               labels: Optional[List[str]] = None,
               method: str = 'Disflow',
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               border: float = 0.2,
               safe: bool = True,
               follow: bool = True,
               **kwargs) -> None:
    """Sets a few attributes.

    Args:
      camera: The camera to use for acquiring the images. It should be one of
        the :ref:`Supported cameras`.
      patches: A list containing the different patches to track. Each patch
        should be given as follows : ``(y min, x_min, height, width)``.
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
      labels: The labels associated with the timestamp and the displacement of
        the patches. If not given, the time label is ``t(s)``, the metadata
        label is ``'meta'``, the first patch displacement labels are ``p0x``
        and ``p0y``, the second ``p1x`` and ``p1y``, etc.
      method: The method to use to calculate the displacement. `Disflow` uses
        opencv's DISOpticalFlow and `Lucas Kanade` uses opencv's
        calcOpticalFlowPyrLK, while all other methods are based on a basic
        cross-correlation in the Fourier domain. `Pixel precision` calculates
        the displacement by getting the position of the maximum of the
        cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. `Parabola` refines the result of
        `Pixel precision` by interpolating the neighborhood of the maximum, and
        have thus sub-pixel resolutions.
      alpha: Weight of the smoothness term in DisFlow.
      delta: Weight of the color constancy term in DisFlow.
      gamma: Weight of the gradient constancy term in DisFlow.
      finest_scale: Finest level of the Gaussian pyramid on which the flow
        is computed in DisFlow (`0` means full scale).
      iterations: Maximum number of gradient descent iterations in the
        patch inverse search stage in DisFlow.
      gradient_iterations: Maximum number of gradient descent iterations
        in the patch inverse search stage in DisFlow.
      patch_size: Size of an image patch for matching in DisFlow
        (in pixels).
      patch_stride: Stride between neighbor patches in DisFlow. Must be
        less than patch size.
      border: Crop the patch on each side according to this value before
        calculating the displacements. 0 means no cropping, 1 means the entire
        patch is cropped.
      safe: If :obj:`True`, checks whether the patches aren't exiting the
        image, and raises an error if that's the case.
      follow: It :obj:`True`, the patches will move to follow the displacement
        of the image.
      **kwargs: Any additional argument to pass to the camera.
    """

    self._patches = Spot_boxes()
    self._patches.set_spots(patches)

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

    self._ve = VE(patches=self._patches,
                  method=method,
                  alpha=alpha,
                  delta=delta,
                  gamma=gamma,
                  finest_scale=finest_scale,
                  iterations=iterations,
                  gradient_iterations=gradient_iterations,
                  patch_size=patch_size,
                  patch_stride=patch_stride,
                  border=border,
                  safe=safe,
                  follow=follow)

    self._config_dis = config

    # Setting the labels
    if labels is None:
      self.labels = ['t(s)', 'meta'] + [elt
                                        for i, _ in enumerate(patches)
                                        for elt in [f'p{i}x', f'p{i}y']]
    else:
      self.labels = labels

    # Initializing the displayer
    if display_images:
      self._displayer_dis = Displayer(f"Displayer {camera} "
                                      f"{Camera.cam_count[self._camera_name]}",
                                      displayer_framerate,
                                      displayer_backend)
    else:
      self._displayer_dis = None

  def prepare(self) -> None:
    """Opens the camera for acquiring images and displays the corresponding
    settings window."""

    super().prepare()

    if self._config_dis:
      config = DISVE_config(self._camera, self._patches)
      config.main()

    if self._displayer_dis is not None:
      self._displayer_dis.prepare()

  def begin(self) -> None:
    """Takes a first image from the camera and uses it to initialize the Disve
    tool."""

    _, img = self._camera.get_image()
    self._ve.set_img0(img)

  def finish(self) -> None:
    """Closes the Disve tool and the camera acquiring the images."""

    if self._displayer_dis is not None:
      self._displayer_dis.finish()

    super().finish()

  def _additional_loop(self, meta: Dict[str, Any], img: np.ndarray) -> None:
    """Simply calculates the displacement and sends it to downstream blocks."""

    ret = self._ve.calculate_displacement(img)
    self.send([meta['t(s)'], meta] + ret)

    if self._displayer_dis is not None:
      for patch in self._patches:
        self._draw_box(img, patch)

      self._displayer_dis.update(img)
