# coding: utf-8

from typing import Literal, List, Tuple, Optional
from ..tool import DISVE as VE, DISVE_config
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


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
               labels: Optional[List[str]] = None,
               method: Literal['Disflow',
                               'Pixel precision',
                               'Parabola',
                               'Lucas Kanade'] = 'Disflow',
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gditerations: int = 10,
               patch_size: int = 8,
               patch_stride: int = 3,
               show_image: bool = False,
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
      labels: The labels associated with the timestamp and the displacement of
        the patches. If not given, the time label is ``t(s)``, the first patch
        displacement labels are ``p0x`` and ``p0y``, the second ``p1x`` and
        ``p1y``, etc.
      method: The method to use to calculate the displacement. `Disflow` uses
        opencv's DISOpticalFlow and `Lucas Kanade` uses opencv's
        calcOpticalFlowPyrLK, while all other methods are based on a basic
        cross-correlation in the Fourier domain. `Pixel precision` calculates
        the displacement by getting the position of the maximum of the
        cross-correlation, and has thus a 1-pixel resolution. It is mainly
        meant for debugging. `Parabola` refines the result of
        `Pixel precision` by interpolating the neighborhood of the maximum, and
        have thus sub-pixel resolutions.
      alpha: Setting for Disflow.
      delta: Setting for Disflow.
      gamma: Setting for Disflow.
      finest_scale: The last scale for Disflow (`0` means full scale).
      iterations: Variational refinement iterations for Disflow.
      gditerations: Gradient descent iterations for Disflow.
      patch_size: Correlation patch size for Disflow.
      patch_stride: Correlation patch stride for Disflow.
      show_image: If :obj:`True`, displays the real-time position of the
        patches on the image. This feature is mainly meant for debugging.
      border: Crop the patch on each side according to this value before
        calculating the displacements. 0 means no cropping, 1 means the entire
        patch is cropped.
      safe: Checks whether the patches aren't exiting the image, and raises an
        error if that's the case.
      follow: It :obj:`True`, the patches will move to follow the displacement
        of the image.
      **kwargs: Any additional kwarg to pass to the camera.
    """

    super().__init__(camera, **kwargs)
    self.niceness = -5

    self._patches = patches
    if labels is None:
      self.labels = ['t(s)'] + [elt
                                for i, _ in enumerate(self._patches)
                                for elt in [f'p{i}x', f'p{i}y']]
    else:
      self.labels = labels

    self._ve_kwargs = {"method": method,
                       "alpha": alpha,
                       "delta": delta,
                       "gamma": gamma,
                       "finest_scale": finest_scale,
                       "iterations": iterations,
                       "gradient_iterations": gditerations,
                       "patch_size": patch_size,
                       "patch_stride": patch_stride,
                       "border": border,
                       "show_image": show_image,
                       "safe": safe,
                       "follow": follow}

  def prepare(self, *_, **__) -> None:
    """Opens the camera for acquiring images and displays the corresponding
    settings window."""

    config = self.config
    self.config = False
    super().prepare(send_img=False)

    if config:
      DISVE_config(self.camera, self._patches).main()

  def begin(self) -> None:
    """Takes a first image from the camera and uses it to initialize the Disve
    tool."""

    _, img = self.camera.read_image()
    self._ve = VE(img0=img, patches=self._patches, **self._ve_kwargs)

  def loop(self) -> None:
    """Acquires an image, uses the Disve tool to calculate the displacement
    of the patches, and sends the timestamp as well as the patches
    positions."""

    t, img = self.get_img()

    if self.inputs and not self.input_label and self.inputs[0].poll():
      self.inputs[0].clear()
      self._ve = VE(img, self._patches, **self._ve_kwargs)
      print("[DISVE block] : Resetting L0")

    ret = self._ve.calculate_displacement(img)
    self.send([t - self.t0] + ret)

  def finish(self) -> None:
    """Closes the Disve tool and the camera acquiring the images."""

    self._ve.close()
    super().finish()
