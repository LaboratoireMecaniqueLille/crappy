# coding: utf-8

import numpy as np
from typing import Callable, Optional, List, Union, Tuple
from pathlib import Path

from .._global import OptionalModule
from ..tool import GPUCorrel as GPUCorrel_tool
from .camera import Camera

try:
  import pycuda.tools
  import pycuda.driver
except (ModuleNotFoundError, ImportError):
  pycuda = OptionalModule("pycuda")


class GPUVE(Camera):
  """This block tracks patches on an image using GPU-accelerated Dense Inverse
  Search.

  The patches must be given explicitly as arguments, they cannot be selected in
  a GUI. The block sends the updated positions of the tracked patches to the
  downstream blocks, as well as the timestamp. It is roughly equivalent to the
  :ref:`Disve` block, but with GPU acceleration.
  """

  def __init__(self,
               camera: str,
               patches: List[Tuple[int, int, int, int]],
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               verbose: int = 0,
               freq: float = 200,
               save_images: bool = False,
               img_name: str = "{self._n_loops:6d}_{t-self.t0:.6f}.tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               labels: Optional[List[str]] = None,
               img_ref: Optional[np.ndarray] = None,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               mul: float = 3,
               **kwargs) -> None:
    """Sets the args and initializes the camera.

    Args:
      camera: The name of the camera to control. See :ref:`Cameras` for an
        exhaustive list of available cameras.
      patches: A :obj:`list` containing the patches to track given as
        :obj:`tuple`. Each patch should contain in that order: the `y` coord of
        its origin, the `x` coord of its origin, its size along the `y` axis
        and its size along the `x` axis. Any number of patches can be given.
      transform: A function taking an image as an argument and returning a
        transformed image. The original image is discarded and only the
        transformed one is kept for processing, display and saving.
      verbose: The verbose level as an integer, between `0` and `3`. At level
        `0` no information is printed, and at level `3` so much information is
        printed that is slows the code down.
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
        carrying the information on the latest position of the patches. If
        not given, the default labels are :
        ::

          ['t(s)', 'p0x', 'p0y', ..., pix, piy]

        with `i` the number of given patches.
      img_ref: The reference image to which all the acquired images will be
        compared for performing the correlation. If not given, the first
        acquired images will be used as the reference image.
      kernel_file: The path to a file containing the kernel modules for
        :mod:`pycuda` to use. If not given ,the default kernels of Crappy are
        used.
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing.
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk that to go past the solution and to diverge. If it's
        too low, the convergence will be slower and require more iterations.
        `3` was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast.
      **kwargs: Any additional argument to pass to the camera.
    """

    # Setting now the pycuda context to avoid setting it multiple times
    pycuda.driver.init()
    context = pycuda.tools.make_default_context()

    self._patches = patches
    self._img_ref = img_ref

    super().__init__(camera=camera,
                     transform=transform,
                     config=False,
                     display_images=False,
                     displayer_backend=None,
                     displayer_framerate=5,
                     software_trig_label=None,
                     verbose=bool(verbose),
                     freq=freq,
                     save_images=save_images,
                     img_name=img_name,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     **kwargs)

    self._correls = [GPUCorrel_tool(context=context,
                                    verbose=verbose,
                                    levels=1,
                                    resampling_factor=2,
                                    kernel_file=kernel_file,
                                    iterations=iterations,
                                    mask=None,
                                    ref_img=img_ref,
                                    mul=mul,
                                    fields=['x', 'y']) for _ in self._patches]

    # Setting the labels
    if labels is None:
      self.labels = ['t(s)'] + [elt
                                for i, _ in enumerate(patches)
                                for elt in [f'p{i}x', f'p{i}y']]
    else:
      self.labels = labels

    if 1 + 2 * len(patches) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time label was given")

    # We can already set the sizes of the images as they are already known
    for correl, (_, __, h, w) in zip(self._correls, self._patches):
      correl.set_img_size((h, w))

  def prepare(self) -> None:
    """Opens the camera and sets the reference sub-image for each patch if a
    reference image was given."""

    super().prepare()

    if self._img_ref is not None:
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(self._img_ref[oy:oy + h, ox:ox + w].astype(np.float32))
        correl.prepare()

  def begin(self) -> None:
    """Acquires a first image and sets it as the reference image if no
    reference image was previously given."""

    if self._img_ref is None:
      _, img = self._camera.get_image()

      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(img[oy:oy + h, ox:ox + w].astype(np.float32))
        correl.prepare()

  def finish(self) -> None:
    """Closes the correlation objects as well as the camera."""

    for correl in self._correls:
      correl.clean()

    super().finish()

  def _additional_loop(self, t: float, img: np.ndarray) -> None:
    """Gets the updated positions of the patches, and sends it to the
    downstream blocks."""

    out = [t - self.t0]
    for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
      out.extend(correl.get_disp(img[oy:oy + h,
                                 ox:ox + w].astype(np.float32)).tolist())

    self.send(out)
