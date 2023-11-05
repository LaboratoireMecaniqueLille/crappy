# coding: utf-8

import numpy as np
from typing import Callable, Union, Optional, List, Dict, Any
from pathlib import Path
from warnings import warn

from ..tool import GPUCorrel as GPUCorrel_tool
from .camera import Camera
from .displayer import Displayer


class GPUCorrel(Camera):
  """This blocks projects a displacement field on a given base of fields, and
  sends the decomposition to downstream blocks.

  It relies on the :ref:`GPU Correl` class. The displacement is calculated for
  the entire image, and it is not possible to select a region of interest.
  """

  def __init__(self,
               camera: str,
               fields: List[str],
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
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
               discard_limit: float = 3,
               discard_ref: int = 5,
               img_ref: Optional[np.ndarray] = None,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               mask: Optional[np.ndarray] = None,
               mul: float = 3,
               res: bool = False,
               **kwargs) -> None:
    """Sets the args and initializes the camera object.

    Args:
      camera: The name of the camera to control. See :ref:`Cameras` for an
        exhaustive list of available cameras.
      fields: A :obj:`list` of :obj:`str` representing the base of fields on
        which the image will be projected during correlation. The possible
        fields are :
        ::

          'x', 'y', 'r', 'exx', 'eyy', 'exy', 'eyx', 'exy2', 'z'

      transform: A function taking an image as an argument and returning a
        transformed image. The original image is discarded and only the
        transformed one is kept for processing, display and saving.
      display_images: If :obj:`True`, the difference between the original and
        the displaced image after correlation will be displayed. `128` means
        no difference, lighter means positive and darker negative.
      displayer_backend: If ``display_images`` is :obj:`True`, the backend to
        use for the display window. Should be one of :
        ::

          'cv2', 'mpl'

        If not given, OpenCV will be used if available.
      displayer_framerate: If ``display_images`` is :obj:`True`, sets the
        maximum framerate for updating the display window. This setting allows
        limiting the resources used by the displayer. Note that the actual
        achieved framerate might differ, this is just the maximum limit.
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
        carrying the displacement projected on the given basis of fields. If
        not given, the labels list is just a copy of the fields list, with
        ``'t(s)'`` added in position 0, ``'meta'`` in position 1, and ``'res'``
        added in last position if the ``res`` argument is :obj:`True`.
      discard_limit: If given, the data is sent to downstream blocks only if
        the residuals are lower than the average of the last few residuals
        multiplied by this value.
      discard_ref: When checking whether the data should be sent based on the
        ``discard_limit`` argument, only that many previous values will be
        considered when calculating the average of the last residuals.
      img_ref: The reference image to which all the acquired images will be
        compared for performing the correlation. If not given, the first
        acquired images will be used as the reference image.
      levels: Number of levels of the pyramid. More levels may help converging
        on images with large strain, but may fail on images that don't contain
        low spatial frequency. Fewer levels mean that the program runs faster.
      resampling_factor: the factor by which the resolution is divided between
        each stage of the pyramid. A low factor ensures coherence between the
        stages, but is more computationally intensive. A high factor allows
        reaching a finer detail level, but may lead to a coherence loss between
        the stages.
      kernel_file: The path to the file containing the kernels to use for the
        correlation. Can be a :obj:`pathlib.Path` object or a :obj:`str`.
      iterations: The maximum number of iterations to run before returning the
        results. The results may be returned before if the residuals start
        increasing.
      mask: The mask used for weighting the region of interest on the image. It
        is generally used to prevent unexpected behavior on the border of the
        image.
      mul: The scalar by which the direction will be multiplied before being
        added to the solution. If it's too high, the convergence will be fast
        but there's a risk that to go past the solution and to diverge. If it's
        too low, the convergence will be slower and require more iterations.
        `3` was found to be an acceptable value in most cases, but it is
        recommended to tune this value for each application so that the
        convergence is neither too slow nor too fast.
      res: If :obj:`True`, the residuals will be sent to downstream blocks
        along with the other information under the label ``'res'``.
      **kwargs: Any additional argument to pass to the camera.
    """
    
    if img_name:
      warn("The img_name argument will be replaced by img_extension in "
           "version 2.0.0", FutureWarning)
    if verbose:
      warn("The verbose argument will be replaced by display_freq and debug "
           "in version 2.0.0", FutureWarning)

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

    self._correl = GPUCorrel_tool(context=None,
                                  verbose=verbose,
                                  levels=levels,
                                  resampling_factor=resampling_factor,
                                  kernel_file=kernel_file,
                                  iterations=iterations,
                                  mask=mask,
                                  ref_img=img_ref,
                                  mul=mul,
                                  fields=fields)

    # Setting the labels
    if labels is None:
      self.labels = ['t(s)', 'meta'] + fields
    else:
      self.labels = labels

    if res:
      self.labels.append('res')
    self._calc_res = res

    if 2 + len(fields) + int(res) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time label was given")

    # Setting the args
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._img_ref = img_ref
    self._res_hist = [np.inf]

    # Initializing the displayer
    if display_images:
      self._displayer_gpu = Displayer(f"Displayer {camera} "
                                      f"{Camera.cam_count[self._camera_name]}",
                                      displayer_framerate,
                                      displayer_backend)
    else:
      self._displayer_gpu = None

  def prepare(self) -> None:
    """Opens the camera, prepares the displayer and sets the reference image if
    one was given."""

    super().prepare()

    if self._img_ref is not None:
      self._correl.set_img_size(self._img_ref.shape)
      self._correl.prepare()

    if self._displayer_gpu is not None:
      self._displayer_gpu.prepare()

  def begin(self) -> None:
    """Acquires an image and sets it as the reference image if no image was
    given as reference in the arguments."""

    if self._img_ref is None:
      _, img = self._camera.get_image()

      self._correl.set_img_size(img.shape)
      self._correl.set_orig(img.astype(np.float32))
      self._correl.prepare()

  def finish(self) -> None:
    """Closes the displayer and the camera object."""

    if self._displayer_gpu is not None:
      self._displayer_gpu.finish()

    self._correl.clean()
    super().finish()

  def _additional_loop(self, meta: Dict[str, Any], img: np.ndarray) -> None:
    """Gets the updated fields of displacement, and sends them to downstream
    blocks.

    Optionally, also checks if the residuals are low enough for safely sending
    the data.
    """

    warn("The _additional_loop method will be removed in version 2.0.0",
         DeprecationWarning)

    out = [meta['t(s)'], meta]
    out += self._correl.get_disp(img.astype(np.float32)).tolist()

    if self._calc_res:
      res = self._correl.get_res()
      out.append(res)

      if self._discard_limit:
        self._res_hist.append(res)
        self._res_hist = self._res_hist[-self._discard_ref - 1:]

        if res > self._discard_limit * np.average(self._res_hist[:-1]):
          print("[GPUCorrel] Residual too high, not sending values")
          return

    self.send(out)
