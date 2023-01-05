# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path
from .gpuve_parallel_process import Gpuve_parallel_process
from .camera_parallel import Camera_parallel
from ..tool import GPUCorrel
from .._global import OptionalModule

try:
  import pycuda.tools
  import pycuda.driver
except (ModuleNotFoundError, ImportError):
  pycuda = OptionalModule("pycuda")


class Gpuve_parallel(Camera_parallel):
  """"""

  def __init__(self,
               camera: str,
               patches: List[Tuple[int, int, int, int]],
               img_shape: Tuple[int, int],
               img_dtype: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: bool = False,
               freq: float = 200,
               save_images: bool = False,
               img_extension: str = "tiff",
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
    """"""

    super().__init__(camera=camera,
                     transform=transform,
                     config=False,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     verbose=verbose,
                     freq=freq,
                     save_images=save_images,
                     img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     img_shape=img_shape,
                     img_dtype=img_dtype,
                     **kwargs)

    # Setting the labels
    if labels is None:
      self.labels = ['t(s)', 'meta'] + [elt
                                        for i, _ in enumerate(patches)
                                        for elt in [f'p{i}x', f'p{i}y']]
    else:
      self.labels = labels

    self._patches = patches
    self._img_ref = img_ref

    if 2 + 2 * len(patches) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time and metadata "
                       "labels were given")

    self._gpuve_kw = dict(verbose=verbose,
                          levels=1,
                          resampling_factor=2,
                          kernel_file=kernel_file,
                          iterations=iterations,
                          mask=None,
                          ref_img=img_ref,
                          mul=mul,
                          fields=['x', 'y'])

  def prepare(self) -> None:
    """"""

    pycuda.driver.init()
    context = pycuda.tools.make_default_context()
    self._gpuve_kw.update(dict(context=context))

    self._correls = [GPUCorrel(**self._gpuve_kw) for _ in self._patches]

    # We can already set the sizes of the images as they are already known
    for correl, (_, __, h, w) in zip(self._correls, self._patches):
      correl.set_img_size((h, w))

    if self._img_ref is not None:
      for correl, (oy, ox, h, w) in zip(self._correls, self._patches):
        correl.set_orig(self._img_ref[oy:oy + h, ox:ox + w].astype(np.float32))
        correl.prepare()

    self._process_proc = Gpuve_parallel_process(
      correls=self._correls, patches=self._patches,
      img0_set=self._img_ref is not None)

    super().prepare()

  def _configure(self) -> None:
    """"""

    ...
