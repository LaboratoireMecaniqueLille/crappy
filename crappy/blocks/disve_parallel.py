# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path
from .disve_parallel_process import Disve_parallel_process
from .camera_parallel import Camera_parallel
from ..tool import DISVE_config, Spot_boxes


class Disve_parallel(Camera_parallel):
  """"""

  def __init__(self,
               camera: str,
               patches: List[Tuple[int, int, int, int]],
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: bool = False,
               freq: float = 200,
               debug: bool = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[Tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
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
               raise_on_patch_exit: bool = True,
               **kwargs) -> None:
    """"""

    super().__init__(camera=camera,
                     transform=transform,
                     config=config,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     verbose=verbose,
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

    # Setting the labels
    if labels is None:
      self.labels = ['t(s)', 'meta'] + [elt
                                        for i, _ in enumerate(patches)
                                        for elt in [f'p{i}x', f'p{i}y']]
    else:
      self.labels = labels

    self._raise_on_exit = raise_on_patch_exit
    self._patches_int = patches

    self._disve_kw = dict(method=method,
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
                          follow=follow,
                          raise_on_exit=raise_on_patch_exit)

  def prepare(self) -> None:
    """"""

    self._patches = Spot_boxes()
    self._patches.set_spots(self._patches_int)
    self._disve_kw['patches'] = self._patches

    self._process_proc = Disve_parallel_process(log_queue=self._log_queue,
                                                log_level=self.log_level,
                                                parent_name=self.name,
                                                **self._disve_kw)

    super().prepare()

  def _configure(self) -> None:
    """"""

    config = DISVE_config(self._camera, self._patches)
    config.main()
    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = config.dtype
