# coding: utf-8

from typing import Optional, Callable, Union, Tuple, Iterable
import numpy as np
from pathlib import Path

from .camera_processes import GPUVEProcess
from .camera import Camera


class GPUVE(Camera):
  """"""

  def __init__(self,
               camera: str,
               patches: Iterable[Tuple[int, int, int, int]],
               img_shape: Tuple[int, int],
               img_dtype: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               verbose: bool = False,
               freq: Optional[float] = 200,
               debug: Optional[bool] = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               labels: Optional[Union[str, Iterable[str]]] = None,
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
                     display_freq=bool(verbose),
                     disp=freq,
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

    # Forcing the patches into a list
    patches = list(patches)

    # Forcing the labels into a list
    if labels is None:
      self.labels = ['t(s)', 'meta'] + [elt
                                        for i, _ in enumerate(patches)
                                        for elt in [f'p{i}x', f'p{i}y']]
    elif isinstance(labels, str):
      self.labels = [labels]
    else:
      self.labels = list(labels)

    self._img_ref = img_ref

    if 2 + 2 * len(patches) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time and metadata "
                       "labels were given")

    self._gpu_ve_kw = dict(patches=patches,
                           verbose=verbose,
                           kernel_file=kernel_file,
                           iterations=iterations,
                           img_ref=img_ref,
                           mul=mul)

  def prepare(self) -> None:
    """"""

    self._process_proc = GPUVEProcess(log_queue=self._log_queue,
                                      log_level=self._log_level,
                                      **self._gpu_ve_kw)

    super().prepare()

  def _configure(self) -> None:
    """"""

    ...
