# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path
from .camera_processes import GpuCorrelProcess
from .camera_parallel import Camera_parallel


class Gpucorrel_parallel(Camera_parallel):
  """"""

  def __init__(self,
               camera: str,
               fields: List[str],
               img_shape: Tuple[int, int],
               img_dtype: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
      self.labels = ['t(s)', 'meta'] + fields
    else:
      self.labels = labels

    if res:
      self.labels.append('res')
    self._calc_res = res

    if 2 + len(fields) + int(res) != len(self.labels):
      raise ValueError("The number of fields is inconsistent with the number "
                       "of labels !\nMake sure that the time label was given")

    self._gpucorrel_kw = dict(discard_limit=discard_limit,
                              discard_ref=discard_ref,
                              calc_res=res,
                              img_ref=img_ref,
                              verbose=verbose,
                              levels=levels,
                              resampling_factor=resampling_factor,
                              kernel_file=kernel_file,
                              iterations=iterations,
                              fields=fields,
                              mask=mask,
                              mul=mul)

  def prepare(self) -> None:
    """"""

    self._process_proc = GpuCorrelProcess(log_queue=self._log_queue,
                                          log_level=self.log_level,
                                          **self._gpucorrel_kw)

    super().prepare()

  def _configure(self) -> None:
    """"""

    ...
