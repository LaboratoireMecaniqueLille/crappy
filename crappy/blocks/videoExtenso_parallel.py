# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path
from .ve_parallel_process import Ve_parallel_process
from .camera_parallel import Camera_parallel
from ..tool import VE_config
from ..tool.videoextenso import VideoExtenso


class Video_extenso_parallel(Camera_parallel):
  """"""

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
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
               img_shape: Optional[Tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
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
                     save_images=save_images,
                     img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     img_shape=img_shape,
                     img_dtype=img_dtype,
                     **kwargs)

    self.labels = ['t(s)', 'meta', 'Coord(px)',
                   'Eyy(%)', 'Exx(%)'] if labels is None else labels

    self._raise_on_lost_spot = raise_on_lost_spot
    self._ve_kw = dict(white_spots=white_spots,
                       update_thresh=update_thresh,
                       num_spots=num_spots,
                       safe_mode=safe_mode,
                       border=border,
                       min_area=min_area,
                       blur=blur)

  def prepare(self) -> None:
    """"""

    self._ve = VideoExtenso(**self._ve_kw)
    self._process_proc = Ve_parallel_process(self._ve,
                                             self._raise_on_lost_spot)

    super().prepare()

  def _configure(self) -> None:
    """"""

    config = VE_config(self._camera, self._ve)
    config.main()
    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = config.dtype