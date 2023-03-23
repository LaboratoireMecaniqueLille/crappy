# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path

from .camera_processes import DICVEProcess
from .camera import Camera
from ..tool.camera_config import DICVEConfig, SpotsBoxes
from .._global import CameraConfigError


class DICVE(Camera):
  """"""

  def __init__(self,
               camera: str,
               transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               config: bool = True,
               display_images: bool = False,
               displayer_backend: Optional[str] = None,
               displayer_framerate: float = 5,
               software_trig_label: Optional[str] = None,
               display_freq: bool = False,
               freq: float = 200,
               debug: Optional[bool] = False,
               save_images: bool = False,
               img_extension: str = "tiff",
               save_folder: Optional[Union[str, Path]] = None,
               save_period: int = 1,
               save_backend: Optional[str] = None,
               image_generator: Optional[Callable[[float, float],
                                                  np.ndarray]] = None,
               img_shape: Optional[Tuple[int, int]] = None,
               img_dtype: Optional[str] = None,
               patches: Optional[List[Tuple[int, int, int, int]]] = None,
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

    if not config and patches is None:
      raise ValueError("If the config window is disabled, patches must be "
                       "provided !")

    super().__init__(camera=camera,
                     transform=transform,
                     config=config,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     display_freq=display_freq,
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
    self.labels = ['t(s)', 'meta', 'Coord(px)', 'Eyy(%)',
                   'Exx(%)', 'Disp(px)'] if labels is None else labels

    # Making sure a coherent number of labels and fields was given
    if len(self.labels) != 6:
      raise ValueError("The number of labels should be 6 !\n"
                       "Make sure that the time label was given")

    self._patches: Optional[SpotsBoxes] = None

    self._raise_on_exit = raise_on_patch_exit
    self._patches_int = patches

    self._dic_ve_kw = dict(method=method,
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

    self._patches = SpotsBoxes()
    if self._patches_int is not None:
      self._patches.set_spots(self._patches_int)
      self._patches.save_length()
    self._dic_ve_kw['patches'] = self._patches

    self._process_proc = DICVEProcess(log_queue=self._log_queue,
                                      log_level=self._log_level,
                                      display_freq=self.display_freq,
                                      **self._dic_ve_kw)

    super().prepare()

  def _configure(self) -> None:
    """"""

    config = None
    try:
      config = DICVEConfig(self._camera, self._log_queue, self._log_level,
                           self._patches)
      config.main()
    except (Exception,) as exc:
      self._logger.exception("Caught exception in the configuration window !",
                             exc_info=exc)
      if config is not None:
        config.stop()
      raise CameraConfigError

    if config.shape is not None:
      self._img_shape = config.shape
    if config.dtype is not None:
      self._img_dtype = config.dtype
