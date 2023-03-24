# coding: utf-8

from typing import Optional, Callable, List, Union, Tuple
import numpy as np
from pathlib import Path

from .camera_processes import DISCorrelProcess
from .camera import Camera
from ..tool.camera_config import DISCorrelConfig, Box
from .._global import CameraConfigError


class DISCorrel(Camera):
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
               freq: Optional[float] = 200,
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
               patch: Optional[Tuple[int, int, int, int]] = None,
               fields: List[str] = None,
               labels: List[str] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,
               **kwargs) -> None:
    """"""

    if not config and patch is None:
      raise ValueError("If the config window is disabled, the patch must be "
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

    # Managing the fields and labels lists
    fields = ["x", "y", "exx", "eyy"] if fields is None else fields
    self.labels = ['t(s)', 'meta', 'x(pix)', 'y(pix)',
                   'Exx(%)', 'Eyy(%)'] if labels is None else labels
    if residual and labels is None:
      self.labels.append('res')

    self._patch_int = patch
    self._patch: Optional[Box] = None

    # Making sure a coherent number of labels and fields was given
    if 2 + len(fields) + int(residual) != len(self.labels):
      raise ValueError(
        "The number of fields is inconsistent with the number "
        "of labels !\nMake sure that the time label was given")

    self._dis_correl_kw = dict(fields=fields,
                               alpha=alpha,
                               delta=delta,
                               gamma=gamma,
                               finest_scale=finest_scale,
                               init=init,
                               iterations=iterations,
                               gradient_iterations=gradient_iterations,
                               patch_size=patch_size,
                               patch_stride=patch_stride,
                               residual=residual)

  def prepare(self) -> None:
    """"""

    if self._patch_int is not None:
      self._patch = Box(x_start=self._patch_int[1],
                        x_end=self._patch_int[1] + self._patch_int[3],
                        y_start=self._patch_int[0],
                        y_end=self._patch_int[0] + self._patch_int[2])
    else:
      self._patch = Box()

    self._process_proc = DISCorrelProcess(log_queue=self._log_queue,
                                          log_level=self._log_level,
                                          display_freq=self.display_freq,
                                          patch=self._patch,
                                          **self._dis_correl_kw)

    super().prepare()

  def _configure(self) -> None:
    """"""

    config = None
    try:
      config = DISCorrelConfig(self._camera, self._log_queue, self._log_level,
                               self._patch)
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
