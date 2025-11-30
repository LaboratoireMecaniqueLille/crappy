# coding: utf-8

from collections.abc import Callable

from .camera_setting import CameraSetting


class CameraBoolSetting(CameraSetting):
  """Camera setting that can only be :obj:`True` or :obj:`False`.

  It is a child of
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Camera_bool_setting* to *CameraBoolSetting*
  """

  def __init__(self,
               name: str,
               getter: Callable[[], bool] | None = None,
               setter: Callable[[bool], None] | None = None,
               default: bool = True) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    super().__init__(name, getter, setter, default)
