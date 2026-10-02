# coding: utf-8

"""Backend-independent application of camera configuration settings.

Backends provide requested values and render the effective results.
"""

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from ....camera.meta_camera.camera_setting import CameraSetting


@dataclass(frozen=True)
class SettingApplyResult:
  """Effective value and model changes caused by one requested edit.

  ``changed`` includes settings reloaded indirectly by the edited setting's
  setter, even when the reload changed only bounds, step, or choices.
  """

  effective: Any
  changed: tuple[CameraSetting, ...]


class SettingManager:
  """Apply ordinary requested values without knowing about GUI controls.

  Camera settings retain their insertion order. Local settings are applied
  first, in registration order. A backend reads each requested value just
  before calling :meth:`apply`, and refreshes its controls using ``changed``.
  This prevents an earlier setter's reload from applying a stale later edit.
  """

  def __init__(self,
               camera_settings: Mapping[str, CameraSetting],
               local_settings: tuple[CameraSetting, ...] = ()) -> None:
    """Begin an interactive configuration session for the given settings.

    All registered settings enter interactive reload mode, allowing dependent
    reloads to replace values protected from override during camera setup.

    Args:
      camera_settings: The camera's name-to-setting mapping. It is retained,
        not copied, so settings added later also enter the application order.
        Existing settings keep the mapping's insertion order.
      local_settings: Configurator-specific settings to apply before camera
        settings, in the order given. They are registered without GUI objects,
        later local settings can be added with :meth:`register_local`.
    """

    self._camera_settings: Mapping[str, CameraSetting] = camera_settings
    self._local_settings: list[CameraSetting] = list()

    for setting in camera_settings.values():
      setting.allow_reload_override()
    for setting in local_settings:
      self.register_local(setting)

  @property
  def local_settings(self) -> tuple[CameraSetting, ...]:
    """Settings supplied by the configurator rather than the camera."""

    return tuple(self._local_settings)

  @property
  def settings(self) -> tuple[CameraSetting, ...]:
    """Application order: local settings, then camera insertion order."""

    camera_settings = tuple(self._camera_settings.values())
    local_settings = (setting for setting in self._local_settings
                      if setting not in camera_settings)
    return *local_settings, *camera_settings

  def register_local(self, setting: CameraSetting) -> None:
    """Include a configurator-specific setting in the application order."""

    # A camera setting added after this manager was constructed also enters
    # the interactive reload phase when its backend control is registered
    if setting in self._camera_settings.values():
      setting.allow_reload_override()
      return
    if setting in self._local_settings:
      return
    setting.allow_reload_override()
    self._local_settings.append(setting)

  def apply(self,
            setting: CameraSetting,
            requested: Any) -> SettingApplyResult:
    """Set a requested value and read back its effective value and reloads.

    The caller is responsible for obtaining ``requested`` from its backend
    control and for displaying the effective values afterward.
    """

    settings = self.settings
    if setting not in settings:
      raise ValueError(f"Setting {setting.name} is not registered")

    revisions = {item: item.revision for item in settings}
    if setting.value != requested:
      setting.value = requested
    effective = setting.value
    changed = tuple(item for item in settings
                    if item.revision != revisions[item])

    return SettingApplyResult(effective, changed)
