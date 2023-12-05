# coding: utf-8

from dataclasses import dataclass


@dataclass
class Zoom:
  """This class stores the upper and lower limits of the image to display in
  the :class:`~crappy.tool.camera_config.CameraConfig` window.

  It also allows updating them when the user changes the zoom ratio or drags
  the image with the mouse.

  .. versionadded:: 2.0.0
  """

  x_low: float = 0.
  x_high: float = 1.
  y_low: float = 0.
  y_high: float = 1.

  def reset(self) -> None:
    """Resets the zoom level to default (no zoom)."""

    self.x_low, self.x_high, self.y_low, self.y_high = 0, 1, 0, 1

  def update_zoom(self, x: float, y: float, ratio: float) -> None:
    """Updates the upper and lower limits of the image when the user scrolls
    with the mousewheel.

    The update is based on the zoom ratio and the position of the mouse on the
    screen.

    Args:
      x: The `x` position of the mouse on the image, as a ratio between `0`
        and `1`.
      y: The `y` position of the mouse on the image, as a ratio between `0`
        and `1`.
      ratio: The zoom ratio to apply. If it is greater than `1` we zoom in,
        otherwise we zoom out.
    """

    prev_x_low, prev_x_high = self.x_low, self.x_high
    prev_y_low, prev_y_high = self.y_low, self.y_high

    # Updating the lower x limit
    self.x_low = max(self.x_low + x * (1 - 1 / ratio), 0.)
    # Updating the upper x limit, making sure it's not out of the image
    if self.x_low + 1 / ratio * (prev_x_high - prev_x_low) > 1.:
      self.x_high = 1.
      self.x_low = 1 - 1 / ratio * (prev_x_high - prev_x_low)
    else:
      self.x_high = self.x_low + 1 / ratio * (prev_x_high - prev_x_low)

    # Updating the lower y limit
    self.y_low = max(self.y_low + y * (1 - 1 / ratio), 0.)
    # Updating the upper y limit, making sure it's not out of the image
    if self.y_low + 1 / ratio * (prev_y_high - prev_y_low) > 1.:
      self.y_high = 1.
      self.y_low = 1 - 1 / ratio * (prev_y_high - prev_y_low)
    else:
      self.y_high = self.y_low + 1 / ratio * (prev_y_high - prev_y_low)

  def update_move(self, delta_x: float, delta_y: float) -> None:
    """Updates the upper and lower limits of the image when the user moves the
    image with a left button click.

    Args:
      delta_x: The `x` displacement to apply to the image, as a ratio of the
        total image width.
      delta_y: The `y` displacement to apply to the image, as a ratio of the
        total image height.
    """

    prev_x_low, prev_x_high = self.x_low, self.x_high
    prev_y_low, prev_y_high = self.y_low, self.y_high

    # Updating the x position
    if delta_x <= 0:
      self.x_low = max(0., prev_x_low + delta_x)
      self.x_high = self.x_low + prev_x_high - prev_x_low
    else:
      self.x_high = min(1., prev_x_high + delta_x)
      self.x_low = self.x_high - prev_x_high + prev_x_low

    # Updating the y position
    if delta_y <= 0:
      self.y_low = max(0., prev_y_low + delta_y)
      self.y_high = self.y_low + prev_y_high - prev_y_low
    else:
      self.y_high = min(1., prev_y_high + delta_y)
      self.y_low = self.y_high - prev_y_high + prev_y_low
