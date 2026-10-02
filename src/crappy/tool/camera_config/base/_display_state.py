# coding: utf-8

"""Backend-independent state and geometry for a camera preview."""

from dataclasses import dataclass

from ..config_tools.zoom import Zoom


@dataclass
class DisplayState:
  """Current camera-preview indicators and user-selected display options.

  Attributes:
    fps: Measured preview refresh rate.
    min_pixel: Minimum value in the latest image.
    max_pixel: Maximum value in the latest image.
    detected_bits: Bits needed to represent the latest image's maximum value.
    zoom_percent: Displayed zoom level, with 100 meaning no zoom.
    reticle_x: Horizontal reticle position in full-image pixels.
    reticle_y: Vertical reticle position in full-image pixels.
    reticle_value: Value of the image pixel under the reticle.
    auto_range: Whether preview contrast is adjusted automatically.
    auto_apply: Whether setting edits are applied without pressing Apply.
  """

  fps: float = 0.0
  min_pixel: int = 0
  max_pixel: int = 0
  detected_bits: int = 0
  zoom_percent: float = 100.0
  reticle_x: int = 0
  reticle_y: int = 0
  reticle_value: int = 0
  auto_range: bool = False
  auto_apply: bool = False


@dataclass
class DisplayGeometry:
  """Display area and centered image dimensions in display coordinates.

  Attributes:
    width: Width of the available display area in pixels.
    height: Height of the available display area in pixels.
    image_width: Width of the image as currently drawn, after fitting.
    image_height: Height of the image as currently drawn, after fitting.
  """

  width: int = 0
  height: int = 0
  image_width: int = 0
  image_height: int = 0

  @property
  def left(self) -> float:
    """Horizontal position of the displayed image's left edge."""

    return (self.width - self.image_width) / 2

  @property
  def top(self) -> float:
    """Vertical position of the displayed image's top edge."""

    return (self.height - self.image_height) / 2

  def contains(self, x: int, y: int) -> bool:
    """Whether a pointer position is on the displayed image."""

    return (self.image_width > 0 and self.image_height > 0 and
            self.left <= x <= self.left + self.image_width and
            self.top <= y <= self.top + self.image_height)

  def relative(self, x: int, y: int) -> tuple[float, float]:
    """Pointer coordinates relative to the displayed image's top left."""

    return x - self.left, y - self.top

  def to_pixel(self,
               x: int,
               y: int,
               image_width: int,
               image_height: int,
               zoom: Zoom) -> tuple[int, int]:
    """Convert a display position to pixel coordinates in the full image."""

    if (self.image_width <= 0 or self.image_height <= 0 or
        image_width <= 0 or image_height <= 0):
      return 0, 0

    relative_x, relative_y = self.relative(x, y)
    x_ratio = relative_x / self.image_width
    y_ratio = relative_y / self.image_height
    x_pixel = int((zoom.x_low + x_ratio * (zoom.x_high - zoom.x_low)) *
                  image_width)
    y_pixel = int((zoom.y_low + y_ratio * (zoom.y_high - zoom.y_low)) *
                  image_height)
    return (min(max(x_pixel, 0), image_width - 1),
            min(max(y_pixel, 0), image_height - 1))

  def fit(self, image_width: int, image_height: int) -> tuple[int, int]:
    """Size an image to fit the display while preserving its aspect ratio."""

    if (self.width <= 0 or self.height <= 0 or
        image_width <= 0 or image_height <= 0):
      return 0, 0

    scale = min(self.width / image_width, self.height / image_height)
    return max(int(image_width * scale), 1), max(int(image_height * scale), 1)
