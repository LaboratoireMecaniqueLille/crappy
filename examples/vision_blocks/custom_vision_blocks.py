# coding: utf-8

"""
This example demonstrates how to implement custom source and consumer
VisionBlocks. It requires no hardware or optional Python module and terminates
automatically after a few seconds.

MovingBarSource is a custom image source. It declares the shape and dtype of
its outgoing images when initializing VisionBlock, creates a simple moving-bar
pattern, associates every frame with the mandatory ``'t(s)'`` and
``'ImageUniqueID'`` metadata, and publishes both with ``send_img``. A source
owns the shared-memory image buffer used by all its downstream ImageLinks.

ImageStatistics is a custom consumer. It calls ``receive_imgs`` on every loop,
uses the returned ImageLink name to access the copied image and metadata in
``last_received``, calculates the mean pixel value and brightest column, and
sends these small results through an ordinary Link. LinkReader prints the
results in the console. The ImageLink is explicitly named to show how names map
to ``last_received`` entries and make larger graphs easier to diagnose.

Both classes validate the ImageLink topology they support before calling
``super().prepare()``. That parent call is essential: it creates or attaches to
the shared image buffer after the graph-wide synchronization objects have been
distributed. ImageStatistics also overrides ``finish`` to report how many
frames it processed and then calls ``super().finish()`` to close its attached
shared-memory handle. MovingBarSource inherits VisionBlock's ``finish`` method,
which closes and unlinks the buffer it owns.

ImageLinks expose the latest frame rather than a queue. If ImageStatistics were
made slower than MovingBarSource, it could skip intermediate frames, but every
copied image would still match its metadata. Run this script and watch the
brightest-column value move across the image in the console. MovingBarSource
stops the complete Crappy test after publishing thirty frames.
"""

import logging
from time import time

import numpy as np

import crappy


class MovingBarSource(crappy.blocks.vision.VisionBlock):
  """Generate a finite stream containing a bright bar moving horizontally."""

  def __init__(self,
               frame_count: int = 30,
               freq: float = 10) -> None:
    """Declare the output format and store the sequence settings.

    Args:
      frame_count: Number of frames to publish before stopping the test.
      freq: Target publication frequency, in frames per second.
    """

    # A source has output ImageLinks, so its image format must be known before
    # VisionBlock.prepare allocates the shared-memory array.
    super().__init__(img_shape=(120, 160),
                     img_dtype='uint8',
                     freq=freq,
                     debug=False)

    self.name = 'Moving bar source'
    self._frame_count = frame_count
    self._frame_index = 0
    self._template: np.ndarray | None = None

  def prepare(self) -> None:
    """Validate the source topology and create its image template."""

    # This source generates images itself and therefore accepts no ImageLink or
    # regular Link input. Without an output ImageLink, generating frames would
    # be useless and VisionBlock would not allocate an output buffer.
    if self.img_inputs:
      raise IOError("MovingBarSource does not accept input ImageLinks")
    if self.inputs:
      raise IOError("MovingBarSource does not accept regular input Links")
    if not self.img_outputs:
      raise IOError("MovingBarSource requires an output ImageLink")

    # The template contains one ten-pixel-wide white bar on a black background.
    # np.roll will move it while preserving the prepared shape and dtype.
    self._template = np.zeros((120, 160), dtype=np.uint8)
    self._template[:, :10] = 255

    # This mandatory call creates the shared output buffer and exposes it to
    # every consumer connected before crappy.start().
    super().prepare()

  def loop(self) -> None:
    """Publish the next image and stop after the finite sequence."""

    if self._template is None:
      raise RuntimeError("The moving-bar template was not prepared")

    # Moving four pixels per frame makes the reported column easy to follow.
    image = np.roll(self._template, 4 * self._frame_index, axis=1)
    metadata = {
        't(s)': time() - self.t0,
        'ImageUniqueID': self._frame_index,
        'pattern': 'moving vertical bar',
    }

    # Both metadata and the NumPy array are copied under one shared lock. Their
    # mandatory timestamp and unique ID let downstream Blocks identify frames.
    self.send_img(metadata, image)
    self._frame_index += 1

    # Block.stop cleanly requests termination of the complete Crappy test.
    if self._frame_index >= self._frame_count:
      self.stop()


class ImageStatistics(crappy.blocks.vision.VisionBlock):
  """Consume one image stream and publish lightweight numeric measurements."""

  def __init__(self, freq: float = 50) -> None:
    """Set the rate at which the custom consumer checks for new frames.

    Args:
      freq: Target loop frequency. It may exceed the source frequency because
        loops without a new image return immediately.
    """

    # A pure consumer has no output ImageLink, so img_shape and img_dtype are
    # unnecessary. They are learned from the upstream shared buffer instead.
    super().__init__(freq=freq, debug=False)

    self.name = 'Image statistics'
    self._processed_count = 0

  def prepare(self) -> None:
    """Require exactly one image input and attach to its shared buffer."""

    if len(self.img_inputs) != 1:
      raise IOError("ImageStatistics requires exactly one input ImageLink")
    if self.img_outputs:
      raise IOError("ImageStatistics does not support output ImageLinks")
    if self.inputs:
      raise IOError("ImageStatistics does not accept regular input Links")

    # This call waits for the source buffer, attaches to it, and prepares a
    # private receive array in self.last_received for the named ImageLink.
    super().prepare()

  def loop(self) -> None:
    """Process the newest frame, if one has been published since last time."""

    # receive_imgs copies each updated source into this process and returns the
    # names of those sources. With exactly one input there is at most one name.
    if not (updated_links := self.receive_imgs()):
      return
    link_name, = updated_links

    received = self.last_received[link_name]
    if received.metadata is None:
      raise RuntimeError("Received image metadata should not be empty")

    # The local image is safe to read without holding the shared-memory lock.
    column_means = np.mean(received.img, axis=0)
    brightest_column = int(np.argmax(column_means))

    # Numeric results belong on a regular Link, not another ImageLink. Sending
    # a dictionary makes a custom labels attribute unnecessary.
    self.send({'t(s)': received.metadata['t(s)'],
               'img_index': received.metadata['ImageUniqueID'],
               'mean_pixel': float(np.mean(received.img)),
               'brightest_column': brightest_column})
    self._processed_count += 1

  def finish(self) -> None:
    """Report the handled-frame count and release the attached image buffer."""

    self.log(logging.INFO, f"Processed {self._processed_count} images")

    # Consumers close their SharedMemory handle in VisionBlock.finish. They do
    # not unlink it, because that responsibility belongs to the source.
    super().finish()


if __name__ == '__main__':

  # Instantiating the two custom VisionBlocks defined above.
  source = MovingBarSource(frame_count=30, freq=10)
  statistics = ImageStatistics(freq=50)

  # LinkReader is an ordinary Block and receives the custom consumer's small
  # result dictionaries through a regular Link.
  reader = crappy.blocks.LinkReader(
      name='Image analysis',
      freq=20,

      # Sticking to defaults for the other arguments
  )

  # Naming the ImageLink makes the corresponding last_received key explicit.
  crappy.img_link(source, statistics, name='moving-bar-images')

  # Only the derived numeric measurements use a regular Crappy Link.
  crappy.link(statistics, reader)

  # The custom source stops this blocking call after publishing thirty frames.
  crappy.start()
