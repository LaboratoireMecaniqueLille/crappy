.. _concepts-regular-links-image-links:

=============================
Regular Links and ImageLinks
=============================

Crappy provides two kinds of one-way connection. A regular
:class:`~crappy.links.link.Link` carries labeled values, while an
:class:`~crappy.links.img_link.ImageLink` carries images and their metadata.
They have different delivery behavior because a stream of small dictionaries
and a stream of large image arrays have different requirements.

Choosing the Link type
----------------------

.. list-table:: Regular Link and ImageLink comparison
   :header-rows: 1
   :widths: 22 39 39

   * - Question
     - Regular Link
     - ImageLink
   * - What does it carry?
     - A dictionary of labeled commands, measurements, results, or other
       values.
     - One NumPy image array and its matching metadata dictionary.
   * - What can it connect?
     - Any two Blocks.
     - Two :class:`~crappy.blocks.vision.block.VisionBlock` objects.
   * - How are updates retained?
     - Accepted dictionaries remain ordered in a buffered pipe until read.
     - Only the newest published image remains in the shared buffer.
   * - What can a slow consumer miss?
     - A consumer can deliberately discard older values. On Linux, new values
       are also discarded if the pipe is full.
     - Any intermediate image overwritten before the consumer copies it.
   * - How is it created?
     - :func:`crappy.link`
     - :func:`crappy.img_link`

Use a regular Link for commands, measurements, processing results, status
information, and image overlays. Use an ImageLink only for an image array and
the metadata that describes that same image.

How the transports differ
-------------------------

.. graphviz:: ../diagrams/link_and_image_link_transport.dot
   :alt: A regular Link stores ordered dictionaries in a pipe, while an ImageLink exposes one latest image and metadata state that may overwrite an earlier frame before a slow consumer reads it.
   :caption: A regular Link buffers dictionaries in order. An ImageLink reuses one latest-frame buffer, allowing independent consumers to skip intermediate frames without delaying the producer.

For the regular Link shown above, dictionaries accepted into the pipe retain
their order. The consumer chooses whether to receive the oldest dictionary,
the newest dictionary while discarding older unread ones, or all currently
available dictionaries. The pipe has finite capacity and is not a permanent
record of the test.

For the ImageLink, the producer publishes frame 42 by replacing the previous
image and metadata under a lock. Each consumer copies a coherent image and
metadata pair. A fast consumer might observe frames 40, 41, and 42, while a
slower consumer might observe frames 40 and 42. Neither consumer delays the
producer merely because it has not handled frame 41.

Regular Link delivery
---------------------

A regular Link passes complete dictionaries in sending order. Its receiving
methods make different choices about unread data:

- :meth:`~crappy.links.link.Link.recv` returns the oldest available dictionary.
- :meth:`~crappy.links.link.Link.recv_last` returns the newest available
  dictionary and discards the other unread dictionaries.
- :meth:`~crappy.links.link.Link.recv_chunk` returns all currently available
  values, grouped by label and ordered from oldest to newest.

If no data is available, these methods return an empty dictionary instead of
waiting. A Block should therefore treat an empty result as “no new data” and
continue according to its task.

Regular Links are buffered but not lossless storage. On Linux, a dictionary
being sent when the pipe is full is discarded and Crappy periodically logs a
warning. On other operating systems, the send operation can wait until buffer
space becomes available. A Recorder or another storage Block should receive
any values that must be preserved.

ImageLink latest-frame delivery
-------------------------------

An image-producing VisionBlock owns one shared image buffer for all of its
outgoing ImageLinks. Publishing replaces the image, its metadata, and an
internal update counter as one protected operation. A consumer also copies the
image and metadata while that state is protected, so it does not receive a
partially written image or metadata from a different frame.

An ImageLink is not a queue. When the producer publishes several images before
a consumer reads again, only the newest image is available. Skipping
intermediate images is expected behavior and lets acquisition, processing,
display, and recording run at different rates.

Every published metadata dictionary must contain:

- ``t(s)``, the image timestamp relative to the test start
- ``ImageUniqueID``, the identifier assigned to that image by its source

A consumer must read these metadata values from the image it actually
received. It must not calculate a timestamp from its own loop or assume that
``ImageUniqueID`` values will be consecutive. A gap can simply mean that newer
frames replaced intermediate frames before the consumer copied them.

The output image shape and data type are established during preparation and
remain fixed during the test. A processing stage that changes either property
publishes through its own output buffer with the new format.

Connection rules
----------------

Both Link types are recorded in the connection graph described in
:doc:`blocks_links_labels`. ImageLinks add these constraints:

- both endpoints must be VisionBlocks
- two VisionBlocks cannot have parallel ImageLinks in the same direction
- a cycle made entirely of ImageLinks is rejected

Regular Links may still connect the same VisionBlocks for commands, processing
results, or overlays. A graph may also contain feedback cycles when at least
one edge in the cycle is a regular Link.

Public behavior and implementation details
------------------------------------------

Dictionary delivery through regular Links and latest-frame delivery through
ImageLinks are the user-facing models. The current pipe, shared-memory,
synchronization, and internal counter objects are implementation details. Code
using Crappy should rely on the Link APIs and image metadata rather than access
those objects directly.

API reference
-------------

- :class:`crappy.links.link.Link` documents regular Link operations.
- :class:`crappy.links.img_link.ImageLink` documents image connection setup.
- :meth:`crappy.blocks.vision.block.VisionBlock.send_img` publishes an image.
- :meth:`crappy.blocks.vision.block.VisionBlock.receive_imgs` copies available
  new images into ``last_received``.
