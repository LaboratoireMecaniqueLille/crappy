.. _tutorial-custom-vision-block:

===========================
Create a custom VisionBlock
===========================

This tutorial has one outcome: turn each camera image into a numerical
measurement with a reusable image-analysis Block.

Prerequisites
-------------

- Complete :doc:`image_pipeline`.
- Know that ImageLinks carry images while regular Links carry labeled results.
  See :doc:`../concepts/regular_links_and_image_links` for a refresher.

The example uses the simulated camera included with Crappy. It requires no
physical hardware, graphical interface, optional Python package, or output
file. It prints the position of the brightest image row and stops
automatically after three seconds.

Define the image analysis
-------------------------

:download:`Download the complete script
</downloads/custom_objects/custom_vision_block_brightness.py>`, or create a
file named ``custom_vision_block.py``. Its custom VisionBlock is:

.. literalinclude:: /downloads/custom_objects/custom_vision_block_brightness.py
   :language: python
   :start-after: # [custom-vision-block-class-start]
   :end-before: # [custom-vision-block-class-end]

Every custom image stage inherits from
:class:`~crappy.blocks.vision.block.VisionBlock`. This one only receives
images, so it does not declare an output image shape or data type.

``prepare()`` checks that exactly one incoming ImageLink is connected and that
there is no outgoing ImageLink. The parent method is called last to complete
the image setup.

Crappy then calls ``loop()`` repeatedly. ``receive_imgs()`` returns the names
of the ImageLinks that have a new image. If there is no new image, the method
returns immediately. The matching image and its metadata are available in
``last_received``.

This example averages each row, finds the row with the largest value, and
sends its position as labeled numerical data. Small results like this belong
on regular Links.

Connect the image stage
-----------------------

The complete script creates a simulated CameraSource, then connects it to the
custom Block:

.. literalinclude:: /downloads/custom_objects/custom_vision_block_brightness.py
   :language: python
   :start-after: # [custom-vision-block-use-start]
   :end-before: # [custom-vision-block-use-end]

``img_link()`` carries the camera images to ``BrightestRow``. The two regular
Links in the complete script carry its numerical results to a LinkReader and a
StopBlock.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_vision_block.py

The terminal displays an image index and a moving position in
``brightest_row(px)``. A ``Stop criterion reached`` warning indicates the
planned end of the example.

Adapt the calculation
---------------------

Replace the NumPy calculation in ``loop()`` with the analysis required by the
experiment. Keep these points unchanged:

- Return promptly when ``receive_imgs()`` reports no new image.
- Read the image and metadata from the same ``last_received`` entry.
- Include the received timestamp with the calculated results.
- Use regular Links for small measurements and ImageLinks for image arrays.

An ImageLink makes the newest complete image available. If analysis is slower
than acquisition, some intermediate images can be skipped. Use the metadata
of the image actually received rather than assuming that image indices are
consecutive.

Choose another VisionBlock role
-------------------------------

A custom VisionBlock can also produce or transform images:

- An image source declares its output shape and data type, creates image
  metadata, and calls
  :meth:`~crappy.blocks.vision.block.VisionBlock.send_img` for every image.
- An image filter has both an incoming and an outgoing ImageLink. It receives
  an image, transforms it without changing its declared output format during
  the test, and publishes the result with ``send_img()``.

Use separate VisionBlocks when acquisition, analysis, display, and recording
should be combined independently. VisionBlocks are recommended for new image
pipelines.

For a real experiment, replace ``FakeCamera`` and its simulated settings with
the appropriate Camera object in ``CameraSource``. The custom
``BrightestRow`` class does not otherwise depend on the image source.

See :class:`~crappy.blocks.vision.block.VisionBlock` for the complete custom
interface and :doc:`../concepts/image_pipelines` for the available image
architectures.
