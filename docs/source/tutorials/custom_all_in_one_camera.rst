.. _tutorial-custom-all-in-one-camera:

=====================================
Customize the all-in-one Camera Block
=====================================

This tutorial has one outcome: add image analysis inside a custom all-in-one
Camera Block.

Prerequisites
-------------

- Complete :doc:`image_pipeline`.
- Read the architecture comparison in :doc:`../concepts/image_pipelines`.
- Be comfortable creating a small Python class.

The example uses the simulated camera included with Crappy. It requires no
physical hardware, graphical interface, optional Python package, or output
file. It prints the position of the brightest image row and stops
automatically after three seconds.

.. important::

   VisionBlocks are recommended for new image pipelines. The all-in-one Camera
   Blocks remain supported and are not planned for deprecation.

Use this advanced route when maintaining an all-in-one Camera integration or
when acquisition, processing, display, and recording should deliberately be
configured through one Block. For a reusable stage in a new image pipeline,
follow :doc:`custom_vision_block` instead.

Define the image processing
---------------------------

:download:`Download the complete script
</downloads/custom_objects/custom_all_in_one_camera.py>`, or create a file
named ``custom_all_in_one_camera.py``. The first custom class performs the
calculation:

.. literalinclude:: /downloads/custom_objects/custom_all_in_one_camera.py
   :language: python
   :start-after: # [custom-all-in-one-camera-process-start]
   :end-before: # [custom-all-in-one-camera-process-end]

A :class:`~crappy.blocks.camera_processes.CameraProcess` receives each new
image in ``self.img`` and its matching metadata in ``self.metadata``. Crappy
calls ``loop()`` once for each image made available to this processing stage.

This example requires a two-dimensional greyscale image. It averages each row,
finds the one with the largest value, then calls ``send()`` to publish the
timestamp, image index, and measured position through the Camera Block's
regular output Links.

Attach it to a Camera Block
---------------------------

The second class selects that processing stage:

.. literalinclude:: /downloads/custom_objects/custom_all_in_one_camera.py
   :language: python
   :start-after: # [custom-all-in-one-camera-block-start]
   :end-before: # [custom-all-in-one-camera-block-end]

Assign ``process_proc`` before calling the parent ``prepare()``. The parent
method opens the Camera object and manages the processing stage together with
any requested display or recording.

``BrightestRowCamera`` inherits the complete constructor of
:class:`crappy.blocks.Camera`. It can therefore be created with the same
camera, configuration, display, and recording arguments:

.. literalinclude:: /downloads/custom_objects/custom_all_in_one_camera.py
   :language: python
   :start-after: # [custom-all-in-one-camera-use-start]
   :end-before: # [custom-all-in-one-camera-use-end]

The complete script connects the custom Camera Block to a LinkReader and a
StopBlock. It does not use an ImageLink because image handling is contained
inside the all-in-one Block.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_all_in_one_camera.py

The terminal displays an image index and a moving position in
``brightest_row(px)``. A ``Stop criterion reached`` warning indicates the
planned end of the example.

Add arguments and resources
---------------------------

If the calculation requires settings, add them to the custom Camera Block,
store them, and pass them to the CameraProcess when it is created in
``prepare()``. Forward the standard Camera Block arguments to the parent
constructor.

Keep the CameraProcess constructor limited to storing settings. Create
expensive analysis objects or open other resources in
:meth:`~crappy.blocks.camera_processes.CameraProcess.init`. Release them in
:meth:`~crappy.blocks.camera_processes.CameraProcess.finish`. Neither method
is needed for this NumPy-only example.

Optional display configuration
------------------------------

Set ``display_images=True`` on the Camera Block when the experiment needs an
image window. A CameraProcess can call
:meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` with
:class:`~crappy.tool.camera_config.config_tools.Overlay` objects to mark
detected regions on that display. Sending overlays has no effect when the
display is disabled.

Some analysis methods require the user to select a region or other settings
before the test. For that advanced case:

1. Override :meth:`crappy.blocks.Camera._configure` to return a suitable
   :class:`~crappy.tool.camera_config.CameraConfig`.
2. Make its ``get_config()`` method return the selected values as a tuple.
3. Define matching arguments on
   :meth:`~crappy.blocks.camera_processes.CameraProcess.set_config`.

Crappy passes those values to the CameraProcess before image handling begins.
Use ordinary constructor arguments instead when interactive selection is not
needed.

Adapt it to a real camera
-------------------------

Replace ``FakeCamera`` with the Camera object name used by the experiment.
Remove the simulated ``width``, ``height``, ``speed``, and ``fps`` settings,
then supply the arguments accepted by that Camera object's ``open()`` method.

With ``config=True``, the configuration window determines the image shape and
data type. With ``config=False``, provide the exact ``img_shape`` and
``img_dtype`` returned by the camera. Convert colour images to a scalar
intensity or adapt the calculation before using ``BrightestRowProcess``.

See :class:`crappy.blocks.Camera` and
:class:`~crappy.blocks.camera_processes.CameraProcess` for the complete
interfaces.
