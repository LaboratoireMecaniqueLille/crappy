.. _tutorial-custom-camera:

=============================
Create a custom Camera object
=============================

This tutorial has one outcome: create a Camera object and acquire its images
with CameraSource.

Prerequisites
-------------

- Complete :doc:`image_pipeline`.
- To adapt the example to equipment, first confirm that Python can open the
  camera and acquire an image.

The example generates images without physical hardware or a graphical window.
It records a few NumPy arrays and their metadata in the operating system's
temporary directory, removes them when the script ends, and stops
automatically after two seconds. No optional Python package is required.

Define the Camera object
------------------------

:download:`Download the complete script
</downloads/custom_objects/custom_camera_source.py>`, or create a file named
``custom_camera.py``. Its custom Camera object is:

.. literalinclude:: /downloads/custom_objects/custom_camera_source.py
   :language: python
   :start-after: # [custom-camera-class-start]
   :end-before: # [custom-camera-class-end]

The methods have separate responsibilities:

- ``__init__()`` declares settings whose allowed values are already known. It
  does not open the camera.
- ``open()`` receives the user's connection and acquisition choices, opens
  and configures the device, and starts acquisition when necessary.
- ``get_image()`` returns one timestamp and one NumPy image. It may instead
  return a metadata dictionary containing at least ``t(s)`` and
  ``ImageUniqueID`` as its first value.
- ``close()`` stops acquisition and releases the camera connection.

This example adds a numeric ``brightness`` setting. Calling ``set_all()`` in
``open()`` applies the requested value, or the setting's default when no value
was supplied. With interactive configuration enabled, the same setting appears
as a slider. A real Camera object can give a setting getter and setter methods
that read from and write to the device.

Supply settings through ``open()``
----------------------------------

CameraSource creates the Camera object itself and then forwards
camera-specific keyword arguments to its ``open()`` method:

.. literalinclude:: /downloads/custom_objects/custom_camera_source.py
   :language: python
   :start-after: # [custom-camera-use-start]
   :end-before: # [custom-camera-use-end]

Here, ``width``, ``height``, and ``brightness`` are therefore parameters of
``GradientCamera.open()``, not constructor arguments. The ``camera`` value
matches the custom class name.

The example disables the configuration window, so it must also provide the
exact image shape and data type expected from ``get_image()``. The remaining
Blocks temporarily record selected images, print saved-frame notifications,
and stop the test.

Add specialized settings
------------------------

Some cameras need settings beyond an ordinary boolean, choice, or numeric
control:

- :meth:`~crappy.camera.meta_camera.camera.Camera.add_trigger_setting` adds the
  standard ``Free run``, ``Hdw after config``, and ``Hardware`` choices.
  Implement its getter and setter using the camera manufacturer's trigger
  controls. ``Hdw after config`` keeps acquisition free-running while the
  configuration window is open, then selects hardware triggering when the
  window closes.
- :meth:`~crappy.camera.meta_camera.camera.Camera.add_software_roi` adds
  controls for a software region of interest (ROI). Call it once the full image
  dimensions are known, then call
  :meth:`~crappy.camera.meta_camera.camera.Camera.apply_soft_roi` on every
  acquired image before returning it.

A software ROI reduces the image passed to later stages but does not make the
camera acquire faster. If another setting changes the full image dimensions,
call :meth:`~crappy.camera.meta_camera.camera.Camera.reload_software_roi` with
the new dimensions. Scale and choice settings also provide a ``reload()``
method for the unusual case where one setting changes another setting's limits
or available choices.

Return camera metadata
----------------------

``get_image()`` may return a metadata dictionary instead of a bare timestamp.
The dictionary must contain:

- ``t(s)`` with the acquisition timestamp
- ``ImageUniqueID`` with an integer identifying the acquired image

Add any measurements supplied by the camera, such as exposure time or the
device's capture timestamp. ImageRecorder writes these fields to
``metadata.csv``. With the Pillow backend, fields whose names are valid
Exchangeable Image File Format (EXIF) tags can also be embedded in the saved
image.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_camera.py

The terminal reports several saved frames before the stop condition is met.
CameraSource then calls ``close()``, and the temporary recording directory is
removed.

Adapt it to real hardware
-------------------------

.. warning::

   Before opening a physical camera, verify its connection, supported image
   formats, exposure limits, trigger mode, and shutdown procedure. Avoid
   enabling an external trigger until the trigger source is ready.

First check whether :doc:`../features` already lists a compatible Camera
object, especially :class:`~crappy.camera.Webcam` for common webcams. If a new
driver is required:

1. Open the device and apply connection arguments in ``open()``.
2. Add settings for choices users may change, and call ``set_all()`` after the
   device is ready.
3. Return the actual image shape and data type consistently from
   ``get_image()``.
4. Make ``close()`` safe after both normal acquisition and a partial setup
   failure.
5. Test with ``config=False`` and explicit ``img_shape`` and ``img_dtype``
   before enabling the interactive configuration window.

The :doc:`test_hardware_object` tutorial shows how to check the Camera object
directly. See :class:`~crappy.camera.meta_camera.camera.Camera` for the full
custom interface and :class:`~crappy.blocks.vision.CameraSource` for
acquisition options. The :doc:`../concepts/image_pipelines` guide explains the
recommended VisionBlock architecture and the supported all-in-one alternative.
