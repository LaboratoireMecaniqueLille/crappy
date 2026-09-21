.. _tutorial-test-hardware-object:

=============================
Test a Camera object directly
=============================

This tutorial has one outcome: open a Camera object, acquire one image, and
close it without building a Crappy test.

Prerequisites
-------------

- Complete :doc:`image_pipeline` or know the difference between a Camera
  object and a CameraSource.

This example uses ``FakeCamera`` and does not access physical hardware. It
requires no graphical interface or optional Python package, opens no window,
and creates no file. It prints the acquired image's shape, data type, and
timestamp, then exits.

Create the direct Camera test
-----------------------------

:download:`Download the complete script
</downloads/more_complexity/test_camera_object.py>`, or create a file named
``test_camera_object.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/test_camera_object.py
   :language: python
   :start-after: # [test-camera-object-start]
   :end-before: # [test-camera-object-end]

The script instantiates :class:`~crappy.camera.FakeCamera` directly instead of
giving its name to :class:`~crappy.blocks.vision.CameraSource`. It then calls
the Camera object's public methods:

1. ``open()`` applies the settings and prepares acquisition.
2. ``get_image()`` returns a timestamp and a NumPy image.
3. ``close()`` releases the Camera object's resources.

The ``finally`` block calls ``close()`` whether acquisition succeeds or raises
an exception. This cleanup pattern is required for direct hardware tests.

Run the test
------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python test_camera_object.py

The reported image shape is ``(240, 320)`` and its data type is ``uint8``.

Test a real Camera object
-------------------------

Direct access is useful for checking a connection or one setting before
building the full image pipeline. To test a webcam, for example:

1. Install OpenCV as described by the Camera API.
2. Replace ``crappy.camera.FakeCamera()`` with
   :class:`~crappy.camera.Webcam`.
3. Replace the arguments to ``open()`` with the Webcam arguments, such as
   ``device_num=0`` and ``channels='1'``.
4. Keep the ``try`` and ``finally`` structure unchanged.

Consult the selected Camera object's API before opening physical hardware.
Direct calls bypass CameraSource's configuration window and lifecycle
management, so the script is responsible for valid settings and cleanup.

The same principle applies to :class:`~crappy.inout.InOut` and
:class:`~crappy.actuator.Actuator` objects: instantiate the specific class,
call its public ``open()`` and acquisition or command methods, and always call
``close()`` in a ``finally`` block. For an Actuator, also call ``stop()``
before ``close()``. Follow the selected object's API because supported methods
and arguments vary by device.

.. warning::

   A direct test can send commands immediately, without the limits or cleanup
   supplied by a complete test. Before testing physical equipment, verify its
   command units, safe range, initial state, and independent emergency-stop
   system.

See :doc:`../concepts/choosing_custom_object_type` for the roles of Camera,
InOut, and Actuator objects. For test startup and shutdown options, see
:doc:`../concepts/lifecycle_shutdown`. Developers who need the separate
startup stages can consult :doc:`../architecture`.
