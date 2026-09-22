.. _tutorial-custom-streaming-inout:

================================
Create a custom streaming InOut
================================

This tutorial has one outcome: create an InOut that returns measurements in
chunks and use it with an IOBlock.

Prerequisites
-------------

- Complete :doc:`custom_inout`.
- Complete :doc:`streaming_acquisition` or know why chunked acquisition is
  useful for the device.

The example uses a simulated signal. It requires no physical hardware,
graphical interface, optional Python package, or output file. It prints one
mean value per chunk and stops automatically after three seconds.

Define the streaming methods
----------------------------

:download:`Download the complete script
</downloads/custom_objects/custom_inout_streamer.py>`, or create a file named
``custom_streaming_inout.py``. Its custom InOut is:

.. literalinclude:: /downloads/custom_objects/custom_inout_streamer.py
   :language: python
   :start-after: # [custom-streaming-inout-class-start]
   :end-before: # [custom-streaming-inout-class-end]

The methods are called in this order:

1. ``open()`` opens and configures the device.
2. ``start_stream()`` starts its continuous acquisition.
3. ``get_stream()`` repeatedly retrieves the available chunks.
4. ``stop_stream()`` stops acquisition during cleanup.
5. ``close()`` releases the device connection.

``get_stream()`` returns two NumPy arrays. For a chunk containing ``m`` samples
from ``n`` channels, timestamps have shape ``(m,)`` and measurements have
shape ``(m, n)``. This example returns five timestamps and a corresponding
``(5, 1)`` signal array on each call.

A dictionary can be returned instead. In that case, use ``t(s)`` for the
timestamp array and consistent labels for the measurement arrays.

Enable streaming on IOBlock
---------------------------

The complete script selects the class and enables its streaming methods:

.. literalinclude:: /downloads/custom_objects/custom_inout_streamer.py
   :language: python
   :start-after: # [custom-streaming-inout-use-start]
   :end-before: # [custom-streaming-inout-use-end]

The first IOBlock label corresponds to the timestamp array, and ``stream``
corresponds to the two-dimensional measurement array. The simulated InOut
generates five samples per chunk at 50 Hz, while IOBlock requests one chunk
about ten times per second.

The example applies :class:`~crappy.modifier.Demux` to each outgoing Link.
With ``mean=True``, Demux converts a chunk to its mean timestamp and mean
signal value so ordinary Blocks can consume it.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_streaming_inout.py

The terminal displays successive means of a simulated sine wave. A
``Stop criterion reached`` warning indicates the planned end of the example.
IOBlock then calls ``stop_stream()`` before ``close()``.

Preserve every sample when required
-----------------------------------

Demux deliberately discards the individual samples after calculating the
selected representative value. It is useful for a low-rate display or stop
decision, but not for lossless recording.

Connect the unmodified stream directly to
:class:`~crappy.blocks.HDFRecorder` when every sample must be saved. The
:doc:`streaming_acquisition` tutorial demonstrates that pattern and explains
the storage requirements.

Adapt it to a real device
-------------------------

Before writing the custom class, confirm the device driver's sample format,
channel order, data type, buffer behavior, and stop procedure. Then:

1. Configure its sample rate and channels in ``open()``.
2. Start acquisition only in ``start_stream()``.
3. Convert each driver buffer to stable NumPy shapes in ``get_stream()``.
4. Stop acquisition promptly in ``stop_stream()``, including after errors.
5. Release all handles in ``close()``.
6. Verify that storage can sustain the complete data rate before a real test.

``make_zero_delay`` relies on the regular ``get_data()`` method before
streaming starts, so it cannot zero a pure streaming InOut. A class that
supports both modes can use it when the regular measurement channels match
the stream columns.

See :class:`~crappy.inout.InOut` for the complete streaming interface and
:class:`~crappy.blocks.IOBlock` for streamer, command, and zeroing options.
