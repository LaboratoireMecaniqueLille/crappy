.. _tutorial-streaming-acquisition:

======================
Acquire data in chunks
======================

This tutorial has one outcome: acquire chunks of simulated measurements,
record every value in an HDF5 file, and display a reduced value in the
terminal.

Prerequisites
-------------

- Complete :doc:`data_acquisition` and :doc:`data_recording`.
- Install the packages used by the simulated InOut and HDF5 recorder:

  .. code-block:: shell-session

     python -m pip install psutil tables

This example reads only the computer's memory usage and sends no hardware
command. It opens no graphical window and stops automatically after three
seconds.

The script creates a uniquely named folder in your operating system's
temporary directory and writes ``memory.h5`` inside it. It prints the complete
path before acquisition begins and after cleanup closes the file. Temporary
files are removed automatically by the operating system, so copy the file
elsewhere if you want to keep it.

Create the streaming script
---------------------------

:download:`Download the complete script
</downloads/more_complexity/streaming_acquisition.py>`, or create a file named
``streaming_acquisition.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/streaming_acquisition.py
   :language: python
   :start-after: # [streaming-acquisition-start]
   :end-before: # [streaming-acquisition-end]

Setting ``streamer=True`` tells the IOBlock to call the InOut's streaming
methods. Instead of returning one measurement at a time, ``FakeInOut`` returns
each group of ten memory measurements as a NumPy array under the ``stream``
label.

The two outgoing Links use that chunk differently:

- The Link to :class:`~crappy.blocks.HDFRecorder` preserves and records every
  value in the chunk.
- The Link to the LinkReader applies :class:`~crappy.modifier.Demux`. With
  ``mean=True``, the Modifier replaces each chunk with its mean timestamp and
  mean memory usage so an ordinary Block can consume it.

Demux is useful for a low-rate display or decision, but it deliberately drops
the individual values in each chunk. Always send the original stream directly
to HDFRecorder when every acquired value must be saved.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python streaming_acquisition.py

The terminal displays successive mean memory values. A
``Stop criterion reached`` warning indicates the planned end of the example.
Crappy then stops the stream, closes the InOut, and closes the HDF5 file.

The saved file contains an extendable array named ``table``. You can inspect
its first rows with PyTables after replacing the path below with the one
printed by the script:

.. code-block:: python

   import tables

   with tables.open_file('PATH_PRINTED_BY_THE_SCRIPT') as file:
     print(file.root.table[:5])

Use a real streaming device
---------------------------

Not every InOut supports streaming. Before adapting this example:

1. Confirm that the selected InOut implements the streaming methods and check
   its API for sample-rate, channel, and connection arguments.
2. Replace ``FakeInOut`` with that InOut's name and configure those arguments
   on the IOBlock.
3. Keep the IOBlock's stream label identical to HDFRecorder's ``label`` and
   Demux's ``stream_label``.
4. Set HDFRecorder's ``atom`` to the acquired data type.
5. Give Demux one label per stream column, in the same order as the device
   returns them. Use ``transpose=True`` if channels are stored by row.
6. Write to durable storage with enough capacity and verified write speed,
   rather than the temporary path used by this tutorial.

An interrupted write can leave an HDF5 file unreadable. Provide a safe way to
stop the test and allow normal cleanup to close the file. See
:doc:`../concepts/lifecycle_shutdown` for shutdown behavior.

Continue with :doc:`organize_scripts` to reduce repetition in larger tests.
