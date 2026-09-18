.. _tutorial-data-recording:

=================================
Record measurements in a CSV file
=================================

This tutorial has one outcome: save selected measurements to a
comma-separated values (CSV) file with a :class:`~crappy.blocks.Recorder`.

Prerequisites
-------------

- Complete :doc:`quickstart` or be familiar with creating Blocks and Links.

This example requires no physical hardware, graphical interface, or optional
Python package. It creates one uniquely named folder in your operating
system's temporary directory and writes a file named ``measurements.csv``
inside it. The script prints the complete path before the test starts and
again after it stops. It does not overwrite an existing recording.

Files in the temporary directory will be removed automatically by the operating
system. Copy the CSV file elsewhere if you want to keep it.

Create the recording script
---------------------------

:download:`Download the complete script
</downloads/getting_started/data_recording.py>`, or create a file named
``data_recording.py`` containing this code:

.. literalinclude:: /downloads/getting_started/data_recording.py
   :language: python
   :start-after: # [data-recording-start]
   :end-before: # [data-recording-end]

The Generator and FakeMachine produce simulated measurements for three
seconds. The Recorder receives those measurements and saves only ``t(s)``,
``F(N)``, and ``x(mm)`` because they are listed in its ``labels`` argument.
The first row of the file contains these label names. The following rows
contain their values.

A Recorder accepts data from one upstream Block. The Link from ``machine`` to
``recorder`` therefore carries all the data this Recorder can select and save.
Use a separate Recorder for each additional upstream Block.

Keep the ``if __name__ == '__main__':`` lines in every Crappy script so that it
starts correctly on all supported operating systems.

Run the test and open the result
--------------------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python data_recording.py

After the test stops, copy the printed path into a text editor or spreadsheet
application. The beginning of the file has this form:

.. code-block:: text

   t(s),F(N),x(mm)
   0.0,...,...

The exact values and number of rows vary between runs.

Record real measurements
------------------------

To record data from a physical device, first follow
:doc:`data_acquisition` and verify the acquisition without the Recorder.
Then link that IOBlock to ``recorder`` instead of linking ``machine`` to it,
and set ``labels`` to the labels produced by the IOBlock. Review the output
path before every experiment.

The regular Recorder is not intended for an IOBlock in streaming mode. Use
:class:`~crappy.blocks.HDFRecorder` for streamed data.

Next steps
----------

Continue with :doc:`signal_display` to display measurements live, or read
:doc:`../concepts/blocks_links_labels` to learn how a Block selects labels
received through a Link.
