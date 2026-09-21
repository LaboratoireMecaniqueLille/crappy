.. _tutorial-data-acquisition:

==================================
Acquire measurements with IOBlock
==================================

This tutorial has one outcome: acquire measurements from an InOut with an
:class:`~crappy.blocks.IOBlock` and display them in the terminal.

Prerequisites
-------------

- Complete :doc:`quickstart` or be familiar with creating Blocks and Links.
- Install the ``psutil`` package used by the simulated InOut:

  .. code-block:: shell-session

     python -m pip install psutil

This example reads only the computer's memory usage. It sends no hardware
command, opens no graphical window, creates no file, and stops automatically
after three seconds.

Create the acquisition script
-----------------------------

:download:`Download the complete script
</downloads/getting_started/data_acquisition.py>`, or create a file named
``data_acquisition.py`` containing this code:

.. literalinclude:: /downloads/getting_started/data_acquisition.py
   :language: python
   :start-after: # [data-acquisition-start]
   :end-before: # [data-acquisition-end]

``FakeInOut`` plays the role of a physical input device. It returns a timestamp
and the computer's memory usage. The IOBlock publishes them under the ``t(s)``
and ``ram(%)`` labels. One Link sends the measurements to ``reader`` for
display, while the other lets ``stop`` end the test after three seconds.

Run the test
------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python data_acquisition.py

The terminal displays successive ``ram(%)`` measurements. A final
``Stop criterion reached`` warning indicates the planned end of this example.
Crappy then closes the InOut as part of its normal cleanup.

Use a real input device
-----------------------

To adapt this script to real hardware:

1. Check :doc:`../features` for an InOut matching the device, or create your
   own InOut object if no driver is distributed with Crappy.
2. Replace ``'FakeInOut'`` with that InOut's name and provide its required
   arguments to IOBlock.
3. Optionally replace ``('t(s)', 'ram(%)')`` with more meaningful labels
   for the values returned by the selected InOut.
4. Install any additional library the InOut requires.

The :doc:`../concepts/blocks_links_labels` page explains how labels identify
values after the IOBlock publishes them.

Next step
---------

Continue with :doc:`signal_display` to plot acquired or simulated values in a
live graph. If the selected InOut supports chunked acquisition, continue with
:doc:`streaming_acquisition`.
