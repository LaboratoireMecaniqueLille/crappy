.. _tutorial-quickstart:

=======================
Run a first Crappy test
=======================

This tutorial has one outcome: run a complete Crappy test that uses simulated
hardware and stops by itself after a few seconds.

Prerequisites
-------------

- Crappy is installed. Follow :doc:`../installation` if needed.
- You can create a Python file and run it from a terminal.

The example requires no physical hardware, graphical interface, or optional
Python package. It does not create any files.

Create the test script
----------------------

:download:`Download the complete script
</downloads/getting_started/quickstart.py>`, or create a file named
``quickstart.py`` containing this code:

.. literalinclude:: /downloads/getting_started/quickstart.py
   :language: python
   :start-after: # [quickstart-start]
   :end-before: # [quickstart-end]

The script creates three Blocks:

- ``command`` sends a constant speed command for three seconds.
- ``machine`` simulates a tensile-test machine and produces measurements.
- ``reader`` prints those measurements in the terminal.

The two calls to :func:`crappy.link` define the direction in which data moves.
The command goes to the simulated machine, then the measurements go to the
reader. :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` starts the
test. The Generator stops the complete test when its three-second command ends.

Keep the ``if __name__ == "__main__":`` lines in every Crappy script. They let
the script start correctly on all supported operating systems.

Run the test
------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python quickstart.py

The terminal displays dictionaries containing the simulated force, position,
and strain measurements. The test then stops automatically. A final
``Generator Path exhausted`` warning indicates the planned end of this example,
not a failure. The meaning of Blocks, Links, dictionaries, and labels is
explained in :doc:`../concepts/blocks_links_labels`.

Next steps
----------

Continue with :doc:`data_acquisition` to read values through an IOBlock or
:doc:`data_recording` to save measurements in a CSV file. Follow
:doc:`signal_display` to plot values in a live graph. To produce and use
commands, continue with :doc:`command_generation` and
:doc:`actuator_control`. To acquire and display images, follow
:doc:`image_pipeline`.
