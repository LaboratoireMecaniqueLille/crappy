.. _tutorial-command-generation:

================================
Generate a sequence of commands
================================

This tutorial has one outcome: create a two-step command sequence with a
:class:`~crappy.blocks.Generator` and inspect its values in the terminal.

Prerequisites
-------------

- Complete :doc:`quickstart` or be familiar with creating Blocks and Links.

This example requires no hardware, graphical interface, or optional Python
package. It creates no file and stops automatically after four seconds.

Create the command script
-------------------------

:download:`Download the complete script
</downloads/getting_started/command_generation.py>`, or create a file named
``command_generation.py`` containing this code:

.. literalinclude:: /downloads/getting_started/command_generation.py
   :language: python
   :start-after: # [command-generation-start]
   :end-before: # [command-generation-end]

The Generator's ``path`` contains two dictionaries that run in order:

1. ``Ramp`` increases the command from zero at one unit per second for two
   seconds.
2. ``Constant`` holds the command at two for another two seconds.

The ``type`` entry selects the Generator Path. Its other entries configure
that Path. Here, both ``condition`` entries use ``delay`` to set the duration.
The Generator publishes values under the ``target`` label. Setting
``spam=True`` makes it publish the current value on every loop, including
while the value remains constant.

Run the test
------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python command_generation.py

The terminal displays the increasing values followed by the constant values.
A final ``Generator Path exhausted`` warning indicates the planned end of the
sequence, not a failure.

Change the sequence
-------------------

Change ``speed`` to adjust the ramp slope, ``value`` to select the held value,
or either delay to adjust a step's duration. The
:ref:`Generator Paths reference <crappy_docs/blocks:generator paths>` lists the
other available signal shapes and their arguments.

Continue with :doc:`actuator_control` to send a generated command to a
simulated motor.
