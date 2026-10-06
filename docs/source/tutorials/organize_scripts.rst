.. _tutorial-organize-scripts:

==================================
Reduce repetition in a test script
==================================

This tutorial has one outcome: use ordinary Python variables, collections,
and a loop to configure repeated parts of a Crappy test.

Prerequisites
-------------

- Complete :doc:`quickstart` and :doc:`data_recording`.
- Be familiar with Python variables and ``for`` loops.

This example requires no physical hardware, graphical interface, or optional
Python package. It creates a unique folder in your operating system's
temporary directory, writes three CSV files inside it, and stops automatically
after three seconds. The script prints the folder path before and after the
test. Copy files elsewhere if you want to keep them.

Create the organized script
---------------------------

:download:`Download the complete script
</downloads/more_complexity/organized_script.py>`, or create a file named
``organized_script.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/organized_script.py
   :language: python
   :start-after: # [organized-script-start]
   :end-before: # [organized-script-end]

Name values that change together
--------------------------------

The ``speed_steps`` tuple keeps the three command levels in one place. A
comprehension turns each level into a one-second Constant Path:

.. code-block:: python

   paths = tuple(
       {'type': 'Constant',
        'value': speed,
        'condition': 'delay=1'}
       for speed in speed_steps)

Changing the sequence now requires editing only ``speed_steps``. The resulting
``paths`` tuple is passed to the Generator exactly like a tuple written out by
hand.

Create similar Blocks in a loop
-------------------------------

The ``measurements_to_record`` dictionary associates each output filename with
one measurement label. The following loop creates one Recorder and one Link
for each entry. Adding another recording requires adding one dictionary entry
instead of copying and editing a complete Block definition.

The Recorder objects are also kept in the ``recorders`` list. Keeping related
objects together makes them available for later inspection or configuration.

The :class:`pathlib.Path` object joins the output directory and each filename
with the correct separator for the current operating system:

.. code-block:: python

   file_name=output_dir / file_name

Run and inspect the result
--------------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python organized_script.py

The printed folder contains ``force.csv``, ``position.csv``, and
``strain.csv``. Each file contains the shared ``t(s)`` label and the selected
measurement.

Use a loop when several definitions follow the same rule. Keep definitions
separate when their differences are easier to understand explicitly. A shorter
script is useful only when its configuration remains clear.

Continue with :doc:`test_hardware_object` to check one integration object
without building a complete Crappy test.
