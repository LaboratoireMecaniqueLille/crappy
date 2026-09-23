.. _tutorial-custom-generator-path:

==============================
Create a custom Generator Path
==============================

This tutorial has one outcome: create a reusable Generator Path that produces
a timed pulse command.

Prerequisites
-------------

- Complete :doc:`command_generation`.
- Read :doc:`generator_conditions` if the Path must react to measurements.

The example requires no physical hardware, graphical interface, optional
Python package, or output file. It prints commands in the terminal and stops
automatically after four seconds.

Define the Path
---------------

:download:`Download the complete script
</downloads/custom_objects/custom_generator_path.py>`, or create a file named
``custom_generator_path.py``. Its custom Path is:

.. literalinclude:: /downloads/custom_objects/custom_generator_path.py
   :language: python
   :start-after: # [custom-generator-path-class-start]
   :end-before: # [custom-generator-path-class-end]

``crappy.Path`` is the base class for Generator Paths. The constructor accepts
the keys that will appear in the Path dictionary and validates their values.
``parse_condition()`` converts a delay, threshold, callable, or ``None`` into
a condition the Path can check consistently.

The Generator repeatedly calls ``get_cmd()``. The ``data`` dictionary contains
lists of values recently received by the Generator. This pulse does not need
measurement feedback, but passes that dictionary to its condition. When the
condition is met, raising ``StopIteration`` tells the Generator to continue to
the next Path. Otherwise, the method returns the next command value.

Use it in a Generator
---------------------

The complete script selects the custom class by name in an ordinary Generator
Path dictionary:

.. literalinclude:: /downloads/custom_objects/custom_generator_path.py
   :language: python
   :start-after: # [custom-generator-path-use-start]
   :end-before: # [custom-generator-path-use-end]

Every one-second period stays at 5 V for 0.2 seconds and at 0 V for the
remaining 0.8 seconds. ``spam=True`` publishes the current value repeatedly,
including while it remains unchanged.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_generator_path.py

The terminal shows short groups of 5 V commands separated by longer groups of
0 V commands. A final ``Generator Path exhausted`` warning indicates the
planned end of the example.

Adapt the Path
--------------

Check the :doc:`../crappy_docs/blocks` reference before creating a new Path,
the built-in Constant, Ramp, Cyclic, Sine, Conditional, Integrator, and Custom
Paths cover many command profiles.

For a new profile:

1. Accept and validate its settings in ``__init__()``.
2. Use ``self.t0`` as the start time of the current Path and
   ``self.last_cmd`` when the previous command matters.
3. Return a numeric command or ``None`` from ``get_cmd()``.
4. Raise ``StopIteration`` when the Path is complete.
5. For a measurement condition, connect the measurement source to the
   Generator and inspect the lists in ``data`` or use ``parse_condition()``.

A Generator Path defines one command segment; it does not replace the
Generator Block that runs the sequence. See
:class:`~crappy.blocks.generator_path.meta_path.path.Path` for the complete
custom interface.
