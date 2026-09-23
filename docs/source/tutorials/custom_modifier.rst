.. _tutorial-custom-modifier:

========================
Create a custom Modifier
========================

This tutorial has one outcome: create a Modifier that converts a simulated
load-cell voltage into force.

Prerequisites
-------------

- Complete :doc:`modifiers`.
- Know the voltage-to-force calibration for the sensor you want to use.

The example requires no physical hardware, graphical interface, optional
Python package, or output file. It prints simulated values in the terminal and
stops automatically after three seconds.

Write the conversion
--------------------

:download:`Download the complete script
</downloads/custom_objects/custom_modifier_calibration.py>`, or create a file named
``custom_modifier.py``. The custom class in that script is:

.. literalinclude:: /downloads/custom_objects/custom_modifier_calibration.py
   :language: python
   :start-after: # [custom-modifier-class-start]
   :end-before: # [custom-modifier-class-end]

A custom Modifier inherits from
:class:`~crappy.modifier.meta_modifier.modifier.Modifier` and defines
``__call__()``. Each call receives a dictionary whose keys are labels. Here,
the Modifier reads ``voltage(V)``, calculates the force, adds it as
``force(N)``, and returns the dictionary.

The sensitivity and zero value are arguments so that the same class can be
used with different calibrations. The call to ``super().__init__()`` prepares
the features supplied by the parent Modifier class.

Attach it to a Link
-------------------

The complete script creates the Modifier and attaches it to one Link:

.. literalinclude:: /downloads/custom_objects/custom_modifier_calibration.py
   :language: python
   :start-after: # [custom-modifier-use-start]
   :end-before: # [custom-modifier-use-end]

The Generator provides a simulated voltage of 0.25 V. With a zero value of
0.02 V and a sensitivity of 100 N/V, the LinkReader receives a force close to
23 N. The original voltage label remains available as well.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_modifier.py

The terminal displays the simulated voltage and converted force. A final
``Generator Path exhausted`` warning indicates the planned end of the example.

Adapt it to a measurement
-------------------------

Replace the Generator with the Block that supplies the sensor measurement,
then make these three values agree:

1. The measurement label used in ``__call__()``.
2. The sensor's calibrated zero.
3. The conversion factor and its units.

A plain Python function can also be passed as a Modifier when the calculation
is small and specific to one script. A class is useful when the conversion has
settings, as in this example, or must be reused.

Keep a Modifier limited to a quick transformation of one dictionary. Returning
``None`` discards that dictionary. If the task needs setup, cleanup, or its own
timing, create a custom Block instead; the
:doc:`../concepts/choosing_custom_object_type` guide explains the choice.

See :class:`~crappy.modifier.meta_modifier.modifier.Modifier` for the complete
custom interface and :doc:`../crappy_docs/modifiers` for the built-in
alternatives.
