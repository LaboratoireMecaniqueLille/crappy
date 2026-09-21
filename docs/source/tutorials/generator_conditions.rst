.. _tutorial-generator-conditions:

==============================================
Change a command when a measurement is reached
==============================================

This tutorial has one outcome: end a Generator Path when a received
measurement crosses a threshold.

Prerequisites
-------------

- Complete :doc:`command_generation`.
- Complete :doc:`feedback_loops` or understand how a Block can return
  measurements to an earlier Block.

This example uses a FakeMachine and sends no command to physical hardware. It
requires no graphical interface or optional Python package, creates no file,
and normally stops after about two seconds.

Create the conditional command script
-------------------------------------

:download:`Download the complete script
</downloads/more_complexity/generator_conditions.py>`, or create a file named
``generator_conditions.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/generator_conditions.py
   :language: python
   :start-after: # [generator-conditions-start]
   :end-before: # [generator-conditions-end]

The first Constant Path sends a simulated speed of 1 mm/s. Unlike the
time-based Paths in :doc:`command_generation`, its condition monitors a
measurement:

.. code-block:: python

   'condition': 'x(mm) > 1'

The Link from ``machine`` back to ``command`` supplies that label. As soon as
one received position is greater than 1 mm, the Generator switches to the
second Path. That Path sends zero for one second, then the example stops.

Setting ``safe_start=True`` makes the Generator wait for its first measurement
before sending the first command. This prevents it from evaluating the
measurement-dependent Path before feedback is available.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python generator_conditions.py

The terminal shows the position increasing past 1 mm and then remaining nearly
constant during the final zero command. A ``Generator Path exhausted`` warning
indicates the planned end of the example.

Write threshold conditions
--------------------------

A Generator Path accepts these string condition forms:

- ``'label > value'`` ends the Path when a received value is above the
  threshold.
- ``'label < value'`` ends the Path when a received value is below the
  threshold.
- ``'delay = seconds'`` ends the Path after a duration.

Spaces around the comparison sign are optional. Equality is not provided
because exact comparisons are unreliable for measured and floating-point
values. Choose an upper or lower threshold that represents the required
transition.

Use a condition with real equipment
-----------------------------------

.. warning::

   A measurement condition is not an independent safety limit. A missing,
   frozen, or incorrectly labeled measurement may prevent the transition.
   Physical equipment requires verified command limits, an independent
   emergency stop, and a separate way to stop the test safely.

To adapt this pattern:

1. Connect the acquisition Block carrying the monitored label to the Generator.
2. Match the condition's label and units exactly to that acquired value.
3. Connect the Generator output to the Block receiving the command.
4. Keep ``safe_start=True`` so the first command waits for feedback.
5. End with a Path that sends an appropriate safe command.

Use a :class:`~crappy.blocks.StopBlock` or another independent stop mechanism
when the entire test must also have a maximum duration. For conditions that
cannot be written with ``<``, ``>``, or ``delay``, see the
:doc:`custom_generator_path` guide.

Continue with :doc:`streaming_acquisition` to acquire data in chunks.
