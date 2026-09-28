.. _tutorial-scheduler:

======================================
Branch between phases with a Scheduler
======================================

This tutorial has one outcome: build a test that moves from loading to holding
when a force target is reached, or aborts the load if a timeout occurs first.
Those two possible next phases make a :class:`~crappy.blocks.Scheduler` a
better fit than a fixed Generator Path sequence.

Prerequisites
-------------

- Complete :doc:`command_generation` or know how to link Blocks.
- Read :doc:`generator_conditions` if measurement-driven transitions are new
  to you.

The script uses a FakeMachine and needs no physical hardware, graphical
interface, optional Python package, or output file. It normally stops after
about two seconds. The timeout branch also stops automatically.

Create the state graph
----------------------

:download:`Download the complete script
</downloads/more_complexity/scheduler.py>`, or create a file named
``scheduler.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/scheduler.py
   :language: python
   :start-after: # [scheduler-start]
   :end-before: # [scheduler-end]

The first State, ``Load``, sends a speed of 1 mm/s to the FakeMachine. Its
transition pairs are checked in order, before that loop's output is generated:

1. If the most recent ``F(N)`` value exceeds 25,000 N, enter ``Hold``.
2. Otherwise, if two seconds have passed in ``Load``, enter ``Abort``.

If both are true in one loop, ``Hold`` wins because its condition comes first.
``Hold`` sends zero speed briefly, then ``Unload`` sends -1 mm/s. ``Abort``
sends zero speed and ends the test after a short pause. The reserved ``End``
State sends ``last_output`` (zero speed) and, after ``end_delay``, stops
Crappy. The Scheduler also adds a ``state`` label to each output, making
transitions visible in the LinkReader.

The Link from ``machine`` back to ``scheduler`` provides force feedback.
``sigma={}`` removes simulated measurement noise so the normal branch is
repeatable.

``input_labels=('F(N)',)`` declares which received label the States need, and
``safe_start=True`` prevents State output functions from being evaluated until
the first force value has arrived. It does *not* pause transition conditions
or their timers. In this script, FakeMachine publishes an initial measurement
when it starts. For other sources, a delayed or missing measurement can still
let a timeout select ``Abort`` before any command is sent.

Run and modify the example
--------------------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python scheduler.py

The LinkReader shows the ``Load``, ``Hold``, ``Unload``, and ``End`` phases
alongside the simulated measurements. Raise the force target above what the
simulation reaches in two seconds to see ``Abort`` instead. Change the Delays
to adjust how long each phase lasts, or add another State and transition pair
to extend the procedure.

Choose between Scheduler and Generator
--------------------------------------

The :class:`~crappy.blocks.Generator` is the simpler choice for a linear
sequence of signal segments, even when a measurement determines when to advance
to the next segment. Its Paths always advance in a fixed order. Scheduler
requires more Python structure, but its named States can branch, loop back,
generate several coordinated output labels, and use arbitrary functions for
outputs or conditions. If you are comfortable with that extra complexity, use a
Scheduler as is easier to extend when a test grows beyond its initial linear
plan.

The built-in :doc:`Scheduler helpers <../crappy_docs/schedulers>` cover common
signals and conditions. They are optional: a State accepts a callable with
``(dt, data)`` arguments for an output or condition. You can define custom
functions at module scope and use them as outputs or conditions.
The :example:`custom Scheduler example <blocks/scheduler/scheduler_custom.py>`
demonstrates this pattern.

.. warning::

   A timeout transition and a final zero command in a test script are not
   independent safety mechanisms. Before controlling physical equipment,
   verify command limits, motion direction, feedback validity, and use an
   emergency-stop system outside this state graph.
