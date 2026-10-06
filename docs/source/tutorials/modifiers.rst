.. _tutorial-modifiers:

==============================
Transform data with a Modifier
==============================

This tutorial has one outcome: attach a Modifier to a Link and observe the new
label that it adds to the data.

Prerequisites
-------------

- Complete :doc:`quickstart`.
- Know that Links carry labeled dictionaries. See
  :doc:`../concepts/blocks_links_labels` for a refresher.

This example requires no physical hardware, graphical interface, optional
Python package, or output file. It prints values in the terminal and stops
automatically after four seconds.

Create the script
-----------------

:download:`Download the complete script
</downloads/more_complexity/modifier.py>`, or create a file named
``modifier.py`` containing this code:

.. literalinclude:: /downloads/more_complexity/modifier.py
   :language: python
   :start-after: # [modifier-start]
   :end-before: # [modifier-end]

The Generator publishes a constant simulated speed under the ``speed(mm/s)``
label. The :class:`~crappy.modifier.Integrate` Modifier calculates the
corresponding position over time and adds it under ``position(mm)``. The
LinkReader therefore receives both the original speed and the calculated
position.

The Modifier is passed to the Link:

.. code-block:: python

   crappy.link(speed, reader, modifier=integrate_speed)

It changes data only on that Link. Another Link from ``speed`` without this
Modifier would still carry the original labels. When a Link has several
Modifiers, Crappy applies them in the order in which they are provided.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python modifier.py

The terminal displays a speed close to 2 mm/s while the calculated position
increases. A final ``Generator Path exhausted`` warning indicates the planned
end of the example.

Choose when to use a Modifier
-----------------------------

Modifiers suit short transformations such as filtering, scaling, renaming, or
selecting values. Use a separate Block when an operation is slow, needs its own
timing, or has setup and cleanup steps. See
:doc:`../concepts/choosing_custom_object_type` for that choice.

The :doc:`../crappy_docs/modifiers` reference lists the built-in Modifiers. If
none matches the required calculation, see
:doc:`custom_modifier`.

Continue with :doc:`generator_conditions` to make a command change when a
measured label reaches a limit.
