.. _concepts-blocks-links-labels:

=========================
Blocks, Links, and labels
=========================

A Crappy test is assembled from **Blocks** that perform tasks and **Links**
that carry data from one Block to another. This page explains how to read and
design that connection graph.

Blocks perform the work
-----------------------

Each :class:`~crappy.blocks.Block` has one responsibility, such as acquiring
measurements, generating commands, controlling an actuator, displaying values,
or recording data. A script creates the Blocks, connects them, and calls
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`. Crappy then runs
each Block independently and coordinates the start and end of the test.

Keeping responsibilities separate lets a script combine existing Blocks in a
new way. For example, the same acquisition Block can send measurements to a
recorder, a live display, and a feedback controller.

Links carry labeled dictionaries
--------------------------------

A regular :class:`~crappy.links.Link` is a one-way connection. Its source Block
sends dictionaries and its destination Block receives them. Each dictionary
maps a **label** to a value:

.. code-block:: python

   {"t(s)": 0.25, "force(N)": 41.2}

Labels identify values as they move through the test. A downstream Block uses
the labels it needs and can ignore the others. Label spelling is significant,
so the sending and receiving Blocks must agree on names such as ``t(s)`` and
``force(N)``.

Calling :func:`crappy.link` creates a Link in the direction given by its first
two arguments:

.. code-block:: python

   crappy.link(acquisition, recorder)

The acquisition Block can now send data to the recorder. Creating the reverse
connection requires another Link.

Modifiers transform data in transit
-----------------------------------

A :class:`~crappy.modifier.Modifier` attached to a regular Link receives each
dictionary before it enters that Link. It can change the dictionary or return
``None`` to discard it. When several Modifiers are attached, they run in the
order in which they were provided.

For example, if ``scale_force`` is a callable Modifier:

.. code-block:: python

   crappy.link(acquisition, recorder, modifier=scale_force)

Modifiers are suitable for small transformations such as renaming, scaling,
filtering, or selecting values. A calculation that is slow or has its own
lifecycle is clearer as a dedicated Block. The :doc:`../tutorials/modifiers`
tutorial demonstrates a Modifier that adds a calculated label.

Receiving rates and missing data
--------------------------------

A regular Link is a live communication channel, not permanent storage. A
consumer can read the oldest available dictionary, keep only the newest one,
or read the currently available dictionaries as a group. In particular,
methods that request only the newest value deliberately discard older unread
values.

If a producer continually sends faster than a consumer can receive, the Link's
underlying buffer can fill. On Linux, a new dictionary is discarded when the
buffer is full. On other operating systems, sending can wait for buffer space.
Links are not meant to store data, they are meant to transfer it between
Blocks. Send the data to save to a Recorder or another storage Block as part of
the test.

Connections are checked before the test
---------------------------------------

Crappy records Blocks and Links as a directed graph while the script creates
them. It rejects duplicate Block names and duplicate Link names. A second
regular Link in the same direction between the same two Blocks is also rejected
unless it is created with ``allow_parallel=True``. This explicit option is
useful when the two Links apply different Modifiers.

Regular Links may form a feedback loop. The Blocks in that loop must still be
able to start before the first feedback value exists. A receiving Block should
therefore handle the absence of new data instead of assuming that every loop
iteration receives a dictionary.

Image connections have additional graph rules because they use a different
transport. The :doc:`../crappy_docs/links` reference lists those rules.

An example Block graph
----------------------

.. graphviz:: ../diagrams/crappy_process_graph.dot
   :alt: A Crappy graph in which a Generator target and IOBlock feedback enter a PID Block, the PID commands a Machine Block, and the IOBlock also sends measurements to a Recorder.
   :caption: A small acquisition, feedback-control, and recording graph. Rounded boxes inside the enclosure are Blocks. Arrows labeled Link carry dictionaries between Blocks; dashed arrows connect the graph to the sensor and actuator through device interactions.

In this graph, the Generator publishes a target and the IOBlock publishes the
measured feedback. The PID Block compares those values and sends a command to
the Machine, which drives the actuator. The IOBlock also sends measurements to
the Recorder. The main script constructs these objects and starts the test,
while Crappy coordinates their execution.

The fact that Blocks currently run in separate operating-system processes is
an implementation detail that matters when writing custom Blocks. The stable
user-facing model is that Blocks run independently and exchange dictionaries
only through their Links. See :doc:`../architecture` for the implementation
model.

API reference
-------------

- :func:`crappy.link` creates a regular Link.
- :class:`crappy.links.Link` documents its receive methods.
- :class:`crappy.modifier.Modifier` is the base class for reusable Modifiers.
- :func:`crappy.display_graph` renders the graph created by a script.
