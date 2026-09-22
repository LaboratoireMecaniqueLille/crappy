.. _tutorial-custom-block:

=====================
Create a custom Block
=====================

This tutorial has one outcome: create a Block that collects several simulated
measurements and publishes their average at its own rate.

Prerequisites
-------------

- Complete :doc:`quickstart`.
- Know that Links carry dictionaries identified by labels. See
  :doc:`../concepts/blocks_links_labels` for a refresher.

The example requires no physical hardware, graphical interface, optional
Python package, or output file. It prints results in the terminal and stops
automatically after four seconds.

Define the Block
----------------

:download:`Download the complete script
</downloads/custom_objects/custom_block_average.py>`, or create a file named
``custom_block.py``. Its custom Block is:

.. literalinclude:: /downloads/custom_objects/custom_block_average.py
   :language: python
   :start-after: # [custom-block-class-start]
   :end-before: # [custom-block-class-end]

Every custom Block inherits from :class:`~crappy.blocks.meta_block.block.Block`
and calls ``super().__init__()``. This Block stores its input and output
labels, then sets its target work rate through ``self.freq``.

Crappy calls ``loop()`` repeatedly. ``recv_all_data()`` returns every unread
value grouped by label. The method may return no force values, so the Block
checks the list before calculating its mean. ``send()`` publishes a dictionary
to every outgoing Link. The timestamp uses the common test start time
``self.t0``.

Connect it to other Blocks
--------------------------

The complete script creates the custom Block like any built-in Block:

.. literalinclude:: /downloads/custom_objects/custom_block_average.py
   :language: python
   :start-after: # [custom-block-use-start]
   :end-before: # [custom-block-use-end]

The Generator publishes ``force(N)`` about twenty times per second. The custom
Block works twice per second, so each result usually combines several force
values. It publishes the average as ``mean_force(N)`` and reports how many
samples were included. Exact counts can vary with timing.

Run the example
---------------

From the directory containing the downloaded file, run:

.. code-block:: shell-session

   python custom_block.py

The terminal displays one batch result roughly every half second. A final
``Generator Path exhausted`` warning indicates the planned end of the example.

Choose the receive method
-------------------------

The appropriate receive method depends on what the calculation needs:

- ``recv_last_data()`` keeps only the newest available value for each label.
- ``recv_all_data()`` keeps all unread values, as required for this average.
- ``recv_all_data_raw()`` additionally keeps incoming Links separate. Use it
  when several sources may use the same labels or their individual timestamps
  must remain associated with the correct source.

All three methods can return no new data during a given call. Keep ``loop()``
short and allow it to return when there is nothing to do.

Add setup and cleanup when needed
---------------------------------

This example needs only ``loop()``, but a custom Block can also define these
lifecycle methods:

- ``prepare()`` opens files, connections, or other resources before the test.
- ``begin()`` performs a one-time action when the shared test clock is ready.
- ``finish()`` flushes and closes resources when the test ends.

Store configuration and validate arguments in ``__init__()``, but acquire
resources in ``prepare()``. Make ``finish()`` tolerate a partially completed
setup. Avoid calls in ``loop()`` that can wait forever, the method must return
regularly so the test can stop cleanly.

Use a Modifier instead when the task is only a quick transformation of each
dictionary on one Link. Use a VisionBlock when receiving or publishing images.
The :doc:`../concepts/choosing_custom_object_type` guide covers this choice,
and :doc:`../concepts/lifecycle_shutdown` describes the full lifecycle.

See :class:`~crappy.blocks.meta_block.block.Block` for the complete custom
interface, including the available Link methods and validated Block settings.
