.. _concepts-lifecycle-shutdown:

======================
Lifecycle and shutdown
======================

Every Block follows the same lifecycle. Understanding where initialization,
repeated work, and cleanup belong is especially useful when writing a custom
:class:`~crappy.blocks.meta_block.block.Block` or diagnosing why a test did not
start.

Lifecycle at a glance
---------------------

.. graphviz:: ../diagrams/block_lifecycle.dot
   :alt: The Block lifecycle from construction through preparation, synchronized start, repeated loop calls, finish, and final cleanup.
   :caption: Gray, gold, and blue boxes identify actions performed by the main script, the Crappy coordinator, and each Block. White boxes mark synchronization and the decision to repeat or stop after an iteration.

The same sequence in text is:

.. list-table:: Block lifecycle
   :header-rows: 1
   :widths: 18 25 57

   * - Stage
     - Where it runs
     - Observable behavior
   * - Construction
     - Main script
     - The script creates Blocks and Links and sets their options.
   * - Preparation
     - Crappy and each Block
     - Crappy creates the runtime state, starts the Blocks, and calls each
       Block's ``prepare`` method.
   * - Ready barrier
     - Crappy and all Blocks
     - No Block begins the test until every Block has finished preparing.
   * - Common start
     - Crappy
     - Crappy records the shared ``t0`` and releases all Blocks.
   * - Initial action
     - Each Block
     - The Block calls ``begin`` once.
   * - Repeated work
     - Each Block
     - The Block calls ``loop`` until the test is asked to stop.
   * - Stop propagation
     - Crappy and all Blocks
     - A stop request or failure tells every Block to leave its loop.
   * - Block cleanup
     - Each Block
     - The Block calls ``finish`` to close hardware, files, and other
       resources.
   * - Final cleanup
     - Crappy
     - Crappy waits for the Blocks, releases shared resources, and resets its
       runtime state.

Construction happens before the test starts
-------------------------------------------

The script first constructs every Block and Link. A custom Block's
``__init__`` method should validate arguments and store configuration. Opening
hardware, network connections, or output files normally belongs in
:meth:`~crappy.blocks.meta_block.block.Block.prepare`, after the Block has
started in its own runtime context.

Constructing the objects also builds the connection graph. Invalid names and
connection structures are therefore reported before the test starts. See
:doc:`blocks_links_labels` for the graph rules.

Preparation and the common start
--------------------------------

Calling :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` first runs
:meth:`~crappy.blocks.meta_block.block.Block.prepare_all`. Crappy creates the
state shared by the Blocks, starts them, and calls each Block's
:meth:`~crappy.blocks.meta_block.block.Block.prepare` method. Typical
preparation work includes opening a device, establishing a connection, or
creating an output file.

After preparing, every Block waits at a synchronization barrier. If one Block
fails during preparation, the barrier is released as an error so that the
others do not wait indefinitely. The test proceeds only after every Block and
the coordinator are ready.

Crappy then records a common start timestamp, available through
:attr:`~crappy.blocks.meta_block.block.Block.t0`, and releases the Blocks. This
timestamp gives all Blocks the same reference for elapsed time. It does not
promise that every Block executes its next instruction at exactly the same
instant.

Running the test
----------------

Once released, each Block calls
:meth:`~crappy.blocks.meta_block.block.Block.begin` once. This hook is for work
that needs the shared start time or must happen immediately before repeated
operation.

The Block then calls :meth:`~crappy.blocks.meta_block.block.Block.loop`
repeatedly. A target frequency can limit how often the loop runs, but it is not
a real-time guarantee. The work performed by the Block, the operating system,
hardware, and other load on the computer all affect the achieved frequency.

Stopping and exception handling
-------------------------------

The test stops when a custom Block calls its ``stop`` method, a Block finishes
naturally, the user presses :kbd:`Control-c`, or an error interrupts normal
execution. Crappy shares the stop request with every Block so that they can
leave their loops.

An unexpected exception in one Block is recorded and causes the whole test to
stop. By default, Crappy raises an exception in the main script after cleanup.
The ``no_raise`` option of :ref:`crappy.start()
<crappy_docs/aliases:crappy.start()>` can suppress that final exception, but it
does not make the failed test successful. Use it only when the calling program
handles the failure state explicitly.

Cleanup belongs in ``finish``
-----------------------------

Each Block normally calls :meth:`~crappy.blocks.meta_block.block.Block.finish`
whether it stops normally or because an error occurred. A custom Block should
use this hook to return hardware to a safe state, close devices and network
connections, flush and close files, and release graphical resources.

``finish`` should tolerate partially completed preparation. For example, check
that a device was opened before trying to close it. Crappy may forcibly
terminate a Block that does not respond during shutdown, so software cleanup
cannot be the only protection for hazardous equipment. Use suitable physical
safety systems and hardware limits independently of Crappy.

After the Blocks finish, Crappy releases framework-owned resources and resets
the Block registry and connection graph. A later test in the same Python
session can then construct a new graph.

Public behavior and implementation details
------------------------------------------

The lifecycle hooks and their order are the public model custom Blocks should
follow. The synchronization primitives, worker implementation, cleanup
timeouts, and operating-system start methods are implementation details. They
are documented in the :doc:`../architecture` guide because they may change
without altering the lifecycle above.

API reference
-------------

- :class:`crappy.blocks.meta_block.block.Block` defines the lifecycle hooks.
- :meth:`crappy.blocks.meta_block.block.Block.prepare_all` prepares and starts
  all Blocks.
- :meth:`crappy.blocks.meta_block.block.Block.launch_all` releases prepared
  Blocks and waits for the test to finish.
- ``Block.stop`` requests a normal shutdown.
