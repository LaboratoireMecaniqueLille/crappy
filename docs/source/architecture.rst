.. _architecture:

========================
Contributor architecture
========================

This page describes Crappy's current runtime implementation for contributors
and maintainers. It assumes that you already know the public models described
in :doc:`concepts`.

The following pages define the behavior application code should rely on:

- :doc:`concepts/blocks_links_labels`
- :doc:`concepts/lifecycle_shutdown`
- :doc:`concepts/regular_links_and_image_links`
- :doc:`concepts/image_pipelines`

The classes and synchronization objects described below are implementation
details unless the API reference explicitly documents them as public. They may
change while the public behavior remains the same.

Runtime ownership
-----------------

The Python interpreter that runs the user's script is the **main process**.
It constructs the graph and coordinates its execution. Every
:class:`~crappy.blocks.Block` is also a
:class:`multiprocessing.Process` and runs as a child process after preparation
begins.

.. list-table:: Runtime ownership
   :header-rows: 1
   :widths: 25 35 40

   * - Owner
     - Main responsibilities
     - Important state
   * - Main process
     - Build and validate the graph, create shared runtime objects, start the
       Blocks, release their common start, collect logs, and perform final
       cleanup.
     - Block registry, LinkGraph, shared events, readiness barrier, shared
       ``t0``, logging queue, and optional image metadata manager.
   * - Block process
     - Run one Block's ``prepare``, ``begin``, repeated ``loop``, and ``finish``
       methods.
     - Its Link endpoints, runtime settings, device or file handles, and local
       state.
   * - Image-producing VisionBlock
     - Publish images and metadata for its downstream ImageLinks.
     - One owned shared-memory image buffer and synchronization state shared by
       all its outgoing ImageLinks.
   * - Image-consuming VisionBlock
     - Copy and handle newer images from its upstream sources.
     - One local image copy and last-seen transport identifier per incoming
       ImageLink.
   * - All-in-one Camera Block
     - Acquire images and manage its optional all-in-one image workers.
     - Camera object, shared latest-frame state, and internal CameraProcesses.

Package organization
--------------------

The public object families live in packages matching their roles:

- ``crappy.blocks`` contains Blocks, VisionBlocks, Generator Paths, and the
  all-in-one CameraProcess family.
- ``crappy.links`` contains regular Links, ImageLinks, and the LinkGraph.
- ``crappy.actuator``, ``crappy.inout``, and ``crappy.camera``
  contain hardware integrations.
- ``crappy.modifier`` contains transformations applied by regular Links.
- :mod:`crappy.tool.camera_config` contains Camera configuration windows.
- :mod:`crappy.tool.image_processing` contains reusable image-processing
  algorithms that are independent of the Block scheduling layer.

The ``ext`` directory beside the ``crappy`` package contains historical C++
extensions used by some objects. Their current compatibility is not verified.
Building one requires a source installation, a compiler, and any required
system libraries or device drivers.

Subclass registration
---------------------

Block, Actuator, InOut, Camera, Modifier, and Generator Path base classes use
``__init_subclass__`` to register their children by class name. A duplicate
name in the same object family raises a definition error when Python creates
the class. The owning Block can later resolve a hardware integration or Path
from the string supplied by the user.

This registration is an implementation convenience, not a reason to combine
unrelated responsibilities in one class. The
:doc:`concepts/choosing_custom_object_type` guide explains which custom object
type matches a device or task.

Construction and graph registration
-----------------------------------

Before startup, all code runs in the main process. Constructing a Block adds
the instance to the Block registry and adds a node to the module-level
:class:`~crappy.links.LinkGraph`. Constructing a regular Link or ImageLink adds
a directed edge and associates the connection with its source and destination
Blocks.

The graph rejects name collisions and invalid ImageLink topology immediately.
Renaming a Block before startup updates its node and incident edges. The
:meth:`~crappy.blocks.Block.reset` class method clears the Block registry and
graph together so that their state cannot diverge between tests.

A regular Link creates its pipe during construction. An ImageLink initially
stores only its graph relationship and placeholders for shared image state.
The image transport is completed during preparation, once Crappy knows the
entire graph.

Startup coordination
--------------------

:meth:`~crappy.blocks.Block.start_all`, exposed as
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`, calls these class
methods in order:

1. :meth:`~crappy.blocks.Block.prepare_all`
2. :meth:`~crappy.blocks.Block.renice_all`
3. :meth:`~crappy.blocks.Block.launch_all`

The three aliases :ref:`crappy.prepare()
<crappy_docs/aliases:crappy.prepare()>`, :ref:`crappy.renice()
<crappy_docs/aliases:crappy.renice()>`, and :ref:`crappy.launch()
<crappy_docs/aliases:crappy.launch()>` expose the same phases separately.

``prepare_all``
+++++++++++++++

``prepare_all`` configures the main logger and creates the synchronization
objects shared with all Blocks. These currently include:

- a readiness barrier with one participant per Block plus the main process
- a shared floating-point value initialized to ``-1`` for the common ``t0``
- events for start, pause, stop, runtime exceptions, and keyboard interruption
- a logging queue and its coordinating thread

References to these objects are assigned to every Block before the processes
start. Block-specific logging levels are also reconciled with the global
logging level at this point.

If at least one VisionBlock is present, preparation also validates that the
Block registry and LinkGraph contain the same names. It walks the ImageLink
graph to route downstream configuration requests to their CameraSource. Each
accepted request receives a dedicated one-way configuration pipe.

Crappy then creates the manager-backed dictionaries used for image metadata
and image-format information. Each image producer creates the buffer name,
lock, readiness event, and transport counter shared by its outgoing
ImageLinks. The actual shared-memory segment is created later in the producing
Block, after its final image shape and data type are known.

Finally, ``prepare_all`` starts every Block process. The main process closes
its copies of configuration-pipe endpoints after startup. Under the ``fork``
start method, each child also closes inherited endpoints that belong to other
Blocks so that an unintended open copy cannot hide an end-of-file condition.

``renice_all``
+++++++++++++++

On Linux and macOS, ``renice_all`` applies each Block's requested niceness.
It can optionally permit negative niceness values when the script has suitable
privileges. Windows does not provide the same operation, so this phase does
nothing there.

``launch_all``
+++++++++++++++

The main process waits on the readiness barrier with the Blocks. After every
participant arrives, it records the current time in the shared ``t0`` value
and sets the start event. It then waits for a Block to finish. Normal completion
of any Block begins the coordinated shutdown of the remaining graph.

Block process sequence
----------------------

:meth:`~crappy.blocks.Block.run` implements the process-side lifecycle. It
performs these steps:

1. Configure the Block logger and close unrelated inherited configuration
   endpoints where required.
2. Call :meth:`~crappy.blocks.Block.prepare`.
3. Wait at the shared readiness barrier.
4. Wait for the main process to set ``t0`` and the start event.
5. Call :meth:`~crappy.blocks.Block.begin` once.
6. Enter :meth:`~crappy.blocks.Block.main`, which calls
   :meth:`~crappy.blocks.Block.loop` repeatedly until the stop event is set.
7. Set the shared stop event and call :meth:`~crappy.blocks.Block.finish` in a
   ``finally`` block.

``main`` also applies the Block's target frequency and pause behavior. A
paused Block continues frequency regulation but does not call ``loop`` while
the pause event applies to it.

A VisionBlock extends ``prepare`` to resolve its configuration responses and
image transports. An image producer creates its shared-memory segment and
publishes its format. Each consumer waits for that state, attaches to the
segment, and allocates its local image copy. These waits periodically inspect
the stop and barrier state so that another Block's preparation failure does not
leave a consumer waiting indefinitely.

Error propagation and cleanup
-----------------------------

A Block that fails during ``prepare`` aborts the readiness barrier. Other
participants receive the broken-barrier state instead of waiting forever. A
runtime exception sets the shared exception state. Keyboard interruption has a
separate shared state so that it can be reported as such after cleanup.

The ``finally`` section of ``Block.run`` sets the shared stop event and attempts
to call ``finish``. An error in ``finish`` is logged and recorded as another
runtime failure. A process that must be terminated externally cannot complete
this path, which is why hardware safety cannot depend only on ``finish``.

The main cleanup routine sets the stop event and gives the Blocks a limited
time to exit. The current timeout is three seconds. It terminates processes
that remain alive, then shuts down the optional image manager and logging
thread. Finally, :meth:`~crappy.blocks.Block.reset` clears the registries,
graph, shared-object references, and lifecycle flags.

Unless ``no_raise`` was selected, the main process raises after cleanup when a
runtime exception, keyboard interruption, or incomplete shutdown was recorded.

Regular Link internals
----------------------

A :class:`~crappy.links.Link` wraps a :func:`multiprocessing.Pipe` and treats
it as a one-way dictionary channel. Before sending, it applies its Modifiers in
order. Each Modifier receives a deep copy of the current dictionary. Returning
``None`` stops that send, while returning a non-dictionary raises a Link data
error.

On Linux, the Link checks whether the pipe is writable without blocking. If
the pipe is full, the new dictionary is discarded and warning messages are
rate-limited. Other operating systems call the pipe's send operation directly.
The receiving methods determine whether to retrieve one, the newest, or all
currently buffered dictionaries.

Application behavior must follow the public loss semantics documented in
:doc:`concepts/regular_links_and_image_links`, not the current pipe layout.

ImageLink internals
-------------------

An image-producing VisionBlock owns one
:class:`multiprocessing.shared_memory.SharedMemory` segment for all its
outgoing ImageLinks. Manager-backed dictionaries hold the current metadata and
the negotiated image shape and data type. A reentrant lock protects reads and
writes, an event announces that the buffer exists, and a shared counter marks
each published buffer state.

``send_img`` checks the metadata and image format, then acquires the lock. It
replaces the metadata, copies the NumPy array, and increments the transport
counter before releasing the lock. Each consumer keeps the last counter value
it copied. ``receive_imgs`` acquires the same lock and copies both metadata and
image only when that value changed.

The counter is a transport implementation detail. It is distinct from the
public ``ImageUniqueID`` stored in image metadata. A consumer must use the
metadata belonging to its local image copy and must allow skipped identifiers.

During normal shutdown, consumers close their handles without unlinking the
segment. The producing VisionBlock owns the segment and closes and unlinks it.

Image configuration requests
----------------------------

A downstream VisionBlock can override ``request_config`` to ask an upstream
CameraSource for specialized interactive configuration. During
``prepare_all``, Crappy validates and copies each request, then assigns one end
of a one-way pipe to the source and the other to the requester.

The CameraSource handles accepted requests sequentially during preparation and
sends each result back. The requesting VisionBlock receives the explicit
result before constructing its processing helpers. A required request must be
answered, while an optional request may return ``None``. This exchange happens
before the common start and is separate from ImageLink frame transport.

.. _architecture-all-in-one-camera:

All-in-one Camera internals
---------------------------

The :class:`~crappy.blocks.Camera` Block owns the Camera object and performs
acquisition. Depending on its options and subclass, it can create internal
:class:`~crappy.blocks.camera_processes.CameraProcess` children for processing,
display, and recording. These children are processes but are not Blocks or
nodes in the public LinkGraph.

The Camera Block creates a shared image array, metadata dictionary, lock,
readiness barrier, and stop event for these workers. Acquisition replaces the
latest shared frame. Each worker checks for a new image and runs at its own
rate. The Camera Block starts, monitors, and stops the workers as part of its
own Block lifecycle.

The Camera configuration window runs before these workers start. A processing
CameraProcess can receive explicit configuration values through the paired
``CameraConfig.get_config`` and ``CameraProcess.set_config`` hooks, then create
its private helpers in ``CameraProcess.init``.

This internal topology explains how the all-in-one interface works. It does
not change its support status. VisionBlocks are recommended for new image
pipelines. The all-in-one Camera Blocks remain supported and are not planned
for deprecation.
