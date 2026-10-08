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
:class:`~crappy.blocks.meta_block.block.Block` is also a
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
:meth:`~crappy.blocks.meta_block.block.Block.reset` class method clears the
Block registry and graph together so that their state cannot diverge between
tests.

A regular Link creates its pipe during construction. An ImageLink initially
stores its graph relationship and placeholders for shared image state.
The image transport is completed during preparation, once Crappy knows the
entire graph.

Startup coordination
--------------------

:meth:`~crappy.blocks.meta_block.block.Block.start_all`, exposed as
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`, calls these class
methods in order:

1. :meth:`~crappy.blocks.meta_block.block.Block.prepare_all`
2. :meth:`~crappy.blocks.meta_block.block.Block.renice_all`
3. :meth:`~crappy.blocks.meta_block.block.Block.launch_all`

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

Preparation validates that the Block registry and LinkGraph contain the same
names for all Blocks. If at least one VisionBlock is present, it walks the
ImageLink graph to route downstream configuration requests to their
CameraSource. Each accepted request receives a dedicated one-way configuration
pipe.

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

The main process waits on the readiness barrier with the Blocks. A temporary
watchdog thread monitors Block process sentinels during this wait, aborts the
barrier if a Block exits, and stops before the start event is set. After every
participant arrives, the main process records the current time in the shared
``t0`` value and sets the start event. It then waits for a Block to finish.
Normal completion of any Block begins the coordinated shutdown of the remaining
graph.

Block process sequence
----------------------

:meth:`~crappy.blocks.meta_block.block.Block.run` implements the process-side
lifecycle. It performs these steps:

1. Configure the Block logger and close unrelated inherited configuration
   endpoints where required.
2. Call :meth:`~crappy.blocks.meta_block.block.Block.prepare`.
3. Wait at the shared readiness barrier.
4. Wait for the main process to set ``t0`` and the start event.
5. Call :meth:`~crappy.blocks.meta_block.block.Block.begin` once.
6. Enter :meth:`~crappy.blocks.meta_block.block.Block.main`, which calls
   :meth:`~crappy.blocks.meta_block.block.Block.loop` repeatedly until the stop
   event is set.
7. Set the shared stop event and call
   :meth:`~crappy.blocks.meta_block.block.Block.finish` in a ``finally`` block.

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
time to exit. The current timeout is three seconds. It terminates survivors,
then kills any still alive, allowing one second for each escalation stage,
before shutting down the optional image manager and logging thread. The
logging thread is asked to stop and given one second to exit, a missed deadline
is recorded as a shutdown failure. Finally,
:meth:`~crappy.blocks.meta_block.block.Block.reset` clears the registries,
graph, shared-object references, and lifecycle flags.

Unless ``no_raise`` was selected, the main process raises after cleanup when a
runtime exception, keyboard interruption, or incomplete shutdown was recorded.

Regular Link internals
----------------------

A :class:`~crappy.links.link.Link` wraps a :func:`multiprocessing.Pipe` and
treats it as a one-way dictionary channel. Before sending, it applies its
Modifiers in order. Each Modifier receives a deep copy of the current
dictionary. Returning ``None`` stops that send, while returning a
non-dictionary raises a Link data error.

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

Each receiver also owns a Condition notified by all its image sources. A
positive ``receive_imgs(timeout=...)`` waits for a new frame on any input or a
shared stop/error flag.

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

.. _architecture-camera-configuration:

Camera configuration
--------------------

Interactive camera configuration lets users adjust camera settings and select
the image regions needed by a processing algorithm before the experiment
starts. The classes in :mod:`crappy.tool.camera_config` support both
:class:`~crappy.blocks.vision.CameraSource` and the all-in-one
:class:`Camera Blocks <crappy.blocks.Camera>`.

Where configuration runs
++++++++++++++++++++++++

The acquisition Block opens the
:class:`~crappy.camera.meta_camera.camera.Camera` first, then runs its
configuration window in its own process during preparation. The preview and
setting controls therefore use the same camera that will acquire images during
the experiment. Closing the window leaves that camera open and the configured
settings applied.

For an image pipeline, this also allows a downstream processing Block to ask
:class:`~crappy.blocks.vision.CameraSource` for a specialized window. For
example, a correlation processor can request a region-of-interest selection on
the source's live image without opening the camera itself. The source handles
the interaction and returns the selection before processing begins. As with
other preparation work, configuration must finish before the common start.

Shared behavior and GUI backends
++++++++++++++++++++++++++++++++

The separation between
:mod:`camera_config.base <crappy.tool.camera_config.base>`
and the :mod:`tkinter <crappy.tool.camera_config.tkinter>` and
:mod:`pyqt <crappy.tool.camera_config.pyqt>` packages keeps processing rules
independent of the GUI implementation. The abstract
:class:`~crappy.tool.camera_config.base.camera_config.CameraConfig` provides
common preview and setting logic, while
:class:`~crappy.tool.camera_config.base.camera_config_boxes.CameraConfigBoxes`
adds rectangular selection. Specialized classes build on these foundations:
for example,
:class:`~crappy.tool.camera_config.base.dis_correl_config.DISCorrelConfig`
defines what makes a valid correlation region and how to export it.

A GUI window combines that shared behavior with a backend's widgets,
image rendering, and event handling. A change to the correlation selection
rules therefore belongs in the shared class, whereas a change to how a Qt
control looks or responds belongs in the Qt implementation. Switching backends
does not require a different camera driver or processing algorithm, and adding
a backend does not require rewriting the selection rules. Reusable image and
selection tools live in
:mod:`camera_config.config_tools <crappy.tool.camera_config.config_tools>`.

Camera drivers define the available settings and how to read or change them,
the windows turn those definitions into controls. When an edit is applied,
the shared logic reads back the value accepted by the camera rather than
assuming the requested value was accepted unchanged. It also refreshes any
other controls affected by that change. This keeps the displayed settings
consistent with the device without putting GUI-specific code in drivers.

The acquisition Block's ``configurator`` class attribute identifies the window
class, or maps backend names to window classes. In the latter case,
``config_backend`` selects the implementation. A custom Block can replace this
attribute to use a specialized window while keeping the existing acquisition
and configuration workflow.

Returning results and stopping configuration
++++++++++++++++++++++++++++++++++++++++++++

Once the user completes configuration,
:meth:`get_config() <crappy.tool.camera_config.base.camera_config.\
CameraConfig.get_config>` exports the information needed by processing, such as
the coordinates of the selected correlation region. This is data rather than a
reference to the window: processing can run in another process without
depending on the GUI toolkit. :class:`~crappy.blocks.vision.CameraSource` sends
the result to the requesting :class:`~crappy.blocks.vision.block.VisionBlock`,
an all-in-one :class:`Camera Block <crappy.blocks.Camera>` passes it to
:meth:`CameraProcess.set_config() <crappy.blocks.camera_processes.\
CameraProcess.set_config>`. Both paths let the processor initialize with the
user's selection before it handles experiment images.

Completing configuration and cancelling preparation are different operations.
A normal close checks that the selection is usable by the processor, but a
shutdown request bypasses that check: an experiment being stopped must not
remain stuck waiting for a region to be selected. Both backends close their
window and release their configuration resources in that case. Errors raised
during GUI callbacks are also passed back to the owning Block after cleanup,
so they participate in Crappy's preparation error handling instead of leaving
the window running independently.

For extension examples, see :doc:`tutorials/custom_camera_configuration`.
The :doc:`crappy_docs/tools` reference describes the individual class contracts
and hooks.

.. _architecture-all-in-one-camera:

All-in-one Camera internals
---------------------------

The :class:`~crappy.blocks.Camera` Block owns the Camera object and performs
acquisition. Depending on its options and subclass, it can create internal
:class:`~crappy.blocks.camera_processes.CameraProcess` children for processing,
display, and recording. These children are processes but are not Blocks or
nodes in the public LinkGraph.

The Camera Block creates a shared image array, metadata dictionary, a separate
lock and Condition per worker, readiness barrier, and stop event. Acquisition
replaces the latest shared frame and notifies the workers, which use timed
Condition waits for new images or shutdown and run at their own rate. The
Camera Block starts, monitors, and stops the workers as part of its own Block
lifecycle.

The Camera configuration window runs before these workers start. A processing
CameraProcess can receive explicit configuration values through the paired
``CameraConfig.get_config`` and ``CameraProcess.set_config`` hooks, then create
its private helpers in ``CameraProcess.init``.

This internal topology explains how the all-in-one interface works. It does
not change its support status. VisionBlocks are recommended for new image
pipelines. The all-in-one Camera Blocks remain supported and are not planned
for deprecation.
