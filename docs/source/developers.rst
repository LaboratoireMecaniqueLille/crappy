======================
Developers information
======================

.. role:: py(code)
  :language: python
  :class: highlight

Contributing to Crappy
----------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

If you want to help developing Crappy with us, we'll be more than happy to
welcome you in the community ! Here you'll find some practical information on
**how Crappy works under the hood, and a few guidelines for contributors**.

If you brought an improvement to your own version of Crappy, and you think it's
worth sharing with the community, don't hesitate to `create a pull request
<https://github.com/LaboratoireMecaniqueLille/crappy/compare>`_ on GitHub ! If
you do so, please enforce the following rules :

- Follow `PEP8 <https://peps.python.org/pep-0008/>`_ as much as possible,
  except for the indents that we chose to lower from 4 to 2 spaces for
  compactness.

- Use the `Google style <https://google.github.io/styleguide/pyguide.html>`_
  for docstrings. Please comment and document your code extensively, and
  update the source of the documentation if needed.

- Use relevant and meaningful titles and descriptions for your commits.
  Starting from v2.0.0, the rules `described here
  <https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/>`_
  should be used for commit messages.

The development branch of Crappy is called `develop
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/develop>`_, and is
the one on which you should commit. Starting from v2.0.0, the `master branch
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master>`_ is never
directly committed to.

Technical description of Crappy
-------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

.. note::
  This is a very simplified overview of how the module actually works. Only the
  main ideas are presented, and many technical aspects are omitted. Reading the
  code remains the only way to truly understand it !

Crappy is written as a pure-Python module, and is divided in a number of
submodules. The breakout of the submodules follow the logical organization of
the objects, according to their types. The different types of objects are
presented in this section.

Blocks and Links
++++++++++++++++

Regular Blocks and Links
""""""""""""""""""""""""

The Blocks are base objects that have each a specific function and are
instantiated by users to achieve a given overall behavior in their scripts.
They exchange small dictionaries through directed
:class:`~crappy.links.Link` objects. Multiple Links can enter or leave a Block,
and a second regular Link between the same ordered pair must explicitly enable
the ``allow_parallel`` option.

Under the hood, every Block is a child of the base
:class:`~crappy.blocks.Block`, which is itself a child of
:obj:`multiprocessing.Process`. Each Block thus runs in its own separate
process, which is the solution we chose for achieving an optimal performance of
the module. The main downsides of this architecture are a high complexity, and
potential difficulties to ensure a smooth termination of all the processes. A
detailed description of the objects and strategies used to achieve a clean
parallelization can be found in the
:ref:`next section <Detailed runtime sequence of Crappy>`.

As Blocks live each in a separate process, sharing data between each other is
not straightforward. In Crappy, data can be sent from one Block to another only
if they have first been linked by a :class:`~crappy.links.Link`. Behind each
Link is a :obj:`multiprocessing.Pipe`, a low-level object that carries the
data. In addition to instantiating the Pipe, the Link object also provides
methods for the :class:`~crappy.blocks.Block` to use when sending data.

Connection graph
""""""""""""""""

Crappy records this topology in a module-level
:class:`~crappy.links.LinkGraph`. Constructing a Block registers a graph node,
and constructing a Link or :class:`~crappy.links.ImageLink` registers an edge.
Block and Link names are checked for uniqueness immediately. The graph also
rejects parallel ImageLinks and cycles consisting only of ImageLinks, while
regular Links remain free to form feedback loops. Renaming a Block before it
starts updates its node and every incident edge.

The graph is useful for more than validation. It can be rendered with
:func:`crappy.display_graph`, and the preparation sequence traverses its image
subgraph to find image sources and route configuration requests to them. The
module-level graph is cleared together with the Block registry by
:meth:`crappy.blocks.Block.reset`.

VisionBlocks and ImageLinks
"""""""""""""""""""""""""""

One major downside of the Pipes is that they can overflow, in which case data
from the sender Block is simply discarded when trying to send it. This behavior
is especially inconvenient for sending images because they are so large. The
:class:`~crappy.blocks.vision.VisionBlock` family therefore transfers images
through :class:`~crappy.links.ImageLink` objects backed by
:class:`multiprocessing.shared_memory.SharedMemory`.

An image-producing VisionBlock owns one shared NumPy buffer for all of its
outgoing ImageLinks. A process-safe lock protects each write and read, while
shared dictionaries contain the current metadata and image format. An Event
indicates when the buffer is ready and a shared counter identifies each
published buffer state. :meth:`~crappy.blocks.vision.VisionBlock.send_img`
replaces the buffer's contents atomically. Each consumer's
:meth:`~crappy.blocks.vision.VisionBlock.receive_imgs` copies a newer state to
a local array before processing it. Consequently, a consumer can skip
intermediate frames but cannot observe a partially written image or metadata
belonging to another frame.

Every VisionBlock is a complete Block and thus has its own graph node, Process,
lifecycle, regular Links, and loop-frequency control. Acquisition, processing,
display, and recording can be represented by independent Blocks that consume
the same source at different rates. Image arrays and matching metadata use
ImageLinks, whereas commands, measurements, and overlays continue to use
regular Links. This explicit architecture requires more graph construction
from the user, in exchange for significantly freer composition.

All-in-one Camera Blocks
""""""""""""""""""""""""

The older :class:`~crappy.blocks.Camera` architecture also shares acquired
images without sending them through regular Pipes, but its receiver processes
are not Blocks. They are children of
:class:`~crappy.blocks.camera_processes.CameraProcess` and are managed
internally by one Camera Block in a similar way as the base
:class:`~crappy.blocks.Block` manages all the top-level Blocks. Acquisition,
display, recording, and processing are parallelized, but their topology is
encapsulated inside the owning Camera Block rather than exposed in the script's
Block graph.

This all-in-one architecture remains supported and is not planned for
deprecation. VisionBlocks are recommended for new scripts and custom image
processing because they give each Block fewer responsibilities and allow the
image workflow to be rearranged. The Camera Block remains convenient when its
fixed internal architecture already matches the application.

Shared buffers are not used for ordinary labeled data because they add
complexity that is unnecessary for small dictionaries and require the data
shape and dtype to be known in advance. This is normally straightforward for
images, either from constructor arguments or after Camera configuration, but
not for arbitrary numerical messages.

Actuators, Cameras, InOuts
++++++++++++++++++++++++++

Some of the Blocks rely on specific types of helper object, that they can
drive. It is the case for :

- The :class:`~crappy.blocks.vision.CameraSource` and
  :class:`~crappy.blocks.Camera` Blocks that each drive one
  :class:`~crappy.camera.Camera` object for acquiring images.
- The :class:`~crappy.blocks.IOBlock` Block that drives one
  :class:`~crappy.inout.InOut` object for acquiring data and/or setting outputs
  on hardware.
- The :class:`~crappy.blocks.Machine` Block that drives one or several
  :class:`~crappy.actuator.Actuator` objects for controlling motors and other
  actuators.

The Actuators, Cameras and InOuts are simple classes that do not derive from a
parent class like the Blocks do. They were introduced to implement standardized
ways for the Camera, IOBlock and Machine Blocks to interface with hardware. If
written correctly. all the children of one of these classes implement the same
methods and are seamlessly interchangeable.

In addition to providing a standardized way to integrate hardware in Crappy,
these classes also provide helper methods to their children. For example, the
InOut class implements a way to offset the inputs to zero before the test
starts. Other example, the Camera class provides support for the integration of
the supported camera settings in the
:class:`~crappy.tool.camera_config.CameraConfig` window.

Modifier objects
++++++++++++++++

The :class:`~crappy.modifier.Modifier` objects provide extra flexibility for
fine-tuning the data flowing through the Links without having to modify or
create Blocks. In practice, they are just callables (functions or classes)
stored by a given Link and called each time data is sent through the Link.
These objects are not meant to perform computationally-intensive tasks, as
their call is not parallelized.

C++ extension modules
+++++++++++++++++++++

In the `src` folder of Crappy, you can find next to the module `crappy` another
directory called `ext`. It contains the C++ extensions that were historically
used by some objects in the module. It is very unsure whether these extensions
still work, but they were kept around as a legacy waiting for pure-Python
replacement solution to be added to Crappy. To enable extension module(s), one
has to locally clone Crappy and install it manually with the correct drivers
installed on the machine.

Other objects
+++++++++++++

Crappy is full of other helper objects, that have lower importance compared to
the ones previously described and are not necessarily exposed to the users.
Here is a non-exhaustive list of the main ones, and how they integrate in the
framework.

Generator Paths objects
"""""""""""""""""""""""

The :class:`~crappy.blocks.generator_path.meta_path.Path` objects are used by
the :class:`~crappy.blocks.Generator` Block to create waveforms to send to
downstream Blocks. Just like the InOuts for example, they standardize the
methods of the Paths to make them interchangeable and implement convenient
helper methods. The Paths are a bit less straightforward to use than the
Actuators, Cameras and InOuts, and the possibility for users to create their
own Paths was only recently added.

CameraConfig window
"""""""""""""""""""

Both image architectures can use a
:class:`~crappy.tool.camera_config.CameraConfig` window. This interactive
`tkinter` based GUI allows the user to visualize images acquired by a
:class:`~crappy.camera.Camera` object and adjust its settings. All the code
managing the GUI is stored in :mod:`crappy.tool.camera_config`, where
specialized CameraConfig children and their helper classes are defined. The
base CameraConfig contains the variables, bindings, and traces required for a
feature-rich interface. It even manages a parallel
:class:`~crappy.tool.camera_config.config_tools.HistogramProcess` that
calculates an image histogram in real time.

In the VisionBlock architecture,
:class:`~crappy.blocks.vision.CameraSource` owns the Camera and runs the
configuration windows. A downstream processor can override
:meth:`~crappy.blocks.vision.VisionBlock.request_config` to describe the
specialized configurator and arguments that it needs. Before the Block
processes start, :meth:`~crappy.blocks.Block.prepare_all` uses the ImageLink
graph to copy this request to the source and consumer and gives each copy one
end of a one-way Pipe. CameraSource runs accepted requests sequentially and
sends each resulting tuple back to its requester. Required requests must be
answered, optional requests may receive :obj:`None`. The processor calls
:meth:`~crappy.blocks.vision.VisionBlock.recv_configs` during preparation and
creates its image-processing helpers from the explicit result.

In the all-in-one architecture, the base Camera Block instead owns the entire
configuration workflow. A Camera child selects the appropriate CameraConfig
and creates its processing
:class:`~crappy.blocks.camera_processes.CameraProcess`. After the window
closes, :meth:`crappy.tool.camera_config.CameraConfig.get_config` returns
either :obj:`None` or a tuple of processing-specific values. The Camera Block
unpacks that tuple into
:meth:`crappy.blocks.camera_processes.CameraProcess.set_config` before starting
the CameraProcess. The signatures of these two hooks therefore form a pair.
This exposes only explicit configuration data to the processing process and
lets it create specific helpers later, in its own
:meth:`~crappy.blocks.camera_processes.CameraProcess.init` method.

CameraSetting objects
"""""""""""""""""""""

To standardize the integration of the available settings for a given
:class:`~crappy.camera.Camera` object, the
:class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` helper class
was added to Crappy. It has three children that implement each a specific type
of setting (boolean, integer/float, or choice from a given list). They manage
the getter and the setter for the setting, as well as its integration in the
:class:`~crappy.tool.camera_config.CameraConfig` window. In the base Camera
object, the :meth:`~crappy.camera.Camera.add_bool_setting`,
:meth:`~crappy.camera.Camera.add_scale_setting` and
:meth:`~crappy.camera.Camera.add_choice_setting` methods allow users to
instantiate the desired settings. In addition, the
:meth:`~crappy.camera.Camera.add_trigger_setting` method provides specific
support for the setting that manages the hardware trigger mode, if available on
the camera. And finally, the :meth:`~crappy.camera.Camera.add_software_roi`
method manages the instantiation of 4
:class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting` at once,
for applying a software ROI on the acquired images.

Image processing
""""""""""""""""

In both architectures, the actual processing algorithms are stored in
:mod:`crappy.tool.image_processing`. This separates reusable correlation and
tracking code from the Blocks and multiprocessing code that schedule it. A
:class:`~crappy.blocks.vision.DICVEProcessor`,
:class:`~crappy.blocks.vision.DISCorrelProcessor`, or
:class:`~crappy.blocks.vision.VideoExtensoProcessor` creates the corresponding
tool directly in its own Block process after receiving any requested
configuration. In the all-in-one architecture, a child of
:class:`~crappy.blocks.Camera` instead manages a
:class:`~crappy.blocks.camera_processes.CameraProcess`, which creates and owns
the same kind of tool.

The two VideoExtenso implementations illustrate the different ownership
chains. For :class:`~crappy.blocks.vision.VideoExtensoProcessor`, the
:class:`~crappy.blocks.vision.CameraSource` runs
:class:`~crappy.tool.camera_config.VideoExtensoConfig`, which owns the
:class:`~crappy.tool.camera_config.config_tools.SpotsDetector` used for initial
spot selection. Only the spot boxes and threshold cross the configuration
Pipe. The VideoExtensoProcessor then creates
:class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` in its
own Process, and that tool manages one
:class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker` process
per spot.

For the all-in-one implementation, the public
:class:`~crappy.blocks.VideoExtenso` Block validates the user-facing options
and chooses its helpers. Its
:class:`~crappy.tool.camera_config.VideoExtensoConfig` creates and owns the
:class:`~crappy.tool.camera_config.config_tools.SpotsDetector` used for initial
spot selection, then exports only the spot boxes and threshold. The
:class:`~crappy.blocks.camera_processes.VideoExtensoProcess` creates the
:class:`~crappy.tool.image_processing.video_extenso.VideoExtensoTool` after it
starts. Finally, that tool creates, communicates with, and stops one
:class:`~crappy.tool.image_processing.video_extenso.tracker.Tracker` process
per spot. The public Block consequently does not construct or manage any of
these low-level helpers directly.

FT232H feature
""""""""""""""

While exploring the module, you will notice many occurrences of the term
*FT232H*. It refers to a chip from FTDI, performing USB to I2C, SPI, Serial and
GPIO conversion. It was integrated on one of Adafruit's boards. We considered
at some point the possibility to use it for achieving communication on
low-level buses with Crappy, using only a PC and an FT232H. It turned out that
the :mod:`pyusb` Python module required to talk to the chip is not
process-safe, and a complex architecture had to be implemented to ensure
multiprocess safety. This code can be found in the :mod:`crappy.tool.ft232h`
submodule. For all the InOuts and Actuators communicating over low-level buses,
a second version communicating through an FT232H was written and stored in the
`ft232h` submodules.

After testing quite many options, we could not get the communication over
FT232H to be completely stable. We always ended up with crashes, probably due
to a wrong design of the server architecture used to ensure multiprocessing
safety. In some cases though, the FT232H option worked really great and could
be used on experimental setups without any problem. We thus decided to keep
this feature in the module, but not to advertise it in the documentation and in
the examples.

Detailed runtime sequence of Crappy
-----------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

Crappy's main strength lies in the use of massive parallelization to maximize
the performance of the module. Unfortunately, this means we had to cope with
Python's notoriously complex :mod:`multiprocessing` architecture, and come up
with a number of solutions to ensure a smooth execution and synchronization of
all the processes. This section describes the different phases of Crappy's
execution, indicating for each phase which objects and mechanisms are involved
and what they exactly do.

In the main Process
+++++++++++++++++++

The __init__ phase
""""""""""""""""""

Before calling :ref:`crappy.start()` or :ref:`crappy.prepare()`, only one
Process is running (the ``__main__`` Process). All the instantiated Blocks will
be children Processes of the ``__main__`` Process, as soon as the next phase
starts. The ``__main__`` Process will normally live until the test is over and
*should* not stop before any of its children.

As all Processes are children of ``__main__``, it is a very natural position
for ``__main__`` to be the director managing all the other ones. And since all
the **instances** of :class:`~crappy.blocks.Block` are meant to live in their
own Process at some point, the methods required to drive all the Processes
cannot be regular methods of Block. Instead, we have to rely on the
:obj:`classmethod` of Block, because these methods will always be executed in
``__main__`` if they are called in ``__main__``. Moreover, because they operate
**at the class level**, the :obj:`classmethod` are perfectly suited for
managing the instances of Block. So, when reading the source code of the Block,
remember that every :obj:`classmethod` is meant to be called directly from the
``__main__`` Process and not by an instance of Block.

The first thing that happens in the Block when calling :py:`import crappy` is
that the class attributes of Block are initialized (mostly to :obj:`None`).
These class attributes are :mod:`multiprocessing` synchronization objects used
for managing the execution of all the Processes. They include :

- Two flags (:obj:`bool`) indicating whether all the Blocks have prepared and
  launched.
- A :obj:`~weakref.WeakSet` storing the reference of all the instantiated
  Blocks.
- A :obj:`list` of all the names of the Blocks.
- An :obj:`int` specifying the minimum level for :mod:`logging`.
- A :obj:`multiprocessing.Value` storing the initial timestamp common to all
  the Blocks.
- A :obj:`multiprocessing.Barrier` used for ensuring that all the Blocks wait
  for each other before starting.
- Three :obj:`multiprocessing.Event` objects indicating when the Blocks should
  start, pause, and stop running.
- Two :obj:`multiprocessing.Event` signaling an :exc:`Exception` or a
  :exc:`KeyboardInterrupt` encountered by Crappy.
- An optional ``multiprocessing.Manager`` providing the shared dictionaries
  used by ImageLinks when at least one VisionBlock is present.
- A :obj:`logging.Logger` recording all the log messages from all the Blocks.
- A :obj:`multiprocessing.Queue` used for sending all the log messages to the
  Logger.
- A :obj:`threading.Thread` managing the execution of the Logger.
- A flag (:obj:`bool`) indicating the Logger Thread when to stop running.
- A flag (:obj:`bool`) indicating whether an :exc:`Exception` should be raised
  when Crappy terminates, in case one has been caught during Crappy's
  execution.

Then, when a :class:`~crappy.blocks.Block` is instantiated, its instance
attributes are initialized (mostly to :obj:`None`). Most of these instance
attributes will later be set equal to the synchronization and logging class
attributes. In addition to the synchronization and logging attributes, each
instance of Block also has :

- A few validated public properties managing its execution (target looping
  frequency, niceness, flag for displaying the achieved looping frequency,
  pausability, and whether it supports image Links). Their private backing
  attributes are implementation details and should not be assigned directly by
  children Blocks.
- A few buffers storing values needed for trying to achieve and displaying the
  looping frequency.
- A name, given by a :obj:`classmethod` to ensure it is unique.
- Lists of regular input and output Links. VisionBlocks additionally keep lists
  of their input and output ImageLinks and the associated image state.
- A list of configuration Pipe endpoints to close when the ``fork`` start
  method makes the child inherit endpoints owned by other Blocks.

The constructor also registers the Block's name and type as a node in the
module-level :class:`~crappy.links.LinkGraph`. Creating regular Links and
ImageLinks adds the corresponding edges during this phase, so the complete
connection topology is available before Crappy starts any Process.

Each instance of Block might of course also perform extra tasks, depending how
the ``__init__`` method of the child class is implemented. The ``__init__``
phase ends when either :ref:`crappy.start()` or :ref:`crappy.prepare()` is
called (the first thing *start* does is to call *prepare*).

The prepare phase
"""""""""""""""""

When the :meth:`crappy.blocks.Block.prepare_all` :obj:`classmethod` (aliased to
:ref:`crappy.prepare()` for conciseness) is called, it first sets the
:obj:`logging.Logger` of the ``__main__`` Process. Note that
:meth:`~crappy.blocks.Block.prepare_all` accepts one argument indicating the
minimum level for logging. Then, all the synchronization class attributes
listed above are instantiated to their target type (most of them were
previously initialized to :obj:`None`). At that point, the number of Blocks is
known, so the :obj:`~multiprocessing.Barrier` is set to this number +1 for the
``__main__`` Process. The :obj:`~multiprocessing.Value` storing the initial
timestamp is initialized to a negative value, to make it clear that it is not
set yet.

Then, the :class:`~crappy.tool.ft232h.USBServer` Process tool is started if
needed (see :ref:`FT232H feature`). After that, for each Block, its
synchronization instance attributes are set to the corresponding class
attributes of Block. Basically, the class attributes are shared with all the
instances of Block. This is only possible because at that point the Blocks do
not live in a separate Process yet, they all run in ``__main__``.

When at least one VisionBlock is present, ``prepare_all`` also performs the
graph-level image setup before starting the children. It first verifies that
the Block registry and :class:`~crappy.links.LinkGraph` contain the same names.
For each image source, it walks every downstream ImageLink descendant and asks
that consumer whether it needs source-side configuration. Each returned
:class:`~crappy.blocks.vision.block.ConfigRequest` is checked against the
source and requester names, duplicated, and registered at both ends with a
dedicated one-way :obj:`multiprocessing.Pipe`. Under the ``fork`` start method,
each Block also receives the list of unrelated endpoints that it will inherit
and must close when its child Process begins. This is important because
otherwise an unintended open copy could prevent a requester from detecting a
failed source through end-of-file.

The main Process then creates one ``multiprocessing.Manager`` for the shared
image metadata and format dictionaries. Each image-producing VisionBlock
creates a compact shared-memory name, lock, readiness Event, and image counter,
and publishes this same set of objects to all its outgoing ImageLinks. This is
the framework-level state for one source buffer, the actual shared-memory
segment is created later by the image-producing child, after its configuration
has established the final image format.

Finally, all the Blocks are started in separate Processes. The main Process
closes its copies of all configuration Pipe endpoints after starting them,
whether preparation succeeds or fails. If an exception is caught during the
*prepare* phase, it first breaks the :obj:`~multiprocessing.Barrier` and then
triggers :ref:`The cleanup phase`.

The renice phase
""""""""""""""""

Right after the *prepare* phase should follow the *renice* phase. It
corresponds to the call of the :meth:`crappy.blocks.Block.renice_all`
:obj:`classmethod` of the Block (aliased to :ref:`crappy.renice()` for
conciseness). This method accepts one attribute, indicating whether negative
nicenesses can be accepted (Linux and macOS only). On Windows, it does nothing
as the concept of niceness is not defined. On Linux and macOS, it renices all
the running Blocks to the value specified in their ``niceness`` attribute.
Whether this value differs from default (0) depends on how the Blocks are
written. If an exception is caught during the *renice* phase, it first breaks
the :obj:`~multiprocessing.Barrier` and then triggers :ref:`The cleanup phase`.

The launch phase
""""""""""""""""

The first thing happening after calling :meth:`crappy.blocks.Block.launch_all`
(aliased to :ref:`crappy.launch()` for conciseness) is that the ``__main__``
Process starts waiting at the synchronization :obj:`~multiprocessing.Barrier`.
This Barrier is shared by all the Blocks, and its value is set to the number of
Blocks +1. Therefore, the Barrier only breaks when all the Blocks have reached
it, as well as the ``__main__`` Process. In case one of the Processes doesn't
make it to the Barrier, a :obj:`~threading.BrokenBarrierError` is raised to
indicate all the other Blocks not to wait forever at the Barrier.

Once every Process has reached the Barrier, it breaks and releases them all.
At that moment, the :obj:`~multiprocessing.Value` storing the initial timestamp
is set to the current time (in seconds since epoch). After that, the start
:obj:`~multiprocessing.Event` indicating all the Blocks to start looping is
set, which releases them all. After that, the ``__main__`` Process remains idle
for most of the test, only waiting for one of the Blocks to finish. As soon as
at least one Block is done, :ref:`The cleanup phase` starts. This phase also
starts in case an Exception is caught.

The cleanup phase
"""""""""""""""""

This phase is triggered every time an exception (of any nature) is caught in
the ``__main__`` Process, or if at least one Block has stopped. The
corresponding method is :meth:`crappy.blocks.Block._cleanup`. Its goal is to
make sure that all the Blocks stop as expected, and that the other Processes
and Threads of Crappy terminate as well. It first sets the stop
:obj:`~multiprocessing.Event`, indicating all the Blocks to stop looping and to
finish as soon as possible. During a normal VisionBlock finish, incoming
shared-memory handles are closed without unlinking them, while an image source
closes and unlinks the output segment that it owns.

The main Process gives all Blocks 3 seconds to finish. If any Block is still
alive past this delay, it is terminated. Then, the
:obj:`~multiprocessing.Process` in charge of the
:class:`~crappy.tool.ft232h.USBServer` is stopped, if applicable. The shared
Manager that provided the ImageLink dictionaries is shut down only after the
Block Processes have finished, and the :obj:`~threading.Thread` collecting all
log messages is also stopped. Shortly before returning, Crappy is reset by
:meth:`~crappy.blocks.Block.reset`. This clears the Block registry and
:class:`~crappy.links.LinkGraph`, drops the shared Manager reference, and
re-initializes the synchronization state because it is no longer needed.
Finally, an exception might be raised in three cases :

- If all the Blocks are not done running at the end of this phase.
- If an :exc:`Exception` was caught during Crappy's execution.
- If Crappy was stopped using :kbd:`Control-c`, resulting in a
  :exc:`KeyboardInterrupt`.

The goal of this exception is to stop the execution of the ``__main__``
Process, to avoid any more code to be executed in case something went wrong in
Crappy. Note that this behavior can be disabled using the *no_raise* argument.
In normal operating mode, if this phase ends without raising an exception, it
indicates that Crappy executed and terminated gracefully.

In the children Processes
+++++++++++++++++++++++++

As soon as the start method of a :class:`~crappy.blocks.Block` is called, it
starts running in a new :obj:`~multiprocessing.Process` separate from the
``__main__`` one. It therefore lives its own independent life, and is only
linked to the ``__main__`` Process by the :mod:`multiprocessing`
synchronization objects. The ``__main__`` Process still has the option to kill
the Blocks, if at the end of Crappy they do not stop by themselves.

When a Block is started, it first sets its :obj:`~logging.Logger`. Under the
``fork`` start method, it immediately closes the configuration Pipe endpoints
that belong to other Blocks. It then runs
:meth:`~crappy.blocks.Block.prepare` to perform any preliminary task.

For a VisionBlock, its specialized ``prepare`` method first performs the work
needed to determine its final image format and processing state. An image
source handles its incoming configuration requests and sends the results.
Requesting consumers wait for these responses and create their processing
helpers. Their eventual call to
:meth:`crappy.blocks.vision.VisionBlock.prepare` obtains the synchronization
objects stored on all incoming ImageLinks. A source with image outputs creates
its single :class:`multiprocessing.shared_memory.SharedMemory` segment,
publishes its shape and dtype, and signals that the buffer is ready. Each
consumer waits for that signal, attaches to the segment, and allocates a local
array into which coherent frames will be copied. These waits periodically
check the preparation Barrier and stop Event, so another Block's failure does
not leave them waiting indefinitely. Rejecting image-only cycles when the
graph is constructed also ensures that buffer dependencies can be resolved.

After preparation, the Block reaches the :obj:`~multiprocessing.Barrier`,
where it waits for all the other Blocks and the ``__main__`` Process to be
ready. If anything goes wrong before that, the Block breaks the Barrier, thus
signaling its failure to the other ones through a
:obj:`~threading.BrokenBarrierError`.

As soon as all the other Processes are ready, the Barrier breaks and releases
the Block. This one then waits a second time for the ``__main__`` Process to
set the common start timestamp, after what all the Blocks are released. The
:meth:`~crappy.blocks.Block.begin` method is then called to perform any action
specific to the first loop, and then the Block starts looping forever by
calling it :meth:`~crappy.blocks.Block.main` method. Under the hood, this
method calls the :meth:`~crappy.blocks.Block.loop` method, performing the main
task for which the Block was written. It also handles the regulation and the
display of the looping frequency, if requested by the user. If a
:class:`~crappy.blocks.Pause` Block is used, all the Blocks having their
``pausable`` attribute set to :obj:`True` might be paused (most Blocks by
default). When paused, the :meth:`~crappy.blocks.Block.main` method keeps
looping at its target frequency, but the :meth:`~crappy.blocks.Block.loop`
method is never called. As soon as the pause ends, the normal behavior is
restored.

There are several ways the Block can stop. First, the stop
:obj:`~multiprocessing.Event` might be set in another Process, which conducts
each Block to stop running. Second, an :exc:`Exception` can be caught in the
Block. And third, the Block might be killed by the ``__main__`` Process if it
becomes unresponsive. In the first two cases, the
:meth:`~crappy.blocks.Block.finish` method is called for performing the cleanup
actions. The Block then stops running, and the associated Process finishes.
