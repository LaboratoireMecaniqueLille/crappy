======================
Developers information
======================

.. role:: py(code)
  :language: python
  :class: highlight

Contributing to Crappy
----------------------

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
instantiated by users to achieve a given overall behavior in their scripts. The
Blocks can exchange information together through Links, with no restriction on
the number of Links and the type of Blocks they connect.

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

Camera-related Blocks
"""""""""""""""""""""

One major downside of the Pipes is that they can overflow, in which case data
from the sender Block is simply discarded when trying to send it. This behavior
is especially inconvenient for sending images (because they're so large), so an
alternative solution was chosen for image-processing Blocks. Instead of sharing
the acquired images via Pipes, they are instead written by the sender Block to
a shared :obj:`~multiprocessing.Array` where they can then be read by receiver
processes. The receiver processes can display the images, record them, or
process them.

Note that these receiver processes are not Blocks. Instead, they are children
of the base :class:`~crappy.blocks.camera_processes.CameraProcess`, and are
managed by a :class:`~crappy.blocks.Camera` Block in a quite similar way as the
base :class:`~crappy.blocks.Block` manages all the other ones. Using this
solution, we were able to parallelize the image acquisition, display,
recording, and processing, which greatly improved the performance compared to
previous versions of Crappy.

The use of shared Arrays to exchange data between Blocks was not chosen in the
general case for several reasons. First, it adds an extra complexity that is
not needed when sending numerical data. And second, it requires to know in
advance the size of the data to share, which is easy to determine for images
but not for numerical data.

Actuators, Cameras, InOuts
++++++++++++++++++++++++++

Some of the Blocks rely on specific types of helper object, that they can
drive. It is the case for :

- The :class:`~crappy.blocks.Camera` Block that drives one
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

During the :meth:`~crappy.blocks.Camera.prepare` method of the Camera Block (or
one of its children), the user can choose to enable the display of a
:class:`~crappy.tool.camera_config.CameraConfig` window. This interactive
`tkinter` based GUI allows to visualize the images acquired by the
:class:`~crappy.camera.Camera` object, and to interactively adjust the settings
available for the Camera. All the code managing the configuration GUI is stored
in the :mod:`crappy.tool.camera_config` submodule of Crappy. There, children
of CameraConfig are defined to handle the specific needs of each child of the
Camera Block. Also, helper classes are stored in separate files. The base
CameraConfig is quite complex on its own, with a number of variables, bindings
and traces that generate a feature-rich GUI. It even manages a parallel process
(:class:`~crappy.tool.camera_config.config_tools.HistogramProcess`) in which a
histogram of the acquired images is calculated in real-time.

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

The children of the :class:`~crappy.blocks.Camera` Block manage the execution
of the various :class:`~crappy.blocks.camera_processes.CameraProcess` that
might be requested by the user, including the one performing the image
processing. The code performing the processing is however not included in the
children of the Camera Block or the CameraProcess class. It is instead stored
in a separate submodule, :mod:`crappy.tool.image_processing`. The rationale
behind is to separate the code dealing with multiprocessing and the one
performing image processing.

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
- Two :obj:`multiprocessing.Event` indicating the Blocks when to start and when
  to stop running.
- Two :obj:`multiprocessing.Event` signaling an :exc:`Exception` or a
  :exc:`KeyboardInterrupt` encountered by Crappy.
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

- A few attributes managing its execution (target looping frequency, niceness,
  flag for displaying the achieved looping frequency).
- A few buffers storing values needed for trying to achieve and displaying the
  looping frequency.
- A name, given by a :obj:`classmethod` to ensure it is unique.

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
not live in a separate Process yet, they all run in ``__main__``. Short after,
all the Blocks are started, meaning that they all run in a separate Process. If
an exception is caught during the *prepare* phase, it first breaks the
:obj:`~multiprocessing.Barrier` and then triggers :ref:`The cleanup phase`.

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
indicate all the other Blocks not to wait forever on the Barrier.

Once every Process has reached the Barrier, it breaks and releases them all.
At that moment, the :obj:`~multiprocessing.Value` storing the initial timestamp
is set to the current time (in seconds since epoch). After that, the start
:obj:`~multiprocessing.Event` indicating all the Blocks to start looping is
set, which releases them all. After that, the ``__main__`` Process remains idle
for most of the test, only waiting for one of the Blocks to finish. As soon as
at least one Block is done, :ref:`The cleanup phase` starts. This phase also
starts in case an Exception is caught.

The cleanup phase
"""""""""""""""""""

This phase is triggered every time an exception (of any nature) is caught in
the ``__main__`` Process, or if at least one Block has stopped. The
corresponding method is :meth:`crappy.blocks.Block._exception`. Its goal is to
make sure that all the Blocks stop as expected, and that the other Processes
and Threads of Crappy terminate as well. It first sets the stop
:obj:`~multiprocessing.Event`, indicating all the Blocks to stop looping and to
finish as soon as possible. It then lets 3 seconds for all the Blocks to
finish. If any Block is still alive passed this delay, it is mercilessly
terminated. Then, the :obj:`~multiprocessing.Process` in charge of the
:class:`~crappy.tool.ft232h.USBServer` is stopped, if applicable. Same goes for
the :obj:`~threading.Thread` collecting all the log messages. Finally, an
exception might be raised in three cases :

- If all the Blocks are not done running at the end of this phase.
- If an :exc:`Exception` was caught during Crappy's execution.
- If Crappy was stopped using CTRL+C, resulting in a :exc:`KeyboardInterrupt`.

The goal of this exception is to stop the execution of the ``__main__``
Process, to avoid any more code to be executed in case something went wrong in
Crappy. Note that this behavior can be disabled using the *no_raise* argument.
In normal operating mode, if this phase ends without raising an exception, it
indicates that Crappy executed and terminated gracefully.

In the children Processes
+++++++++++++++++++++++++

As soon as the start method of a :class:`~crappy.blocks.Block` is called, it
starts running in a new :obj:`~multiprocessing.Process` separate from the
``__main__`` one. It therefore lives it own independent life, and is only
linked to the ``__main__`` Process by the :mod:`multiprocessing`
synchronization objects. The ``__main__`` Process still has the option to kill
the Blocks, if at the end of Crappy they do not stop by themselves.

When a Block is started, it firsts sets its :obj:`~logging.Logger` and runs its
:meth:`~crappy.blocks.Block.prepare` method to perform any preliminary task.
Then, it reaches the :obj:`~multiprocessing.Barrier`, where it waits for all
the other Blocks and the ``__main__`` Process to be ready. If anything wrong
happens before that, the Block breaks the Barrier, thus signaling its failure
to the other ones through a :obj:`~threading.BrokenBarrierError`.

As soon as all the other Processes are ready, the Barrier breaks and releases
the Block. This one then waits a second time for the ``__main__`` Process to
set the common start timestamp, after what all the Blocks are released. The
:meth:`~crappy.blocks.Block.begin` method is then called to perform any action
specific to the first loop, and then the Block starts looping forever by
calling it :meth:`~crappy.blocks.Block.main` method. Under the hood, this
method calls the :meth:`~crappy.blocks.Block.loop` method, performing the main
task for which the Block was written. It also handles the regulation and the
display of the looping frequency, if requested by the user.

There are several ways the Block can stop. First, the stop
:obj:`~multiprocessing.Event` might be set in another Process, which conducts
each Block to stop running. Second, an :exc:`Exception` can be caught in the
Block. And third, the Block might be killed by the ``__main__`` Process if it
becomes unresponsive. In the first two cases, the
:meth:`~crappy.blocks.Block.finish` method is called for performing the cleanup
actions. The Block then stops running, and the associated Process finishes.
