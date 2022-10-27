======================
Developers information
======================

Contributing to Crappy
----------------------

If you want to help developing Crappy with us, we'll be more than happy to
welcome you in our community ! Here you'll find some practical information on
how Crappy works under the hood, and a few specific rules to follow when coding
in Crappy.

If you notice bugs or points needing improvement, you can notify us by creating
a `new issue <https://github.com/LaboratoireMecaniqueLille/crappy/issues>`_ on
GitHub. Alternatively if you came up with a fix, added hardware or blocks that
you think may be useful to others, or simply improved Crappy in any way, please
add a `new pull request
<https://github.com/LaboratoireMecaniqueLille/crappy/pulls>`_ on GitHub to share
it with all Crappy users.

Crappy's specific rules
-----------------------

There's only a limited number of specific rules applying to Crappy, so you
shouldn't have much trouble following them.

**1.** PEP8 consideration
  We try to respect, as much as possible, the `PEP8
  <https://peps.python.org/pep-0008/>`_ convention, except on one
  specific point. Indents in Crappy are 2 spaces wide, while they should be 4
  spaces wide following the PEP8. This choice was made to improve readability.

**2.** Docstrings
  The `Google style <https://google.github.io/styleguide/pyguide.html>`_ of
  docstrings was chosen for Crappy. Other docstring styles may not be recognized
  when building documentation, so Google style should always be used.

**3.** Naming
  The chosen naming convention for classes is lowercase separated with
  underscores with only the leading letter capital. For instance ``Block``, or
  ``Other_block`` are valid class names. There are however a few exceptions
  (e.g. ``IOBlock``). The ``.py`` file names should be in lowercase separated
  with underscores. This way no confusion in possible between file and class
  imports.

**4.** Versioning
  The chosen convention for versioning is the following : the first figure is
  increased when a non backward-compatible change is added, the second figure
  is increased when a new (and backward-compatible) functionality is added, the
  third figure is increased at each new release not matching one of the previous
  cases.

How Crappy works
----------------

.. note::
  This is a very simplified overview of how the code actually works. Only the
  main ideas are presented, and many technical aspects are omitted. Reading the
  code remains the only way to truly understand it !

The different objects
+++++++++++++++++++++

The Block and the blocks
""""""""""""""""""""""""

To the regular user, Crappy is organized in blocks of different types having
each a specific function and linked together with "links". Under the hood each
block, no matter his type, is an instance of the `Block
<https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/crappy/blocks/
block.py>`_ class. This class is itself a subclass of `multiprocessing.Process
<https://docs.python.org/3/library/multiprocessing.html#multiprocessing.
Process>`_, so all the instantiated blocks actually start their own process at
some point.

The different block types (e.g. IOBlock, Camera, etc. ) are subclasses of
`Block <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/crappy/
blocks/block.py>`_ implementing specific methods for serving various purposes.
Except for class inheritance, the only other constraint on blocks is to
overwrite the ``loop`` method of Block. Everything else is left free to the
user. Most of the blocks are self-sufficient for fulfilling their goal (e.g.
`Displayer <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/
crappy/blocks/displayer.py>`_, `PID <https://github.com/LaboratoireMecanique
Lille/crappy/blob/master/crappy/blocks/pid.py>`_, etc.) while others rely on
specific classes that they instantiate just like "tools". These blocks are :

- The `Camera block <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
  master/crappy/blocks/camera.py>`_ that instantiates the `camera objects
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/crappy/
  camera>`_.

- The `IOBlock <https://github.com/Laboratoire MecaniqueLille/crappy/blob/
  master/crappy/blocks/ioblock.py>`_ block that instantiates the `inout objects
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/crappy/
  inout>`_.

- The `Machine <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
  master/crappy/blocks/machine.py>`_ block that instantiates the `actuator
  objects <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
  crappy/actuator>`_.

- The `Link <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/
  crappy/links/link.py>`_ class that may, but doesn't always, instantiate the
  `modifier objects <https://github.com/LaboratoireMecaniqueLille/crappy/tree/
  master/crappy/modifier>`_ objects. It is a particular case as we'll see below,
  since links are not blocks.

The cameras, inouts and actuators
"""""""""""""""""""""""""""""""""

The camera, inout and actuator objects are subclasses of `Camera <https://
github.com/LaboratoireMecaniqueLille/crappy/blob/master/crappy/camera/
camera.py>`_, `InOut <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
master/crappy/inout/inout.py>`_ and `Actuator blocks <https://github.com/
LaboratoireMecaniqueLille/crappy/blob/master/crappy/actuator/actuator.py>`_
respectively. Each type has its own `metaclass <https://docs.python.org/3/
reference/datamodel.html#metaclasses>`_, specifying specific mandatory methods
to be implemented. Except for these two constraints, everything else il left
free to the user.

The links
"""""""""

Whenever a link is added between two blocks, this basically creates a
`multiprocessing.Pipe <https://docs.python.org/3/library/multiprocessing.
html#multiprocessing.Pipe>`_ linking the two processes of these blocks. The
links are then responsible for sending and receiving data exchanged between the
blocks. The data is always exchanged in the form of dictionaries, the keys being
the so-called 'labels' that many blocks take as arguments. If specified in their
arguments, the links can also instantiate one or several modifier objects that
will perform operations on the dict going through the link.

The tools
"""""""""

Additionally, `tool <https://github.com/LaboratoireMecaniqueLille/crappy/tree/
master/crappy/tool>`_ files can be used by any block and any camera, inout and
actuator object. Unlike the objects previously described, tools do not follow
any specific template and do not even need to be ``.py`` files.

The program execution
+++++++++++++++++++++

Initialization
""""""""""""""

When the user starts a script for running a test with Crappy, the entire module
is first loaded at ``import crappy``. The different Block objects are then
instantiated, at each line ``<name> = crappy.blocks.<Block_name>(*args,
**kwargs)``. At this very moment the ``__init__`` method of the Block is called,
and the code was written so that any "tool" object the Block would need (camera,
actuator, etc.) is also instantiated at the same time. When
``crappy.link(<name1>, <name2>)`` is called, a Pipe is instantiated between the
blocks (that are themselves processes, remember). Before ``crappy.start()`` is
called, only the main process is running and only ``__init__`` methods have been
called.

Beginning
"""""""""

When the ``crappy.start()`` method is called, all blocks execute the
``self.start()`` method, which starts as many processes. Then each Block calls
its ``prepare`` method, which is meant to perform any preliminary task (opening
a port, initializing a device, etc.).

Main
""""

Once all blocks are done preparing, they all start the test phase in a
synchronized way. The ``begin`` method is called for performing a specific task
at the very beginning of the assay (e.g. sending a trigger signal to a device),
and then the ``main`` method is called. It is actually an infinite loop calling
the ``loop`` method, thus repeatedly performing the main task the Block has to
do during the test. Only during the ``begin`` and ``main`` methods execution can
the user receive or send data through the links.

Exiting
"""""""

When either an exception is raised, or a `Generator Block <https://github.com/
LaboratoireMecaniqueLille/crappy/blob/master/crappy/blocks/generator.py>`_
reaches the end of its path, or the user issues a CTRL+C key, the program will
try to end. This is achieved by raising a ``CrappyStop`` exception that is
caught by all the blocks, and also by sending specific signals in the links. The
blocks will then let the current ``loop`` call return in a normal way, then call
their ``finish`` method. It allows to perform any action that should occur after
the assay ends (e.g. closing a port, switching off a device). The ``finish``
method only has a limited time to return, otherwise the Process will be abruptly
terminated.

The case of non-Block objects
"""""""""""""""""""""""""""""

The code is written so that the cameras, inouts and actuator objects follow a
similar scheme. Their ``__init__`` method is called during their "parent" Block
``__init__``, their ``open`` method is called during ``begin``, their ``close``
method is called during ``finish``. The method(s) called during ``loop`` differ
according to the class type. On cameras only ``get_image`` is called, on inouts
``get_data`` and ``set_cmd`` may be called, and on actuators either
``get_position`` and ``set_position`` or ``get_speed`` and ``set_speed`` may be
called (depending on the driving mode).

All Crappy features
-------------------

Additionally to the Python module ``crappy``, Crappy also comes with extra files
and functionalities (including the documentation you're currently reading !).
Here's a list of all these features, that can all be found on the `GitHub repo
<https://github.com/LaboratoireMecaniqueLille/crappy>`_.

- **Examples** :
  A very complete set of ready-to-run examples illustrating how to use Crappy's
  main functionalities. Most of them can be run without any specific hardware,
  and some of them require a camera (a webcam is fine) and/or a cuda-compatible
  GPU.
- **crappy** :
  The Python module itself, described in details on this very page.
- **Documentation**:
  You are currently reading it ! The Makefile and all the ``.rst`` files
  necessary for building the doc are in the ``docs`` folder on GitHub. See the
  :ref:`Documentation` section for more information.
- **Templates** :
  A set of templates, mainly intended for internal use in our lab. They can
  still prove interesting to a larger public as they show how an actual complex
  test is written in Crappy. They're located in the ``impact`` folder on Github.
- **C/C++ codes** :
  Codes for working with some National Instruments boards and with CameraLink
  modules. They're not included in the ``pip`` install and can thus be used only
  with a ``setup`` install. See :ref:`Installation` for details. They're located
  in the ``sources`` folder on GitHub.
- **Utilities** :
  Various helper programs, e.g. for writing udev rules or performing an
  auto-tune of a PID. They can be found in the ``util`` folder on GitHub.
