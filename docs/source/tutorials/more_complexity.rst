=======================
Towards more complexity
=======================

.. role:: py(code)
  :language: python
  :class: highlight

This page introduces feedback loops, Modifiers, streaming acquisition, Python
techniques for reducing repetition, direct use of hardware objects, and
advanced runtime controls.

1. Using feedback loops
-----------------------

The previous tutorial used linear data-flow patterns. This section introduces
feedback loops. Although :ref:`Links <crappy_docs/links:links>` are unidirectional, they can
possible to have them form a loop to send back information to a Block. This is
especially useful for driving :ref:`Generator <crappy_docs/blocks:generator>` Blocks, as detailed in
:ref:`a next section <tutorials/more_complexity:3. advanced generator condition>`. For now, let's look at
the example script given in the :ref:`actuator-control tutorial
<tutorial-actuator-control>`. The :ref:`Fake Machine <crappy_docs/blocks:fake machine>` Actuator that is used
takes its commands as a voltage, which is quite unsatisfying since the achieved
speed will vary depending on the characteristics of the motor. Instead, it
would be preferable to send speed commands, and to somehow have the motor adapt
and reach this speed.

To achieve this behavior, a possibility is to use the :ref:`PID <crappy_docs/blocks:pid>` Block. It will
receive on the one hand the target speed, and on the other hand the current
speed of the motor. Based on these inputs, it will generate a voltage command
to send to the Machine Block driving the Fake Motor. If the PID is well set,
the measured speed converges toward the target value. The example uses these
Blocks:

.. literalinclude:: /downloads/more_complexity/tuto_loops.py
   :language: python
   :emphasize-lines: 7-15, 17-24, 26-27, 29-37
   :lines: 1-38

The Generator sends the target speed under the label
:py:`'target_speed'`, the Machine Block takes the :py:`'voltage'` label as a
command and returns the :py:`'actual_speed'`, and the PID Block takes both the
:py:`'target_speed'` and :py:`'actual_speed'` labels as inputs and returns the
:py:`'voltage'` label. There is also a Grapher Block plotting both the
:py:`'target_speed'` and :py:`'actual_speed'` labels. Also, notice the
:py:`'spam'` argument of the Generator Block, that ensures that the Block sends
the command at each loop so the graph receives repeated values. Next, link the
Block together consistently:

.. literalinclude:: /downloads/more_complexity/tuto_loops.py
   :language: python
   :emphasize-lines: 39-40, 42, 44-45

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

The calls :py:`crappy.link(mot, pid)` and :py:`crappy.link(pid, mot)` form a
feedback loop. A feedback loop is necessary when a Block must modify its output based on
the effect it has on another target Block. You can :download:`download this
feedback loop example </downloads/more_complexity/tuto_loops.py>` to run it
locally on your machine. You can then tune the settings of the motor and see
how the PID will react.

2. Using Modifiers
------------------

Use :ref:`Modifiers <crappy_docs/modifiers:modifiers>` to alter data flowing
through :ref:`Links <crappy_docs/links:links>`. A Block's output might not always be
exactly what you need. For example, data from a sensor might be too noisy and
require some filtering. Or a command might have to be sent to two different
motors, but with an offset on one of them. Such small alterations of the data
should not require a new :ref:`Block <crappy_docs/blocks:block>` or a change to an existing
one. Use a
:class:`~crappy.modifier.Modifier` objects.

The principle of Modifiers is that each Modifier is attached to a given Link.
Every time a Block sends data through the Link, the Modifier alters it first.
The same Link can have several Modifiers attached, in
which case they are called in the same order as they are given. Unlike the
operations performed by Blocks, Modifiers run while the source sends its data.
Use them for short operations that should not delay that source. The
:doc:`../concepts/choosing_custom_object_type` guide explains when a separate
Block is more appropriate. The following example adds a Modifier to a Link.

Starting from the example of the previous section, we now want to know the
current position of the motor. To calculate this value, we just have to
integrate the measured speed over time. Crappy provides the
:ref:`Integrate <crappy_docs/modifiers:integrate>` Modifier for
integrating a signal over time. Let's add it on a Link starting from the
Machine Block and pointing towards a new :ref:`Grapher <crappy_docs/blocks:grapher>` for the position:

.. literalinclude:: /downloads/more_complexity/tuto_modifiers.py
   :language: python
   :emphasize-lines: 39, 49-51

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

Pass Modifiers to the :py:`'modifier'`
argument of the :func:`crappy.link()` function. Each Modifier has to be
instantiated, and might require arguments. To know what the effect of a
Modifier is, and which argument it takes, refer to the :ref:`Modifiers <crappy_docs/modifiers:modifiers>` section
of the API. Here, the chosen Modifier is :class:`~crappy.modifier.Integrate`.
It must be given the name of the label to integrate, and here the name of the
label carrying the integral value is also specified. This new label is added to
the data flowing through the Link, and can then be used by the downstream
Block. In the case of the Integrate Modifier, all the other labels are
preserved.

Modifiers provide focused transformations for data flowing through Links. If
the built-in Modifiers do not cover a use case, see :ref:`how to code your own
Modifiers <tutorials/custom_objects:1. custom modifiers>`. You can :download:`download this Modifier
example </downloads/more_complexity/tuto_modifiers.py>` to run it locally on
your machine. The Modifiers distributed with Crappy are also showcased in the
`examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/tree/master/examples/modifiers>`_.

3. Advanced Generator condition
-------------------------------

In the :ref:`command-generation tutorial <tutorial-command-generation>`, the
:ref:`Generator <crappy_docs/blocks:generator>` Block and its :ref:`Generator Paths <crappy_docs/blocks:generator paths>` were introduced. In that
section, two possible syntaxes were given for the :py:`'condition'` key of a
Path :obj:`dict`. The value :obj:`None` can be given, in which case the Path
never ends. Alternatively, a :obj:`str` in the format :py:`'delay=xx'` can be
given, in which case the Path ends after the specified delay. There is actually
another way to specify the stop condition, detailed in this
section.

Getting right to the point, the third way to specify a stop condition is to
give a :obj:`str` in the format :py:`'label>value'` or :py:`'label<value'`.
Replace :py:`'label'` with the name of the label to monitor, and :py:`'value'`
with a numerical value to compare the label with. The principle of this type
of condition is that the Generator should be sent the label to monitor. At
each loop, it checks if any point of the received label is below or above the
given threshold. If that is the case, the stop condition is met and the
Path ends. The reason why this type of stop condition was not introduced in the
section dedicated to Generators is that it requires the concept of feedback
loop, introduced :ref:`earlier on this page
<tutorials/more_complexity:1. using feedback loops>`.

.. Note::
   In the stop conditions given as :obj:`str`, you can freely add spaces around
   the *=*, *<* and *>* characters. The condition will still be recognized in
   the same way.

.. Note::
   Generator conditions do not provide an :py:`'=='` comparison. Exact equality
   is unreliable for floating-point values. To implement a tolerance, combine
   a *<* condition with a :class:`~crappy.modifier.Modifier` and
   :obj:`abs`, or use :obj:`math.isclose` in a custom condition.

Let's now use such a stop condition in an example. In the very first example of
the tutorials, a *delay* condition was used for stopping the script. It was
chosen so that the stop condition is met shortly after the sample breaks. If
the elongation rate changes, the delay no longer matches the sample failure.
Instead, the new condition can stop the test shortly before the sample
breaks, no matter the elongation speed. The code is as follows:

.. literalinclude:: /downloads/more_complexity/advanced_generator.py
   :language: python
   :emphasize-lines: 9, 26

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

You can :download:`download this advanced Generator example
</downloads/more_complexity/advanced_generator.py>` to run it locally on your
machine. Try to modify the value of the speed command, and see how the script
always stops at the given condition.

.. Note::
   There is actually one more possibility to define custom stop conditions,
   that is much more advanced and is described in :ref:`a later tutorial
   section <tutorials/complex_custom_objects:1. custom generator paths>`.

4. Dealing with streams
-----------------------

In the :ref:`data-acquisition tutorial <tutorial-data-acquisition>`,
only the regular usage mode of the :ref:`IOBlock <crappy_docs/blocks:ioblock>` was presented. In this mode,
the data points are acquired from the :ref:`In / Out <crappy_docs/inouts:in / out>` object one by one, which
acquires one sample per IOBlock loop. InOuts that support the *streamer* mode
can instead return chunks of samples. The achievable data rate depends on the
device, driver, sample shape, and host system. Stream data is not directly
compatible with most Blocks, as detailed below.

As you may have guessed, in *streamer* mode the data points are acquired and
returned as chunks rather than individually. This means that the IOBlock sends
multiple points at once to the downstream Blocks, which is totally unexpected
for most Blocks. Two Crappy objects directly support *streamer* mode: the
:ref:`HDF Recorder <crappy_docs/blocks:hdf recorder>` Block and the
:ref:`Demux <crappy_docs/modifiers:demux>` Modifier. The following example
uses them together.

The first requirement when using the *streamer* mode is to use an InOut
supporting this mode. Consult the documentation for that InOut to verify its
support. The :class:`~crappy.inout.FakeInOut` InOut supports streaming and does
not require hardware. Second, the
*streamer* mode needs to be enabled on the IOBlock, via the :py:`'streamer'`
argument. If these two conditions are met, *streamer* mode is enabled. The
beginning of the example script looks as follows:

.. literalinclude:: /downloads/more_complexity/tuto_streamer.py
   :language: python
   :emphasize-lines: 7-10, 12-14
   :lines: 1-19

Notice how the :py:`'streamer'` is indeed set on the IOBlock. Except for that,
the syntax for the IOBlock is the same as usual, and the HDFRecorder Block is
also very close to the regular :ref:`Recorder <crappy_docs/blocks:recorder>` one. The differences are that
instead of multiple labels to record, it only expects one stream label
containing all the data at once. It also requires the expected data format to
be specified. Now that the involved Blocks are instantiated, it is time to link
them together:

.. literalinclude:: /downloads/more_complexity/tuto_streamer.py
   :language: python
   :emphasize-lines: 20-24

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` and
   :mod:`psutil` Python modules installed.

The IOBlock and HDFRecorder both handle stream
data, they can be linked together in a normal way. However, the :ref:`Grapher <crappy_docs/blocks:grapher>`
Block cannot accept stream data, so the Demux Modifier must be added to their
Link. This :class:`~crappy.modifier.Modifier` accepts stream data
as an input and outputs regular data usable by most other Blocks. Since streams
can produce data faster than a regular Link consumes it, the Demux discards
intermediate values to avoid overflowing the Link. It still outputs values that can be
used for plotting or any other application. Here, the data from the IOBlock
should be successfully displayed on the graph even though it originates from a
stream.

You can :download:`download this streamer example
</downloads/more_complexity/tuto_streamer.py>` to run it locally on your
machine. If the InOut documentation does not explain its stream format, request
help through the project's GitHub Discussions page.

5. Writing scripts efficiently
------------------------------

Crappy scripts can use ordinary Python tools and third-party packages. The
following techniques reduce repetition and make configuration values easier to
maintain.

5.a. Use variables
++++++++++++++++++

When providing arguments to a Block or any other object, remember that you can
use variables instead of plain text or numbers. It will make your scripts
easier for yourself and others to read and to modify.

Do not write:

.. code-block:: python

   record_pos = crappy.blocks.Recorder('tests/example/data/pos.csv')

   record_force = crappy.blocks.Recorder('tests/example/data/force.csv')

   record_extenso = crappy.blocks.Recorder('tests/example/data/ext.csv')

But write instead:

.. code-block:: python

   base = 'tests/example/data/'

   record_pos = crappy.blocks.Recorder(base + 'pos.csv')

   record_force = crappy.blocks.Recorder(base + 'force.csv')

   record_extenso = crappy.blocks.Recorder(base + 'ext.csv')

5.b. Use loops
++++++++++++++

In a similar way as plain text or numbers can be replaced with variables, you
can also replace :obj:`list`, :obj:`dict`, :obj:`tuple` and other collections
with variables defined elsewhere. This is particularly interesting if you have
large objects that can be generated following a known pattern. In that case,
using loops will save many lines and avoid typos. If you're familiar with the
concept of comprehensions, they can make this code more compact than a loop.

Do not write:

.. code-block:: python

   gen = crappy.blocks.Generator([
       {'type': 'Constant', 'value': 0, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 1, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 2, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 3, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 4, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 5, 'condition': 'delay=5'}])

But write instead:

.. code-block:: python

   path = list()
   for i in range(6):
       path.append({'type': 'Constant', 'value': i, 'condition': 'delay=5'})

   gen = crappy.blocks.Generator(path)

Or even more concise:

.. code-block:: python

   gen = crappy.blocks.Generator([
       {'type': 'Constant', 'value': i, 'condition': 'delay=5'}
       for i in range(6)])

5.c. Use other packages
+++++++++++++++++++++++

Although many examples import only Crappy, a script can use other packages.
This can be convenient for performing operations before Crappy starts, or after
it ends. The :mod:`pathlib` module, for example, handles file paths in a
cross-platform way, so
that you don't have to care about */* and *\\* if you want to make your code
runnable on both Linux and Windows. It is also part of the standard library of
Python, so it doesn't need to be installed.

Do not write:

.. code-block:: python

   record_pos = crappy.blocks.Recorder('tests/example/data/pos.csv')

   record_force = crappy.blocks.Recorder('tests/example/data/force.csv')

   record_extenso = crappy.blocks.Recorder('tests/example/data/ext.csv')

Using :mod:`pathlib`, write instead:

.. code-block:: python

   from pathlib import Path

   base = Path('tests/example/data')

   record_pos = crappy.blocks.Recorder(base / 'pos.csv')

   record_force = crappy.blocks.Recorder(base / 'force.csv')

   record_extenso = crappy.blocks.Recorder(base / 'ext.csv')

6. Using Crappy objects outside of a Crappy test
------------------------------------------------

Hardware integration objects can be used outside a Crappy test without calling
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` or an equivalent
method. Direct use is suitable for a focused operation such as acquiring one
image or one sensor reading. In this case, your code is responsible for opening
and closing the hardware object.

In Crappy, the :ref:`In / Out <crappy_docs/inouts:in / out>`, :ref:`Actuators <crappy_docs/actuators:actuators>` and :ref:`Cameras <crappy_docs/cameras:cameras>` objects
each implement the code needed to interact with a specific equipment. They make
sure that this code is organized and can be called in a standard way, so that
it can be used by the :ref:`IOBlock <crappy_docs/blocks:ioblock>`, :ref:`Machine <crappy_docs/blocks:machine>` and :ref:`Camera <crappy_docs/cameras:camera>` Blocks
respectively. Their public lifecycle methods allow direct hardware access
outside a Crappy test.

To learn more about the mandatory and optional methods that each class can
implement, you should refer to the :ref:`Creating and using custom objects in
Crappy <tutorials/custom_objects:creating and using custom objects in crappy>` page of the tutorials. Here, a very basic example will be used to
demonstrate how a :class:`~crappy.camera.Camera` object can be used for
acquiring and visualizing images. The :class:`~crappy.camera.FakeCamera` will
be used, so that no hardware is required to run the script. The trick to use
this class directly is to instantiate it, without using a :ref:`Camera <crappy_docs/cameras:camera>` Block.
Let's write the first part of the script:

.. literalinclude:: /downloads/more_complexity/outside_test.py
   :language: python
   :emphasize-lines: 7-9
   :lines: 1-3, 5-10

The FakeCamera is directly instantiated, whereas its name is normally given as
an argument to an all-in-one Camera Block. Then, the
:meth:`~crappy.camera.FakeCamera.open` and
:meth:`~crappy.camera.FakeCamera.get_image` methods are called for respectively
initializing the Camera and acquiring an image. The detail of the methods
exposed by the FakeCamera and their exact syntax have to be looked up in the
API. Notice how the arguments to provide to the FakeCamera are passed to the
*open* method instead of being given to the Camera Block. The script above
initializes a FakeCamera and acquires one image from it. The next step displays
that image:

.. literalinclude:: /downloads/more_complexity/outside_test.py
   :language: python
   :emphasize-lines: 4, 12-13

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` and
   *opencv-python* Python modules installed.

You can :download:`download this FakeCamera example
</downloads/more_complexity/outside_test.py>` to run it locally on your
machine. With the visualization added, it should now acquire a picture from the
FakeCamera, display it for 3 seconds and return. With this example, we managed
to use a Camera object without ever calling :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`. Note that
the same principle applies to InOut and Actuator objects, the Camera was only
used here because it is more visual.

7. Advanced control over the runtime
------------------------------------

Crappy provides two forms of advanced runtime control: arguments to
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` and separate startup
methods.

7.a. Alternative startup methods
++++++++++++++++++++++++++++++++

So far, the only option that was presented for starting a script in Crappy was
to use the :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` method. There are actually more options
available, that can be used in very specific situations.

If you look inside the :meth:`~crappy.blocks.Block.start_all` method, that is
the alias behind :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`, you'll see that it is just made of
three consecutive calls to :meth:`~crappy.blocks.Block.prepare_all`,
:meth:`~crappy.blocks.Block.renice_all` and
:meth:`~crappy.blocks.Block.launch_all`. These methods are aliased to
:ref:`crappy.prepare() <crappy_docs/aliases:crappy.prepare()>`, :ref:`crappy.renice() <crappy_docs/aliases:crappy.renice()>` and :ref:`crappy.launch() <crappy_docs/aliases:crappy.launch()>` for
being called by the user in a script. To get an exact description of what each
of these methods does, refer to :doc:`../concepts/lifecycle_shutdown` and the
:doc:`../architecture` guide. In short, the :ref:`crappy.prepare()
<crappy_docs/aliases:crappy.prepare()>` method initializes all
the Blocks, but does not start the test. For example, after calling this
method, the actuators are powered on, the sensors are configured, and the files
for recording data are created. The :ref:`crappy.renice() <crappy_docs/aliases:crappy.renice()>` method can be
ignored by most users. And the :ref:`crappy.launch() <crappy_docs/aliases:crappy.launch()>` actually starts the test
and is blocking, just like :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`.

Splitting the three operations performed by
:ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` allows code to run
between the :ref:`crappy.prepare() <crappy_docs/aliases:crappy.prepare()>` and :ref:`crappy.launch() <crappy_docs/aliases:crappy.launch()>` methods. This
mostly gives you the capacity to interact with hardware once it is initialized
but the test is not yet started. For example, it is used on some setups to
allow the user to place samples on the device once the motors reach an initial
position. On other setups, we use it to drive an actuator in manual mode, and
only start the test once the desired position is reached and the actuator is
switched back to software-controlled mode.

.. Warning::
   If code is included between the :ref:`crappy.prepare() <crappy_docs/aliases:crappy.prepare()>` and
   :ref:`crappy.launch() <crappy_docs/aliases:crappy.launch()>` methods, there is no warranty that Crappy terminates
   gracefully if this code crashes. Account for failure and its effect on the
   setup before performing an operation in this interval.

As the alternatives :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` are much more difficult to use in a
safe way, and have very few clean use cases, no example will be showed for this
section. We consider that users skilled enough to use these methods safely
should be able to do so without an example. Still, these methods exist and are
part of the API, and as such they are presented in this tutorial section.

7.b. Arguments to the startup method
++++++++++++++++++++++++++++++++++++

The :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>` method, alias to the
:meth:`~crappy.blocks.Block.start_all` method of the class
:class:`~crappy.blocks.Block`, accepts three arguments that can help customize
a bit the behavior of Crappy. They are briefly detailed in this section.

The first possible argument is :py:`'allow_root'`, which is a :obj:`bool`. If
set to :obj:`True`, it permits renicing the Blocks to negative nicenesses. To
do so, the root access will be requested. It only applies on Linux, and if a
negative niceness was attributed to a custom-written Block. It is therefore a
very specific setting that most users can ignore and leave to :obj:`False`.

The second argument is :py:`'log_level'`, that can accept the values
:obj:`logging.DEBUG`, :obj:`logging.INFO`, :obj:`logging.WARNING`,
:obj:`logging.ERROR`, :obj:`logging.CRITICAL`, or :obj:`None`. The given value
corresponds to the maximum level of the log messages displayed in the console
and recorded in the log file. Logging can also be totally disabled, by setting
it to :obj:`None`. This argument does not have many actual use cases, except
maybe for making Crappy silent, or to better spot the errors by disabling the
messages with inferior priority. In the general case, it is advised to leave
this argument to its default value.

Finally, the :py:`'no_raise'` argument is a :obj:`bool` that disables
the exceptions raised at the end of a script. The default behavior of Crappy is
to raise an exception when it stops, if either an unexpected error was raised
during its execution or if a :exc:`KeyboardInterrupt` was caught (script
stopped using :kbd:`Control-c`). The purpose of this behavior is to prevent the
execution of any line of code that would come after :ref:`crappy.start() <crappy_docs/aliases:crappy.start()>`,
since it might not be safe to run it after Crappy has failed or the user
interrupted the test. By setting :py:`'no_raise'` to :obj:`True`, the
exceptions are disabled and Python goes on after Crappy finishes, even if it
crashed. This can allow unsafe follow-up code to run after a failure. The
argument can be changed by users who
prefer to use :kbd:`Control-c` to stop tests but don't want exceptions to be
raised, although we discourage using this strategy.
