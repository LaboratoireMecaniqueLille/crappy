=======================
Towards more complexity
=======================

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

.. role:: py(code)
  :language: python
  :class: highlight

In this second page of the tutorials, we're going to cover **topics that will**
**help you customize your scripts and write them more efficiently**. The tools
and concepts presented here are really not that advanced or complicated, and
will be needed by anyone who wants to write scripts with a minimum of
complexity. So, make sure to read this page until the end !

1. Using feedback loops
-----------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

In the previous tutorials page, we only used linear data flow patterns. Here,
we're going to **introduce the concept of feedback loops in a script**. The
main idea is that although :ref:`Links` are unidirectional, it is totally
possible to have them form a loop to send back information to a Block. This is
especially useful for driving :ref:`Generator` Blocks, as detailed in
:ref:`a next section <3. Advanced Generator condition>`. For now, let's look at
the example script given in :ref:`the tutorial section dedicated to the Machine
Block <2.f. The Machine Block>`. The :ref:`Fake Machine` Actuator that is used
takes its commands as a voltage, which is quite unsatisfying since the achieved
speed will vary depending on the characteristics of the motor. Instead, it
would be preferable to send speed commands, and to somehow have the motor adapt
and reach this speed.

To achieve this behavior, a possibility is to use the :ref:`PID` Block. It will
receive on the one hand the target speed, and on the other hand the current
speed of the motor. Based on these inputs, it will generate a voltage command
to send to the Machine Block driving the Fake Motor. If the PID is well set,
the measured speed should reach the target value after some time ! Here's how
the Blocks look like :

.. literalinclude:: /downloads/more_complexity/tuto_loops.py
   :language: python
   :emphasize-lines: 7-15, 17-24, 26-27, 29-37
   :lines: 1-38

As you can see, the Generator sends the target speed under the label
:py:`'target_speed'`, the Machine Block takes the :py:`'voltage'` label as a
command and returns the :py:`'actual_speed'`, and the PID Block takes both tha
:py:`'target_speed'` and :py:`'actual_speed'` labels as inputs and returns the
:py:`'voltage'` label. There is also a Grapher Block plotting both the
:py:`'target_speed'` and :py:`'actual_speed'` labels. Also, notice the
:py:`'spam'` argument of the Generator Block, that ensures that the Block sends
the command at each loop, for a nice display on the graph. Let's now link the
Block together consistently :

.. literalinclude:: /downloads/more_complexity/tuto_loops.py
   :language: python
   :emphasize-lines: 39-40, 42, 44-45

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

Can you see it ? We have both :py:`crappy.link(mot, pid)` and
:py:`crappy.link(pid, mot)`, which means that there is a feedback loop in the
script ! The whole point of this section is to **outline that feedback loops**
**are not only possible in Crappy, but also necessary in some cases**. Most of
the time, it is in situations when a Block needs to modify its output based on
the effect it has on another target Block. You can :download:`download this
feedback loop example </downloads/more_complexity/tuto_loops.py>` to run it
locally on your machine. You can then tune the settings of the motor and see
how the PID will react.

2. Using Modifiers
------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

One of Crappy's most powerful features is the possibility to **use**
:ref:`Modifiers` **to alter the data flowing through the** :ref:`Links`. The
rationale behind is that the data that a Block outputs might not always be
exactly what you need. For example, data from a sensor might be too noisy and
require some filtering. Or a command might have to be sent to two different
motors, but with an offset on one of them. Such small alterations of the data
should not necessitate to use a new :ref:`Block`, or to modify an existing
one ! To deal with these minor adjustments, we created the
:class:`~crappy.modifier.Modifier` objects.

The principle of Modifiers is that each Modifier is attached to a given Link.
**Every time a Block wants to send data through the Link, the Modifier alters**
**it before it gets sent**. A same Link can have several Modifiers attached, in
which case they are called in the same order as they are given. Unlike the
operations performed by the Blocks, the ones carried out by the Modifiers are
not optimized at all. Therefore, **Modifiers should only be used for simple**
**tasks**. The syntax for adding Modifiers is very simple, let's get familiar
with it in an example !

Starting from the example of the previous section, we now want to know the
current position of the motor. To calculate this value, we just have to
integrate the measured speed over time. This is numerically a very simple
operation, since it is equivalent to a sum. It is thus a perfect job for a
Modifier ! Luckily, Crappy already implements the :ref:`Integrate` Modifier for
integrating a signal over time. Let's add it on a Link starting from the
Machine Block and pointing towards a new :ref:`Grapher` for the position :

.. literalinclude:: /downloads/more_complexity/tuto_modifiers.py
   :language: python
   :emphasize-lines: 39, 49-51

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

As you can see, the Modifiers are expected to be given to the :py:`'modifier'`
argument of the :func:`crappy.link()` function. Each Modifier has to be
instantiated, and might require arguments. To know what the effect of a
Modifier is, and which argument it takes, refer to the :ref:`Modifiers` section
of the API. Here, the chosen Modifier is :class:`~crappy.modifier.Integrate`.
It must be given the name of the label to integrate, and here the name of the
label carrying the integral value is also specified. This new label is added to
the data flowing through the Link, and can then be used by the downstream
Block ! In the case of the Integrate Modifier, all the other labels are
preserved.

As illustrated with this example, Modifiers are a simple yet powerful way to
tune the data flowing through the Links. As the Modifiers distributed with
Crappy will surely not cover all the possible use cases, we strongly encourage
you to have a look at the section detailing :ref:`how to code your own
Modifiers <1. Custom Modifiers>`. You can :download:`download this Modifier
example </downloads/more_complexity/tuto_modifiers.py>` to run it locally on
your machine. The Modifiers distributed with Crappy are also showcased in the
`examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/tree/master/examples/modifiers>`_.

3. Advanced Generator condition
-------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

In :ref:`a previous section <2.a. The Generator Block and its Paths>`, the
:ref:`Generator` Block and its :ref:`Generator Paths` were introduced. In that
section, two possible syntax were given for the :py:`'condition'` key of a
Path :obj:`dict`. The value :obj:`None` can be given, in which case the Path
never ends. Alternatively, a :obj:`str` in the format :py:`'delay=xx'` can be
given, in which case the Path ends after the specified delay. There is actually
an other way to specify the stop condition, that we are going to detail in this
section.

Getting right to the point, the third way to specify a stop condition is to
give a :obj:`str` in the format :py:`'label>value'` or :py:`'label<value'`.
Replace :py:`'label'` with the name of the label to monitor, and :py:`'value'`
with a numerical value to compare the label with. The principle of this type
of condition is that the Generator should be sent the label to monitor. **At**
**each loop, it checks if any point of the received label is below (or above)**
**the given threshold**. If that's the case, the stop condition is met and the
Path ends. The reason why this type of stop condition was not introduced in the
section dedicated to Generators is that it requires the concept of feedback
loop, that is only introduced :ref:`earlier on this page
<1. Using feedback loops>` !

.. Note::
   In the stop conditions given as :obj:`str`, you can freely add spaces around
   the *=*, *<* and *>* characters. The condition will still be recognized in
   the same way.

.. Note::
   And what about an :py:`'=='` condition ? As in a vast majority of situations
   users are dealing with :obj:`float`, it is very unlikely that a label would
   be exactly equal to the given threshold ! Then what about something using
   :obj:`math.isclose` ? It could indeed come in use, but a similar behavior
   can be obtained using the *<* condition, a
   :class:`~crappy.modifier.Modifier`, and the :obj:`abs` function !

Let's now use such a stop condition in an example. In the very first example of
the tutorials, a *delay* condition was used for stopping the script. It was
conveniently chosen so that the stop condition is met short after the sample
breaks. But if the elongation pace is changed, the delay will for sure not
match anymore with the sample failure ! Instead, we can use the new type of
condition introduced here to always have the test stop short before the sample
breaks, no matter the elongation speed. The code is as follows :

.. literalinclude:: /downloads/more_complexity/advanced_generator.py
   :language: python
   :emphasize-lines: 9, 26

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

You can :download:`download this advanced Generator example
</downloads/more_complexity/advanced_generator.py>` to run it locally on your
machine. Try to modify the value of the speed command, and see how the script
always stops at the given condition. With the new type of stop condition for
the Generator, you are now ready to use this block to its full extent !

.. Note::
   There is actually one more possibility to define custom stop conditions,
   that is much more advanced and is described in :ref:`a later tutorial
   section <1. Custom Generator Paths>`.

4. Dealing with streams
-----------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

In :ref:`the tutorial section dedicated to IOBlocks <2.e. The IOBlock Block>`,
only the regular usage mode of the :ref:`IOBlock` was presented. In this mode,
the data points are acquired from the :ref:`In / Out` object one by one, which
results in a limited data rate usually around a few hundred samples per second.
To circumvent this limitation, another acquisition mode was added for InOuts
supporting it : the *streamer* mode. **In streamer mode, the data rate of**
**InOut objects can get as high as a few kHz** ! This comes however at the cost
of direct compatibility with most of the Blocks, as detailed below.

As you may have guessed, in *streamer* mode the data points are acquired and
returned as chunks rather than individually. This means that the IOBlock sends
multiple points at once to the downstream Blocks, which is totally unexpected
for most Blocks. Therefore, only two objects in Crappy are natively compatible
with the *streamer* mode : the :ref:`HDF Recorder` Block and the :ref:`Demux`
Modifier. Let's see with an example how to use these objects together !

The first requirement when using the *streamer* mode is to use an InOut
supporting this mode. To know if that is the case, you need to consult the
documentation for that InOut. Luckily, the :class:`~crappy.inout.FakeInOut`
InOut supports it, so we'll use it here for the demo. And second, the
*streamer* mode needs to be enabled on the IOBlock, via the :py:`'streamer'`
argument. If these two conditions are met, the *streamer* mode is enabled ! The
beginning of the example script looks as follows :

.. literalinclude:: /downloads/more_complexity/tuto_streamer.py
   :language: python
   :emphasize-lines: 7-10, 12-14
   :lines: 1-19

Notice how the :py:`'streamer'` is indeed set on the IOBlock. Except for that,
the syntax for the IOBlock is the same as usual, and the HDFRecorder Block is
also very close to the regular :ref:`Recorder` one. The differences are that
instead of multiple labels to record, it only expects one stream label
containing all the data at once. It also requires the expected data format to
be specified. Now that the involved Blocks are instantiated, it is time to link
them together :

.. literalinclude:: /downloads/more_complexity/tuto_streamer.py
   :language: python
   :emphasize-lines: 20-24

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` and
   :mod:`psutil` Python modules installed.

Compared to the regular IOBlock usage, this is when things get a bit more
complicated ! As the IOBlock and HDFRecorder are both meant to handle stream
data, they can be linked together in a normal way. However, the :ref:`Grapher`
Block cannot accept stream data, so the Demux Modifier must be added to their
Link ! Basically, this :class:`~crappy.modifier.Modifier` accepts stream data
as an input and outputs regular data usable by most other Blocks. Since streams
might have a very high data rate, most of the information is discarded in the
process to avoid overflowing the Link. Still, it outputs values that can be
used for plotting or any other application. Here, the data from the IOBlock
should be successfully displayed on the graph even though it originates from a
stream.

You can :download:`download this streamer example
</downloads/more_complexity/tuto_streamer.py>` to run it locally on your
machine. The *streamer* mode is neither as used nor as well documented as the
regular operation mode, so do not hesitate to request help on the GitHub page
if you would have trouble using it !

5. Writing scripts efficiently
------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

Because Crappy requires script with a specific syntax to run, users may forget
that they can still make use of Python's great flexibility and tools even
inside scripts for Crappy ! This section is just a short and surely not
exhaustive reminder of what is possible to do with Python in a script written
for Crappy. **Follow the given hints to write your scripts in a more**
**efficient and elegant way** !

5.a. Use variables
++++++++++++++++++

When providing arguments to a Block or any other object, remember that you can
use variables instead of plain text or numbers. It will make your scripts
easier for yourself and others to read and to modify.

Do not write :

.. code-block:: python

   record_pos = crappy.blocks.Recorder('tests/example/data/pos.csv')

   record_force = crappy.blocks.Recorder('tests/example/data/force.csv')

   record_extenso = crappy.blocks.Recorder('tests/example/data/ext.csv')

But write instead :

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
big objects, that can be generated following a known patter. In that case,
using loops will save many lines and avoid typos. If you're familiar with the
concept of comprehension, it can also help you make your code even more compact
than with loops !

Don not write :

.. code-block:: python

   gen = crappy.blocks.Generator([
       {'type': 'Constant', 'value': 0, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 1, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 2, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 3, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 4, 'condition': 'delay=5'},
       {'type': 'Constant', 'value': 5, 'condition': 'delay=5'}])

But write instead :

.. code-block:: python

   path = list()
   for i in range(6):
       path.append({'type': 'Constant', 'value': i, 'condition': 'delay=5'})

   gen = crappy.blocks.Generator(path)

Or even more concise :

.. code-block:: python

   gen = crappy.blocks.Generator([
       {'type': 'Constant', 'value': i, 'condition': 'delay=5'}
       for i in range(6)])

5.c. Use other packages
+++++++++++++++++++++++

Even though in all the examples and tutorials Crappy is the only package to be
imported, it is totally fine to import and use other packages in your script !
This can be convenient for performing operations before Crappy starts, or after
it ends. One of the best example of a module that can come in use in a script
is :mod:`pathlib`. It handles file paths in a cross-platform compatible way, so
that you don't have to care about */* and *\\* if you want to make your code
runnable on both Linux and Windows. It is also part of the standard library of
Python, so it doesn't need to be installed.

Do not write :

.. code-block:: python

   record_pos = crappy.blocks.Recorder('tests/example/data/pos.csv')

   record_force = crappy.blocks.Recorder('tests/example/data/force.csv')

   record_extenso = crappy.blocks.Recorder('tests/example/data/ext.csv')

Using :mod:`pathlib`, write instead :

.. code-block:: python

   from pathlib import Path

   base = Path('tests/example/data')

   record_pos = crappy.blocks.Recorder(base / 'pos.csv')

   record_force = crappy.blocks.Recorder(base / 'force.csv')

   record_extenso = crappy.blocks.Recorder(base / 'ext.csv')

6. Using Crappy objects outside of a Crappy test
------------------------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

In the new section of the tutorial, let's see how you can use the classes
distributed with Crappy to interact freely with hardware outside the context of
a Crappy test (i.e. without calling :ref:`crappy.start()` or an equivalent
method). But first, why would you do that ? Well, while Crappy is a nice
framework for running entire experimental protocols, it is a bit cumbersome if
you only want to acquire one image or one data point from a sensor. That's why
this "hack" is presented here ! Note that it is truly not an intended feature
of Crappy, but rather a consequence of its implementation.

In Crappy, the :ref:`In / Out`, :ref:`Actuators` and :ref:`Cameras` objects
each implement the code needed to interact with a specific equipment. They make
sure that this code is organized and can be called in a standard way, so that
it can be used by the :ref:`IOBlock`, :ref:`Machine` and :ref:`Camera` Blocks
respectively. Knowing how to properly call the corresponding code, **it is**
**thus possible to use these classes to directly interface with hardware**
**outside the context of a Crappy test**.

To learn more about the mandatory and optional methods that each class can
implement, you should refer to the :ref:`Creating and using custom objects in
Crappy` page of the tutorials. Here, a very basic example will be used to
demonstrate how a :class:`~crappy.camera.Camera` object can be used for
acquiring and visualizing images. The :class:`~crappy.camera.FakeCamera` will
be used, so that no hardware is required to run the script. The trick to use
this class directly is to instantiate it, without using a :ref:`Camera` Block.
Let's write the first part of the script :

.. literalinclude:: /downloads/more_complexity/outside_test.py
   :language: python
   :emphasize-lines: 7-9
   :lines: 1-3, 5-10

As you can see, the FakeCamera is directly instantiated whereas normally its
name would have been given as an argument to a Camera Block. Then, the
:meth:`~crappy.camera.FakeCamera.open` and
:meth:`~crappy.camera.FakeCamera.get_image` methods are called for respectively
initializing the Camera and acquiring an image. The detail of the methods
exposed by the FakeCamera and their exact syntax have to be looked up in the
API. Notice how the arguments to provide to the FakeCamera are passed to the
*open* method, instead of being given to Camera Block. As you may have guessed,
the script above initializes a FakeCamera, and acquires one image from it. As
This is not very interesting to watch, let's add some visualization :

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
to use a Camera object without ever calling :ref:`crappy.start()`. Note that
the same principle applies to InOut and Actuator objects, the Camera was only
used here because it is more visual.

7. Advanced control over the runtime
------------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

For the last section of this tutorial page, let's see how you can achieve a
finer-grained control over Crappy's runtime. **There are two ways to control**
**Crappy in a more accurate way : passing arguments to** :ref:`crappy.start()`,
**and/or using alternative startup methods**.

7.a. Alternative startup methods
++++++++++++++++++++++++++++++++

So far, the only option that was presented for starting a script in Crappy was
to use the :ref:`crappy.start()` method. There are actually more options
available, that can be used in very specific situations.

If you look inside the :meth:`~crappy.blocks.Block.start_all` method, that is
the alias behind :ref:`crappy.start()`, you'll see that it is just made of
three consecutive calls to :meth:`~crappy.blocks.Block.prepare_all`,
:meth:`~crappy.blocks.Block.renice_all` and
:meth:`~crappy.blocks.Block.launch_all`. These methods are aliased to
:ref:`crappy.prepare()`, :ref:`crappy.renice()` and :ref:`crappy.launch()` for
being called by the user in a script. To get an exact description of what each
of these methods do, refer to the :ref:`Developers information` section of
the documentation. In short, the :ref:`crappy.prepare()` method initializes all
the Blocks, but does not start the test. For example, after calling this
method, the actuators are powered on, the sensors are configured, and the files
for recording data are created. The :meth:`crappy.renice()` method can be
ignored by most users. And the :meth:`crappy.launch()` actually starts the test
and is blocking, just like :meth:`crappy.start()`.

But why would you want to split up the three methods of
:meth:`crappy.start()` ? By doing so, you gain the possibility to add some code
between the :meth:`crappy.prepare()` and :meth:`crappy.launch()` methods. This
mostly gives you the capacity to interact with hardware once it is initialized
but the test is not yet started. For example, it is used on some setups to
allow the user to place samples on the device once the motors reach an initial
position. On other setups, we use it to drive an actuator in manual mode, and
only start the test once the desired position is reached and the actuator is
switched back to software-controlled mode.

.. Warning::
   If code is included between the :meth:`crappy.prepare()` and
   :meth:`crappy.launch()` methods, there is no warranty that Crappy terminates
   gracefully in case this code crashes ! Be extremely cautious when performing
   operations that can potentially fail, and make sure to understand what the
   effects would be on your setup !

As the alternatives :meth:`crappy.start()` are much more difficult to use in a
safe way, and have very few clean use cases, no example will be showed for this
section. We consider that users skilled enough to use these methods safely
should be able to do so without an example. Still, these methods exist and are
part of the API, and as such they are presented in this tutorial section.

7.b. Arguments to the startup method
++++++++++++++++++++++++++++++++++++

The :meth:`crappy.start()` method, alias to the
:meth:`~crappy.blocks.Block.start_all` method of the class
:class:`~crappy.blocks.Block`, accepts three arguments that can help customize
a bit the behavior of Crappy. They are briefly detailed in this section.

The first possible argument is :py:`'allow_root'`, which is a :obj:`bool`. If
set to :obj:`True`, it allows renicing the Blocks to negative nicenesses. To
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

Finally, the :py:`'no_raise'` argument is a :obj:`bool` that allows to disable
the exceptions raised at the end of a script. The default behavior of Crappy is
to raise an exception when it stops, if either an unexpected error was raised
during its execution or if a :exc:`KeyboardInterrupt` was caught (script
stopped using :kbd:`Control-c`). The purpose of this behavior is to prevent the
execution of any line of code that would come after :meth:`crappy.start()`,
since it might not be safe to run it after Crappy has failed or the user
interrupted the test. By setting :py:`'no_raise'` to :obj:`True`, the
exceptions are disabled and Python goes on after Crappy finishes, even if it
crashed. **Use this feature with caution, as it can lead to unexpected or**
**even unsafe behavior** ! This argument can be changed by users who would
prefer to use :kbd:`Control-c` to stop tests but don't want exceptions to be
raised, although we discourage using this strategy.
