=======================
Towards more complexity
=======================

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

In the previous tutorials page, we only used linear data flow patterns. Here,
we're going to **introduce the concept of feedback loops in a script**. The
main idea is that although :ref:`Links` are unidirectional, it is totally
possible to have them form a loop to send back information to a Block. This is
especially useful for driving :ref:`Generator` Blocks, as detailed in
:ref:`a next section <3. Advanced Generator Paths>`. For now, let's look at
the example script given in :ref:`the tutorial section dedicated to the Machine
Block <2.f The Machine Block>`. The :ref:`Fake Machine` Actuator that is used
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

.. literalinclude:: /downloads/tuto_loops.py
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

.. literalinclude:: /downloads/tuto_loops.py
   :language: python
   :emphasize-lines: 39-40, 42, 44-45

Can you see it ? We have both :py:`crappy.link(mot, pid)` and
:py:`crappy.link(pid, mot)`, which means that there is a feedback loop in the
script ! The whole point of this section is to **outline that feedback loops**
**are not only possible in Crappy, but also necessary in some cases**. Most of
the time, it is in situations when a Block needs to modify its output based on
the effect it has on another target Block. You can :download:`download this
feedback loop example </downloads/tuto_loops.py>` to run it locally on your
machine. You can then tune the settings of the motor and see how the PID will
react.

2. Using Modifiers
------------------

One of Crappy's most powerful features is the possibility to **use**
:ref:`Modifiers` **to alter the data flowing through the** :ref:`Links`. The
rationale behind is that the data that a Block outputs might not always be
exactly what you need. For example, data from a sensor might be too noisy and
require some filtering. Or a command might have to be sent to two different
motors, but with an offset on one of them. Such small alterations of the data
should not necessitate to use a new :ref:`Block`, or to modify an existing
one ! To deal with these minor adjustments, we created the Modifier objects.

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

.. literalinclude:: /downloads/tuto_modifiers.py
   :language: python
   :emphasize-lines: 39, 49-51

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
example </downloads/tuto_modifiers.py>` to run it locally on your machine.

3. Advanced Generator Paths
---------------------------

In a previous section we saw how to create everlasting generator paths and ones
ending after a given delay. In many tests, this is not sufficient. Let's imagine
that you have a tensile test setup on which you want to perform force-driven
cyclic stretching. Consider the example from :ref:`the second section
<2.a The Generator Block and its Paths>`. We still want to perform 5 cycles of
stretching and relaxation, still at a 5/60 mm/s pace, but now the condition for
switching from stretching to relaxation is to reach 10kN. This needs to be
somehow indicated to the ``'condition'`` key.

Luckily, this is actually pretty easy to do in Crappy ! The first step is to
make the Generator block aware of the current force value, which means to create
a link from the Machine to the Generator. Remember that the label of the force
output was ``'F(N)'``, so the condition can simply be written :

.. code-block:: python

   {'condition1': 'F(N)>10000'}

Quite elegant, right ? Similarly, the second condition would be :

.. code-block:: python

   {'condition2': 'F(N)<0'}

Why only ``>`` and ``<`` conditions and no ``==`` ? Because it's very unlikely
that the force will take exactly the value 0, so the condition may never be
satisfied even though the force switches from positive to negative.
Consequently, only the ``>`` and ``<`` conditions are valid.

The code including the new link and the new conditions is the following :

.. code-block:: python
   :emphasize-lines: 7,8,30

   import crappy

   if __name__ == '__main__':

       gen = crappy.blocks.Generator(path=[{'type': 'cyclic',
                                            'value1': 5/60, 'value2': -5/60,
                                            'condition1': 'F(N)>10000',
                                            'condition2': 'F(N)<0',
                                            'cycles': 5},
                                           {'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')

       machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

       record = crappy.blocks.Recorder(filename='data.csv',
                                       labels=['t(s)', 'F(N)', 'x(mm)'])

       graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))

       graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))

       crappy.link(gen, machine)

       crappy.link(machine, record)
       crappy.link(machine, graph_pos)
       crappy.link(machine, graph_force)

       crappy.link(machine, gen)

       crappy.start()

This section was quick, but this is actually all there's to know about the
generator path !

4. Dealing with streams
-----------------------

5. Advanced control over the runtime
------------------------------------

6. Writing scripts efficiently
------------------------------

This last section of the Getting started tutorial focuses on how to use Python's
great flexibility to write scripts more efficiently and elegantly. Because
they're within Crappy's particular framework, some of our users tend to forget
that they can actually use all the other Python packages or methods ! Here we're
going to show a few examples of code simplification.

6.a. Using variables
++++++++++++++++++++

Until now in this tutorial all the numeric values needed as arguments in the
blocks have been written explicitly in the block definition. But there's
absolutely no obligation to do so ! Consider the following script :

.. code-block:: python

   import crappy

   if __name__ == '__main__':

       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 1,
                                            'condition': None}])

       machine = crappy.blocks.FakeMachine()

       record = crappy.blocks.Recorder(filename='data_1.csv')

       crappy.link(gen, machine)

       crappy.link(machine, record)

       crappy.start()

It is likely that when the speed value for driving the fake machine changes,
the name of the file where the data is saved should change accordingly. Not very
optimal, right ? Let's improve it very simply by adding a variable for the
speed, that will automatically change both the value in the generator and the
path in the recorder :

.. code-block:: python
   :emphasize-lines: 5,6,9,14

   import crappy

   if __name__ == '__main__':

       speed = 1
       path = 'data' + '_' + str(speed) + '.csv'

       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': speed,
                                            'condition': None}])

       machine = crappy.blocks.FakeMachine()

       record = crappy.blocks.Recorder(filename=path)

       crappy.link(gen, machine)

       crappy.link(machine, record)

       crappy.start()

Now a unique variable handles all the changes implied, more convenient
isn't it ?

6.b. Defining arguments efficiently
+++++++++++++++++++++++++++++++++++

As previously showed in the tutorial, some Crappy objects have to take lists or
dicts as arguments. Until now, we always created these objects explicitly and
inside the blocks definition in order to keep the code simple and easily
understandable. If you followed :ref:`the previous section <6.a. Using
variables>`, you should know that it is also possible to define these objects
before instantiating the block by storing them in variables. This allows
building lists and dicts in a smart and efficient way, as we're now going to
demonstrate taking generator paths as examples.

So let's consider a tensile test, during which we want to perform cyclic
stretching with an increasing distance at each cycle. Let's say that we want 40
cycles with a stretching distance starting at 1mm and increasing by 1mm at each
cycle. This means that we're going to need to give the generator path as a list
containing no less than 40 different dicts, writing it explicitly is not even
an option ! Instead, we're going to take advantage of Python's flexibility and
define the path using a ``for`` loop. This can be done this way :

.. code-block:: python

   import crappy

   if __name__ == '__main__':

       path = []
       n_cycles = 40
       init_stretch = 1
       stretch_step = 1
       for i in range(n_cycles):
           stretch = init_stretch + i * stretch_step
           path.append({'type': 'cyclic',
                        'value1': 5/60, 'value2': -5/60,
                        'condition1': 'x(mm)>' + str(stretch),
                        'condition2': 'x(mm)<0',
                        'cycles': 1})

       gen = crappy.blocks.Generator(path=path)

Look how easy it is now to tune the test protocol with only three variables !
And having 400 or even 4000 cycles instead of 40 would absolutely not be a
problem.

Once you understand the big idea behind the code we just wrote, there's no limit
anymore to the complexity of you generator paths. For instance let's say that we
now want half of the cycles to run at a 3/60 mm/s pace, while the other half
remains at 5/60 mm/s. Look how easy it is to modify the code accordingly :

.. code-block:: python
   :emphasize-lines: 9,10,13-16,18

   import crappy

   if __name__ == '__main__':

       path = []
       n_cycles = 40
       init_stretch = 1
       stretch_step = 1
       speed1 = 5/60
       speed2 = 3/60
       for i in range(n_cycles):
           stretch = init_stretch + i * stretch_step
           if i % 2 == 0:
               speed = speed1
           else:
               speed = speed2
           path.append({'type': 'cyclic',
                        'value1': speed, 'value2': -speed,
                        'condition1': 'x(mm)>' + str(stretch),
                        'condition2': 'x(mm)<0',
                        'cycles': 1})

       gen = crappy.blocks.Generator(path=path)

Hopefully at this point you shouldn't be scared anymore to use include complex
list ou dict arguments in your Crappy scripts. It is even possible to go one
step further in efficiency, what although comes at the cost of readability:

.. code-block:: python
   :emphasize-lines: 10-15

   import crappy

   if __name__ == '__main__':

       n_cycles = 40
       init_stretch = 1
       stretch_step = 1
       speed1 = 5/60
       speed2 = 3/60
       path = [{'type': 'cyclic',
                'value1': speed1 if i % 2 else speed2,
                'value2': -speed1 if i % 2 else -speed2,
                'condition1': 'x(mm)>' + str(init_stretch + i * stretch_step),
                'condition2': 'x(mm)<0',
                'cycles': 1} for i in range(n_cycles)]

       gen = crappy.blocks.Generator(path=path)

Note that if you choose to define the path this way, it doesn't even need to be
defined before the block instantiation and you could simply write
``path=[{...} for ...]``.

6.c. Using other packages
+++++++++++++++++++++++++

In this section of the tutorial, we're going to demonstrate how libraries other
than Crappy can be used before the ``crappy.start()`` call to highly customize
your test protocol. Remember that before this call, your script is just a
regular Python script in which you can literally perform any task you want.
First we're going to use the :mod:`pathlib` module to make the use of a Recorder
cross-platform compatible, and then we're going to use :mod:`psutil` to start a
script only if the current CPU usage is less than a given value. These two
modules are builtins so you can try the examples on your machine if you want !

So first we would like to save data using a Recorder, and in a cross-platform
compatible way. As you may know, paths on Windows use backslashes ``\`` while
paths on Linux and Mac use slashes ``/``, so one solution could be to check the
platform using the :mod:`os` module and to write the path accordingly. A more
elegant solution is to use :mod:`pathlib`, that generates cross-platform
compatible paths.

Let's say we want to save the data to a ``data.csv`` file in a ``Tutorial``
folder located where the ``.py`` script file is. Note that the folder will be
created if it doesn't already exist. The code could look as follows :

.. code-block:: python

   import crappy
   from pathlib import Path

   if __name__ == '__main__':

       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')

       path = Path(__file__).parent / 'Tutorial' / 'data.csv'

       machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

       record = crappy.blocks.Recorder(filename=path,
                                       labels=['t(s)', 'F(N)', 'x(mm)'])

       crappy.link(gen, machine)

       crappy.link(machine, record)

       crappy.start()

Now consider a situation where our computer has limited cooling capacity (a
Raspberry Pi for example), and reduces its performance when heating. In this
case, we want to avoid too high CPU usage, and it might be relevant to condition
the script execution to a low CPU usage. To do so, we'll simply use
the :mod:`psutil` module with an ``if`` statement :

.. code-block:: python
   :emphasize-lines: 3,23-26

   import crappy
   from pathlib import Path
   from psutil import cpu_percent

   if __name__ == '__main__':

       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')

       path = Path(__file__).parent / 'Tutorial' / 'data.csv'

       machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

       record = crappy.blocks.Recorder(filename=path,
                                       labels=['t(s)', 'F(N)', 'x(mm)'])

       crappy.link(gen, machine)

       crappy.link(machine, record)

       if cpu_percent(interval=1) > 50:
           print("Crappy not started, CPU usage is too high !")
       else:
           crappy.start()

As you can see, there are countless ways of customizing your scripts to include
unique features. This is a good transition towards :ref:`the second tutorial
<Creating and using custom objects in Crappy>`, that pushes customization even
further by presenting how to create and use your own Crappy objects !

6.d. Using Crappy objects outside of a Crappy test
++++++++++++++++++++++++++++++++++++++++++++++++++

To conclude this tutorial, we're going to see how Crappy objects can actually be
instantiated outside the context of a test and used as tools. Here we'll
consider that starting a Crappy test means executing the ``crappy.start()``
command. So how does this work ?

If you have a look at the :ref:`second tutorial <Creating and using custom
objects in Crappy>`, you'll see that camera, inout and actuator objects are simply
classes performing elementary actions on a given device. So if you instantiate
these objects, you can just perform the same basic actions as Crappy would
(moving an actuator, grabbing a video frame, etc.) except here you need to call
the methods yourself instead of Crappy automatically calling them for you.

And why would you do that ? Because if Crappy's framework is truly nice for
running complex tests, it is a bit cumbersome when you only want to perform
simple tasks informally. As an example, we'll use a Crappy camera for taking
just one picture.

So let's get started ! Taking a single picture will of course be done using
a camera. You can have a look at :ref:`this section <4. Custom Cameras>` of the
second tutorial to see how the :ref:`Camera` block should be used in Crappy.
Here we're not going to use the camera block, but rather one of the
:ref:`Cameras` objects that are normally used as "tools" by the camera block.
For instantiating the object, we simply need to write :

.. code-block:: python

   import crappy

   if __name__ == '__main__':

       cam = crappy.camera.<Name_of_the_camera>(<args>, <kwargs>)

If your computer has a webcam, you can use the :ref:`Webcam` camera. Otherwise,
the :ref:`Fake Camera` doesn't require any hardware (it also doesn't take any
actual picture, of course). Cameras usually take no arguments but inouts and
actuators may, so be sure to check the corresponding documentation. Let's
suppose you have a webcam, then the instantiation looks like :

.. code-block:: python

   import crappy

   if __name__ == '__main__':

       cam = crappy.camera.Webcam()

Then you need to call the ``open`` method to initialize the camera. This also
applies to inouts and actuators. This method takes no arguments.

.. code-block:: python
   :emphasize-lines: 6

   import crappy

   if __name__ == '__main__':

       cam = crappy.camera.Webcam()
       cam.open()

Now all you need to do is grab a single frame, which is equivalent to taking a
picture. On all cameras this will be done by calling the ``get_image`` method.
It takes no argument, and the image is the second object returned by the method.

.. code-block:: python
   :emphasize-lines: 8

   import crappy

   if __name__ == '__main__':

       cam = crappy.camera.Webcam()
       cam.open()

       img = cam.get_image()[1]

The image is returned as a :mod:`numpy` array, and now you're free to do
whatever you want with it ! You can for instance save it, or simply display it
as we're going to do now :

.. code-block:: python
   :emphasize-lines: 2,10-11

   import crappy
   from cv2 import imshow, waitKey

   if __name__ == '__main__':

       cam = crappy.camera.Webcam()
       cam.open()

       img = cam.get_image()[1]
       imshow('picture', img)
       waitKey(3000)

We also shouldn't forget to close the camera before exiting the program. This
also applies to inouts and actuators.

.. code-block:: python
   :emphasize-lines: 13

   import crappy
   from cv2 import imshow, waitKey

   if __name__ == '__main__':

       cam = crappy.camera.Webcam()
       cam.open()

       img = cam.get_image()[1]
       imshow('picture', img)
       waitKey(3000)

       cam.close()

And that's it ! You should now be able to visualize the picture you just took.
It will last 3 seconds on the screen then close. Notice that there's no
``crappy.start()`` method, we're not actually running a Crappy test program
here.

You can perform similar actions with inouts and actuators, for example if you
want to acquire one single data point from a sensor or if you want to set an
output to a given value. The ``open`` and ``close`` methods would remain, still
without any argument, while the ``get_image`` method would change according to
the object you're using. Of course the name of the object and the arguments to
give it would also differ.
