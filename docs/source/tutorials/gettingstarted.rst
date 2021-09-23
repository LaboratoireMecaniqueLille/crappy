========================================
Getting started : writing test protocols
========================================

0. General concepts
-------------------

In this short tutorial we're going to learn the very basics of writing scripts
for running test protocols with Crappy. Only a beginner's level in Python is
required, don't worry !

0.a. Blocks
+++++++++++

In Crappy, even the most complex setups can be described with only two elements
: the **blocks** and the **links**. The blocks are responsible for managing
data. There are many different block types, that all have a unique function.
Some will acquire data, others transform it, or use it to drive hardware, etc.
As the blocks perform very specific tasks an assay is always made up of at
least two blocks, but there can be many more. Blocks either take data as an
input, or output data, or both.

0.b. Links
++++++++++

Blocks are always blissfully ignorants of each other. So the links are there to
allow data transfers between them. A link is established between two blocks and
is oriented. Establishing a link between block 1 and block 2 means that block 2
will be aware of all of block 1's outputs. Because the link is oriented, block 1
will however not be aware of block 2's outputs. There is no limit in the number
of links towards and from a given block.

0.c. Labels
+++++++++++

Data transiting between the blocks through the links is always labeled. Labels
are simply names associated with a given stream of data. Let's say that block 1
outputs three data stream labeled ``'time', 'Force', 'Position'``, and is
linked with block 2 that only takes two inputs. As we said, block 2 is aware of
all of block 1's outputs and thus needs a way to differentiate them. Thanks to
labels, the user can simply specify in the arguments of block 2 to only listen
to labels ``'time', 'Position'`` (for example). The data stream labeled
``'Force'`` will be lost to block 2, but maybe block 1 is also linked with a
block 3 that's using it !

1. Understanding Crappy's syntax
--------------------------------

Before writing a Python script for Crappy you need of course to :ref:`install
Crappy <Installation>`, and then to open a new ``.py`` file. The first step for
writing the script is to import the ``crappy`` module :

.. code-block:: python

   import crappy

Then you need to choose which blocks will be used in your script, depending on
how you want your setup to run. For this first example, let's say that we want
to acquire both position and force vs time from a tensile test machine, plot the
data and save it. As driving an entire machine is ambitious for a first example
and not everyone has a machine available, let's cheat and use the :ref:`Fake
machine` bloc instead. We also need a :ref:`Recorder` bloc for saving the data,
and a :ref:`Grapher` bloc for plotting it. There will also be a :ref:`Generator`
block for driving the fake machine, but it will be covered in the next section.

We must now add these blocs to our script. The syntax is as follows :

.. code-block:: python

  <chosen_name> = crappy.blocks.<Block_name>(<arguments>)

``<chosen_name>`` is a unique name you can freely choose. ``<Block_name>`` is
the name of the block in Crappy's block library. The possible ``<arguments>``
differ for every block, the only way to know for sure is to look at the
documentation !

.. Note::
   To easily access the documentation, simply type in a Python terminal :

     >>> import crappy
     >>> crappy.doc()

In our specific case, the script could be as follows :

.. code-block:: python
   :emphasize-lines: 3-17

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
       record = crappy.blocks.Recorder(filename='data.csv',
                                       labels=['t(s)', 'F(N)', 'x(mm)'])
  
       graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  
       graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))

Now that the blocks are there, they need to be linked together. The generator
provides the speed command to the fake machine, so it needs to be lined to it.
And then the fake machine outputs the time, force and position to the graphers
and the recorder, so links are also needed between them.

The syntax for adding a link is as follows :

.. code-block:: python

  crappy.link(<block1>, <block2>)

Here ``<block1>`` and ``<block2>`` are the names you assigned to the blocks. So
in our program this would give :

.. code-block:: python
   :emphasize-lines: 19-23

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
       record = crappy.blocks.Recorder(filename='data.csv',
                                       labels=['t(s)', 'F(N)', 'x(mm)'])
  
       graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  
       graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))
  
       crappy.link(gen, machine)
  
       crappy.link(machine, record)
       crappy.link(machine, graph_pos)
       crappy.link(machine, graph_force)

Llet's have a look at how data is exchanged between the blocks. First an
``'input_speed'`` signal is created by the generator, then transmitted to the
fake machine. It uses it as an input, and outputs four signals with the labels
``'t(s)', 'F(N)', 'x(mm)', 'Exx(%)'``. This is not obvious reading the script,
but it is written in the block documentation ! These signals are then
transmitted to the recorder and the graphers, that are using them as inputs and
output nothing.

Notice the syntax in the arguments of each block. In each block an argument
specifies either the labels of the inputs or the labels of the outputs (except
for the fake machine which is a special block). It is easy to keep track of the
information flow throughout the code ! You can also notice the ``filename``
argument of the saver block, indicating where to save the data. Here it will
be in a new file located where our ``.py`` file is.

So is our program reading to run ? Almost ! We just need to add the method that
tells Crappy to start the test :

.. code-block:: python

  crappy.start()

And here's the final code :

.. code-block:: python
   :emphasize-lines: 25

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
       record = crappy.blocks.Recorder(filename='data.csv',
                                       labels=['t(s)', 'F(N)', 'x(mm)'])
  
       graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  
       graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))
  
       crappy.link(gen, machine)
  
       crappy.link(machine, record)
       crappy.link(machine, graph_pos)
       crappy.link(machine, graph_force)
  
       crappy.start()

As you run it, you can see the graphers displaying the data in real-time. The
data is also being saved, which you can check after the program ends. How does
it end by the way ? In Crappy there are three ways a program can end: either an
error occurs and the program crashes, or a generator is done generating signal
and stops the program, or the user hits the CTRL+C key. So whenever you want to
stop the program simply type CTRL+C on the keyboard and it will properly
terminate.

2. Adding signal generators
---------------------------

Most of the time actuators need to be driven according to a pre-determined
scheme, which thus needs to be given by the user to the program. In Crappy, this
is achieved using the :ref:`Generator` block. This section specifically
illustrates the syntax for building signals with a Generator. We'll start from
the example described in the previous section.

Previously, we were simply driving the :ref:`Fake machine` at a constant pace.
Let's say that we now want to perform cyclic stretching and relaxation (5
cycles), and then stretch the sample until failure at a constant pace. The only
thing that needs to be changed in our previous script is actually the ``path``
argument in the Generator block !

This argument must be a :obj:`list` containing :obj:`dict`. Each :obj:`dict`
provides information for generating signal following a specific pattern. All
the patterns can be found in :ref:`the generator path section <generator path>`.
The dicts in the list are considered successively by the Generator, until
there's no dict left in which case the program stops.

We previously used the :ref:`constant` pattern, which is why we specified
``'type': 'constant'``. The only argument characterizing a constant is its
value, specified by ``'value': '5/60'``. The third key entered is
``'condition'``. It tells Crappy which condition must be satisfied for the
Generator to move on to the next dict. Here it is simply :obj:`None`, the signal
will be generated indefinitely if the program doesn't stop.

Now for a cyclic stretching, we have to use the :ref:`cyclic` pattern. It
alternatively switches between two constant signals, here allowing to impose
either a positive or a negative speed. To know what arguments it takes, we need
to refer to the documentation. So we have to specify the ``'value1'`` and the
``'value2'``, as well as the ``'condition1'`` and ``'condition2'``. When the
condition associated with the value currently generated is met, it switches to
the other value. the fifth argument, ``'cycles'``, indicates how many cycles
should be run before the Generator switches to the next dict.

For the two speed values, let's stick to the 5/60 mm/s we previously had. For
the cycles, we said we wanted 5 of them. And regarding the condition, let's say
we want our cycles to last 4 seconds, so 2 seconds stretching and 2 seconds
relaxing. The syntax is as follows: ``'condition1': 'delay=2'``. The dict
for the cyclic pattern is thus :

.. code-block:: python

   {'type': 'cyclic',
    'value1': 5/60, 'value2': -5/60,
    'condition1': 'delay=2', 'condition2': 'delay=2',
    'cycles': 5}

We still need to add a second dictionary for the second part of the assay, the
monotonic stretching. This is actually what was performed in the last section,
so let's just reuse the same dict. Our generator block now looks like this :

.. code-block:: python

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'cyclic',
                                            'value1': 5/60, 'value2': -5/60,
                                            'condition1': 'delay=2',
                                            'condition2': 'delay=2',
                                            'cycles': 5},
                                           {'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')

Now you can try to run the script and see the changes. The program still needs
to be stopped using CTRL+C otherwise it will run forever.

.. code-block:: python
   :emphasize-lines: 15-30

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'cyclic',
                                            'value1': 5/60, 'value2': -5/60,
                                            'condition1': 'delay=2',
                                            'condition2': 'delay=2',
                                            'cycles': 5},
                                           {'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
       record = crappy.blocks.Recorder(filename='data.csv',
                                       labels=['t(s)', 'F(N)', 'x(mm)'])
  
       graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))
  
       graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))
  
       crappy.link(gen, machine)
  
       crappy.link(machine, record)
       crappy.link(machine, graph_pos)
       crappy.link(machine, graph_force)
  
       crappy.start()

So now you should be able to build any protocol, it is actually just a matter
of adding dictionaries to the path list ! The many path types we provide should
be more than sufficient for most protocols.

3. Towards more complexity
--------------------------

3.a. Loops
++++++++++

In the previous sections we only used linear data flow patterns. Here we're
going to spice it up by introducing loop patterns in our script. To this end
let's consider a new example. No we simply want to drive a DC motor with known
properties to a target speed, but the motor takes Volts as an input so we need
to setup a control for converting the speed command into a voltage input.

Since not everyone has a DC motor at home, let's use a :ref:`Fakemotor` object
instead. It simply simulates the dynamic behavior of a DC motor. We're going to
use a `PID controller <https://en.wikipedia.org/wiki/PID_controller>`_ for
converting the speed command into Volts, implemented in the :ref:`PID` block. We
also need a :ref:`Generator` for generating the speed command, and a
:ref:`Grapher` for plotting the command speed next to the actual speed.

The beginning of the code should be fairly understandable if you followed the
previous sections:

.. code-block:: python

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator([
             {'type': 'constant', 'value': 1000, 'condition': 'delay=3'},
             {'type': 'ramp', 'speed': 100, 'condition': 'delay=5',
              'cmd': 0},
             {'type': 'constant', 'value': 1800, 'condition': 'delay=3'},
             {'type': 'constant', 'value': 500, 'condition': 'delay=3'},
             {'type': 'sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
              'condition': 'delay=15'}], spam=True,
                                     cmd_label='command_speed')
  
       mot = crappy.blocks.Machine([{'type': 'Fake_motor',
                                     'cmd': 'voltage',
                                     'mode': 'speed',
                                     'speed_label': 'actual_speed',
                                     'kv': 1000,
                                     'inertia': 4,
                                     'rv': .2,
                                     'fv': 1e-5}])
  
       graph = crappy.blocks.Grapher(('t(s)', 'speed'), ('t(s)', 'cmd'))
  
       pid = crappy.blocks.PID(kp=0.038,
                               ki=2,
                               kd=0.05,
                               out_max=10,
                               out_min=-10,
                               i_limit=0.5,
                               target_label='command_speed',
                               labels=['t(s)', 'voltage'],
                               input_label='actual_speed')

Notice the ``spam`` argument in the Generator, it is meant to ensure that the
command is resent again and again even if it is constant (the default behavior
is not to resend it if it doesn't change). Also notice the syntax for
instantiating the fake motor: it is pretty similar to the Generator paths. This
syntax allows controlling several actuators from a same :ref:`Machine` block
(one dict per actuator), which is convenient if they need to be synchronized.

In order for the PID to run it needs to know the speed command and the actual
speed, so it needs inputs from the Generator and the Machine (the fake motor
outputs its current speed). And the Machine needs a voltage input, which is
outputted by the PID block. Finally the grapher needs to know the current speed
and the command speed, just like the PID. So let's add the appropriate links :

.. code-block:: python
   :emphasize-lines: 37-45

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator([
             {'type': 'constant', 'value': 1000, 'condition': 'delay=3'},
             {'type': 'ramp', 'speed': 100, 'condition': 'delay=5',
              'cmd': 0},
             {'type': 'constant', 'value': 1800, 'condition': 'delay=3'},
             {'type': 'constant', 'value': 500, 'condition': 'delay=3'},
             {'type': 'sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
              'condition': 'delay=15'}], spam=True,
                                     cmd_label='command_speed')
  
       mot = crappy.blocks.Machine([{'type': 'Fake_motor',
                                     'cmd': 'voltage',
                                     'mode': 'speed',
                                     'speed_label': 'actual_speed',
                                     'kv': 1000,
                                     'inertia': 4,
                                     'rv': .2,
                                     'fv': 1e-5}])
  
       graph = crappy.blocks.Grapher(('t(s)', 'command_speed'),
                                     ('t(s)', 'actual_speed'))
  
       pid = crappy.blocks.PID(kp=0.038,
                               ki=2,
                               kd=0.05,
                               out_max=10,
                               out_min=-10,
                               i_limit=0.5,
                               target_label='command_speed',
                               labels=['t(s)', 'voltage'],
                               input_label='actual_speed')
  
       crappy.link(gen, pid)
       crappy.link(mot, pid)
  
       crappy.link(pid, mot)
  
       crappy.link(gen, graph)
       crappy.link(mot, graph)
  
       crappy.start()

Did you notice ? We have both ``crappy.link(gen, graph)`` and
``crappy.link(mot, graph)``, there's a loop in the data ! As this kind of
pattern is not uncommon in experimental setups, we wanted to make it clear that
it can be used in Crappy with no additional effort. You can now test it, and
notice that unlike the previous examples this one will terminate on its own
because the Generator path comes to an end at some point.

3.b. Modifiers
++++++++++++++

When you setup a test, it is common that the data outputted by a sensor can't be
used as such and needs a bit of processing, for example if it is very noisy. You
may also want to perform logical operations on data, like driving a device only
if a condition on an input is satisfied. To handle all these situations, Crappy
features :ref:`Modifiers` able to perform operations on data traveling through
the links.

To put it in a simple way, the modifiers can access all the labels sent through
a link and modify their values, delete them or even add new labels. They can
also choose not to transmit the labels to the target block, based on a condition
on them for example.

To illustrate that, let's consider the following example: using the same (fake)
DC motor as in the previous example, we want to measure the temporal derivative
of speed to make sure it never goes too high (what may for example damage a real
setup). We now also want to save the speed, but we don't need to save it at the
maximum frequency (which is probably higher than 100 Hz, depending on your
computer). The :ref:`Differentiate` and :ref:`Mean` modifiers will allow us to
write the corresponding script.

A modifier is always added on a given link. The syntax for adding one is as
follows :

.. code-block:: python

  crappy.link(<block1>, <block2>, modifier=crappy.modifier.<Name>(<args>))

The syntax for adding several is very similar, except the multiple modifiers
need to be put in a :obj:`list` :

.. code-block:: python

  crappy.link(<block1>, <block2>, modifier=[crappy.modifier.<Name1>(<args>),
                                            crappy.modifier.<Name2>(<args>)])

To know which arguments the modifiers take, the only way is to look in the
documentation. Here let's say we want to average the signals by a factor of 10
before saving, and the derivative of speed will have the label ``'accel'``.
After adding the recorders and the modifiers and modifying the grapher so that
it plots ``'accel'``, the code is now :

.. code-block:: python
   :emphasize-lines: 24,36-37,44-48

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator([
             {'type': 'constant', 'value': 1000, 'condition': 'delay=3'},
             {'type': 'ramp', 'speed': 100, 'condition': 'delay=5',
              'cmd': 0},
             {'type': 'constant', 'value': 1800, 'condition': 'delay=3'},
             {'type': 'constant', 'value': 500, 'condition': 'delay=3'},
             {'type': 'sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
              'condition': 'delay=15'}], spam=True,
                                     cmd_label='command_speed')
  
       mot = crappy.blocks.Machine([{'type': 'Fake_motor',
                                     'cmd': 'voltage',
                                     'mode': 'speed',
                                     'speed_label': 'actual_speed',
                                     'kv': 1000,
                                     'inertia': 4,
                                     'rv': .2,
                                     'fv': 1e-5}])
  
       graph = crappy.blocks.Grapher(('t(s)', 'accel'))
  
       pid = crappy.blocks.PID(kp=0.038,
                               ki=2,
                               kd=0.05,
                               out_max=10,
                               out_min=-10,
                               i_limit=0.5,
                               target_label='command_speed',
                               labels=['t(s)', 'voltage'],
                               input_label='actual_speed')
  
       rec = crappy.blocks.Recorder(filename='speeds.csv',
                                    labels=['t(s)', 'actual_speed'])
  
       crappy.link(gen, pid)
       crappy.link(mot, pid)
  
       crappy.link(pid, mot)
  
       crappy.link(mot, graph,
                   modifier=crappy.modifier.Diff(label='actual_speed',
                                                 out_label='accel'))

       crappy.link(mot, rec)
  
       crappy.start()

As illustrated here, modifiers are a powerful and simple way of tuning the way
your script manages data. As not every need can be covered by the provided
Crappy modifiers, it is truly worth having a look at :ref:`the section detailing
how to easily implement your own modifiers <1.e. modifiers>` !

3.c. Advanced generator paths
+++++++++++++++++++++++++++++

In a previous section we saw how to create everlasting generator paths and ones
ending after a given delay. In many tests, this is not sufficient. Let's imagine
that you have a tensile test setup on which you want to perform force-driven
cyclic stretching. Consider the example from :ref:`the second section
<2. Adding signal generators>`. We still want to perform 5 cycles of stretching
and relaxation, still at a 5/60 mm/s pace, but now the condition for switching
from stretching to relaxation is to reach 10kN. This needs to be somehow
indicated to the ``'condition'`` key.

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
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
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

3.d. Writing scripts efficiently
++++++++++++++++++++++++++++++++

This last section of the Getting started tutorial focuses on how to use Python's
great flexibility to write scripts more efficiently and elegantly. Because
they're within Crappy's particular framework, some of our users tend to forget
that they can actually use all the other Python packages or methods ! Here we're
going to show a few examples of code simplification.

3.d.1. Using variables
""""""""""""""""""""""

Until now in this tutorial all the numeric values needed as arguments in the
blocks have been written explicitly in the block definition. But there's
absolutely no obligation to do so ! Consider the following script :

.. code-block:: python

   import crappy
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 1,
                                            'condition': None}])
  
       machine = crappy.blocks.Fake_machine()
  
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
  
       machine = crappy.blocks.Fake_machine()
  
       record = crappy.blocks.Recorder(filename=path)
  
       crappy.link(gen, machine)
  
       crappy.link(machine, record)
  
       crappy.start()

Now a unique variable handles all the changes implied, more convenient
isn't it ?

3.d.2. Building lists and dicts smartly
"""""""""""""""""""""""""""""""""""""""

As previously showed in the tutorial, some Crappy objects have to take lists or
dicts as arguments. Until now, we always created these objects explicitly and
inside the blocks definition in order to keep the code simple and easily
understandable. If you followed :ref:`the previous section <3.d.1. Using
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

3.d.3. Using other packages
"""""""""""""""""""""""""""

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
folder located where the ``.py`` script file is. The code could look as
follows :

.. code-block:: python

   import crappy
   from pathlib import Path
  
   if __name__ == '__main__':
  
       gen = crappy.blocks.Generator(path=[{'type': 'constant',
                                            'value': 5/60,
                                            'condition': None}],
                                     cmd_label='input_speed')
  
       path = Path(__file__).parent / 'Tutorial' / 'data.csv'
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
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
  
       machine = crappy.blocks.Fake_machine(cmd_label='input_speed')
  
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
<Using custom blocks and adding them to Crappy>`, that pushes customization even
further by presenting how to create and use your own Crappy objects !

3.d.4. Using Crappy objects outside of a Crappy test
""""""""""""""""""""""""""""""""""""""""""""""""""""

To conclude this tutorial, we're going to see how Crappy objects can actually be
instantiated outside the context of a test and used as tools. Here we'll
consider that starting a Crappy test means executing the ``crappy.start()``
command. So how does this work ?

If you have a look at the :ref:`second tutorial <1. Using custom blocks in
scripts>`, you'll see that camera, inout and actuator objects are simply
classes performing elementary actions on a given device. So if you instantiate
these objects, you can just perform the same basic actions as Crappy would
(moving an actuator, grabbing a video frame, etc.) except here you need to call
the methods yourself instead of Crappy automatically calling them for you.

And why would you do that ? Because if Crappy's framework is truly nice for
running complex tests, it is a bit cumbersome when you only want to perform
simple tasks informally. As an example, we'll use a Crappy camera for taking
just one picture.

So let's get started ! Taking a single picture will of course be done using
a camera. You can have a look at :ref:`this section <1.b. cameras>` of the
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
