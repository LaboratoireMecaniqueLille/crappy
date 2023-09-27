===========================================
Getting started : writing scripts in Crappy
===========================================

.. role:: py(code)
  :language: python
  :class: highlight

In the following tutorials, you're going to **learn the very basics of**
**writing scripts** for running test protocols with Crappy. Only a beginner's
level in Python is required, don't worry !

0. General concepts
-------------------

This first section of the tutorials introduces the very basic concepts of
Crappy. No code is involved for now, it only describes the general way data can
flow in Crappy.

0.a. Blocks
+++++++++++

In Crappy, even the most complex setups can be described with only two elements
: the **Blocks** and the **Links**. The Blocks are responsible for managing
data. There are many different types of Blocks, that all have a unique
function. Some will acquire data, others transform it, or use it to drive
hardware, etc. As the Blocks perform very specific tasks, a test script usually
contains several Blocks (there is no upper limit). **Blocks either take data**
**as an input, or they output data, or both**.

0.b. Links
++++++++++

Blocks are always blissfully ignorants of each other, so the Links are there to
allow data transfers between them. **A Link is established between two Blocks**
and is oriented. Establishing a link between Block 1 and Block 2 means that
Block 2 will receive of all of Block 1's outputs. Because the Link is oriented,
Block 1 will however not be aware of Block 2's outputs. There is no upper limit
in the number of Links pointing towards and originating from a given Block.

0.c. Labels
+++++++++++

**Data flowing between the Blocks through the Links is always labeled**. Labels
are simply names associated with a given stream of data. Let's say that Block 1
outputs three data streams labeled :py:`'time', 'Force', 'Position'`, and is
linked with Block 2 that only takes two inputs. As we said, Block 2 is aware of
all of Block 1's outputs and thus needs a way to differentiate them. Thanks to
labels, the user can simply specify in the arguments of Block 2 which labels to
consider (for example only :py`'time', 'Position'`). The data stream labeled
:py:`'Force'` will be lost to Block 2, but maybe Block 1 is also linked with a
Block 3 that's using it !

1. Understanding Crappy's syntax
--------------------------------

In this second part of the tutorials, we're going to **write step-by-step an**
**actual script for Crappy** that you can run locally on your machine ! All the
following tutorials will also follow the same principle. If a script would not
work as expected, please signal it to the developers (see the
:ref:`Troubleshooting` page). Note that this first example script requires the
`matplotlib <https://matplotlib.org/>`_ Python module to run.

The first thing to do when writing a script for Crappy is to open a new *.py*
file. In this new file, you should start by **importing the module Crappy** :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 3
   :lines: 1-3

Then, depending on the requirements of your experimental setup, you can **add**
**various Blocks and link them together**. For this first example, let's say
that we want to acquire both the position and the force signal from a tensile
test machine, plot the data against time and save it. So that everyone can run
this first example without requiring any hardware, let's use the
:ref:`Fake machine` Block instead of a real machine. To add this Block to the
script, follow this syntax :

.. code-block:: python

  <chosen_name> = crappy.blocks.<Block_name>(<arguments>)

:py:`<chosen_name>` is the name of the instance of the Block. There can be
several instances of a same Block running simultaneously, for example several
:ref:`Grapher` Blocks plotting different labels. :py:`<Block_name>` is the
exact name of the Block, as given in Crappy's :ref:`API`. The possible
:py:`<arguments>` differ for every Block, the only way to know for sure is to
**refer to the documentation** !

.. Note::
   To easily access the online documentation on a computer that has an internet
   access, simply type in a Python terminal :

     >>> import crappy
     >>> crappy.docs()

In the case of the Fake Machine Block, its description is given in the API at
:class:`crappy.blocks.FakeMachine`. As you can see, all its arguments are
optional, and it outputs data over specific labels. Let's still specify the
:py:`cmd_label` argument. The code now looks as follows :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 7
   :lines: 1-6, 12-13

.. Warning::
   If you're not familiar with the :py:`if __name__ == '__main__':` statement,
   you can find technical documentation `here
   <https://docs.python.org/3/library/__main__.html>`_. Crappy might not run if
   you don't wrap your code in this statement !

In addition to the Fake Machine, we also need a :ref:`Recorder` Block for
saving the data, and two :ref:`Grapher` Blocks for plotting it. There will also
be a :ref:`Generator` Block for driving the Fake Machine. The usage of the most
used Blocks is detailed in :ref:`the next section <2. The most used Blocks>`.
In our specific example, the script could be as follows :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 7-10, 14-15, 17, 19
   :lines: 1-20

Now that all the Blocks are instantiated, **they need to be linked together**
so that they can share data between each other. To link two Blocks together,
simply add the following line to the script :

.. code-block:: python

  crappy.link(<block1>, <block2>)

Where :py:`<block1>` and :py:`<block2>` are the names you assigned to the
instances of the Blocks. In the example, we need the Generator to drive the
Fake Machine, and the Fake Machine has to transfer the data it acquired to both
Graphers and to the Recorder Block. Here's what the script becomes after adding
the Links :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 21, 23-25
   :lines: 1-25

Let's have a more detailed look at what the script is doing ! First, the
Generator is generating a constant signal, that it sends to the Fake Machine
over the label :py:`'input_speed'`. Obviously, this is the target speed at
which the Fake Machine should operate for the fake tensile test, and its value
is *5 mm/min*. Notice the :py:`'delay=40'` condition, that indicates the
Generator Block to stop the test after 40s. As stated in the documentation, the
Fake Machine Block outputs the following labels :
:py:`'t(s)', 'F(N)', 'x(mm)', 'Exx(%)', 'Eyy(%)'`. They are all transmitted to
the Grapher and Recorder Blocks, that respectively plot and record only part of
these labels. The Recorder will save the received data to a :py:`'data.csv'`
file, at the same level as the script.

Notice how in the Blocks you can often specify the names of the labels to use
as inputs and/or the ones to output. This way, it is straightforward to keep
track of the data flow throughout the code.

There's only one final line to add before you can run this first example :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 27

You can now execute the file like any regular Python file :

.. code-block:: shell-session

  python crappy_syntax.py

As the script starts, two windows should appear and plot the data coming from
the Fake Machine Block. In the mean time, a *data.csv* file should appear at
the same level as the script that was just started. It contains the data being
acquired by the Recorder Block. As mentioned earlier, the execution of the
script will stop after 40s as specified to the Generator Block. The script can
also stop earlier if an error occurs (e.g. missing dependency), or if the user
hits CTRL+C. Note that this last way of ending a script should only be used in
case something goes wrong, e.g. if the script crashes. You can find more about
the different ways to stop a script in Crappy in :ref:`a later section <3.
Properly stopping a script>`.

**You now know learned the very basics of writing scripts for Crappy** ! You
can :download:`download this first example </downloads/crappy_syntax.py>` to
run it locally on your computer. This second section of the tutorials was only
a brief introduction, there's still much more to learn in the following
sections !

2. The most used Blocks
-----------------------

2.a The Generator Block
+++++++++++++++++++++++

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
the patterns can be found in :ref:`the generator path section <Generator Paths>`.
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

       machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

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

2.b The Camera Block
++++++++++++++++++++

2.c The Grapher Block
+++++++++++++++++++++

2.d The Recorder Block
++++++++++++++++++++++

2.e The IOBlock Block
+++++++++++++++++++++

2.f The Machine Block
+++++++++++++++++++++

3. Properly stopping a script
-----------------------------
