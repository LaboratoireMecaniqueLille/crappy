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

In this third section, you will **learn how to handle the most used Blocks of**
**Crappy**. These Blocks are all essential, and you'll come across at least one
of them in most scripts. For an extensive list of all the implemented Blocks,
refer to the :ref:`Current functionalities` section of the documentation.

2.a The Generator Block and its Paths
+++++++++++++++++++++++++++++++++++++

Let's start this tour of the most used Blocks with the :ref:`Generator`. It
allows to **generate a signal according to a pre-defined pattern**, and to send
it to downstream Blocks. It is mostly used for generating commands when driving
actuators or motors, but has actually many more possible applications (trigger
generation, target for a PID, etc.). In the previous section, the presented
example already contained an instance of the Generator Block. So let's take a
closer look at it :

.. literalinclude:: /downloads/crappy_syntax.py
   :language: python
   :emphasize-lines: 7-10
   :lines: 1-11

As you can see, the first argument of the Generator is its *path*. It describes
the shape of the generated signal, and is the main parameter to set when
instantiating a Generator Block. It has to be an
:obj:`~collections.abc.Iterable` (like a :obj:`list` or a :obj:`tuple`), that
contains one or several :obj:`dict` with the correct keys. **Each dictionary**
**represents one type of signal to generate**, and these signals are generated
in the same order as the dictionaries are given. The moment when the Generator
switches to the next dictionary is usually determined by the :py:`'condition'`
argument of the current dictionary, if applicable. After finishing the last
dictionary, the default behavior for the Generator is to stop the current
script.

To know the available types of signals and their mandatory and optional
arguments, you'll need to refer to the :ref:`Generator Paths` section of the
:ref:`API` page. There are quite many options available, and if you have a very
specific need you can always
:ref:`create your own Generator Path <1. Custom Generator Paths>`. The name of
the :class:`~crappy.blocks.generator_path.meta_path.Path` to use (the type of
signal to generate) is given by the :py:`'type'` key of each dictionary. The
other keys represent the possible arguments for the given Path, and thus
depend on the type of Path.

In the example above, the first and only chosen Path is the
:class:`~crappy.blocks.generator_path.Constant` one. As you can read in the
API, it requires the :py:`'condition'` and :py:`'value'` arguments, which are
indeed present in the dictionary. The Constant Path generates a constant signal
of value :py:`'value'`, and stops when :py:`'condition'` is met. The syntax for
the conditions is described in detail in the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method of
the base Path, and :ref:`a tutorial section <3. Advanced Generator Paths>` is
dedicated to the advanced uses of this argument. For a number of applications,
setting it to :py:`'delay=xx'` (next Path after *xx* seconds) or to :obj:`None`
(never switches to next Path) is fine.

Let's now try to modify the previous example, so that the :ref:`Fake machine`
is driven with a more complex pattern than just a constant speed. Let's say
that we now want to perform cyclic stretching and relaxation on the fake
sample, and then stretch it until failure. Compared to the previous example, we
can keep the :class:`~crappy.blocks.generator_path.Constant` Path, but we must
add a :class:`~crappy.blocks.generator_path.Cyclic` Path before to perform the
cyclic stretching. Here's how it looks :

.. literalinclude:: /downloads/tuto_generator.py
   :language: python
   :emphasize-lines: 7-16
   :lines: 1-17

As you may have guessed (or read in the API), the Cyclic Path alternates
between two Constant Paths, and must thus be given the arguments for these two
Paths. The :py:`'condition{1,2}'` keys indicate when to switch to the other
Constant, and the :py:`'cycles'` key indicates after how many cycles to end the
Cyclic Path and switch to the next one. Here, the Generator will switch to the
Constant Path once the Cyclic one ends, and then end the test once the
Constant Path finishes. To make the script runnable, let's complete it with the
same code as in the previous example :

.. literalinclude:: /downloads/tuto_generator.py
   :language: python
   :emphasize-lines: 18, 20-21, 23, 25, 27, 29-31, 33

The script should run in the exact same way as the one of the previous section,
except this time there should be two cycles of stretching and relaxation before
the final step of stretching until failure. **That reflects the changes we**
**made to the path of the Generator Block**. Just like previously, the script
will stop by itself. You can stop it earlier with CTRL+C, but this is not
considered as a clean way to stop Crappy. :download:`Download this Generator
example </downloads/tuto_generator.py>` to run it locally on your machine !

**You should now be able to build an run a variety of patterns for your**
**Generator Blocks** ! It is after all just a matter of reading the API,
selecting the Paths that you want to use, and include them in the *path*
argument of the Generator Block with the correct parameters. As mentioned
earlier in this section, more information about the Generator Paths can be
found in :ref:`another tutorial section <3. Advanced Generator Paths>`.

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
