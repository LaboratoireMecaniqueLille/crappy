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

.. Note::
   To run this example, you'll need to have the *matplotlib* Python module
   installed.

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
found in :ref:`another tutorial section <3. Advanced Generator Paths>`. More
examples of the Generator Block can be found in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_.

2.b The Camera Block
++++++++++++++++++++

For this second highlighted Block, let's introduce the :ref:`Camera` Block. It
allows to **acquire images from real or virtual** :ref:`Cameras` objects, and
to **record, display, and process** them. No processing is included in the base
Camera Block, but all of its children are meant to perform some sort of
processing (see the :ref:`Video extenso`, :ref:`DIS Correl` or :ref:`DIC VE`
Blocks). The instantiation of a Camera Block is quite simple, here's how it
looks like :

.. literalinclude:: /downloads/tuto_camera.py
   :language: python
   :emphasize-lines: 7-12
   :lines: 1-13

.. Note::
   To run this example, you'll need to have the *opencv-python*, *matplotlib*
   and *Pillow* Python modules installed.

The first given argument is the name of the :class:`~crappy.camera.Camera` to
use for acquiring the images. In this demo, the :ref:`Fake Camera` is used so
that the code can run without any hardware. Then, the user specifies whether
they want the captured images to be displayed and/or recorded. There are more
options available for tuning the acquisition, the recording and the display,
they are all listed in the documentation of the :class:`~crappy.blocks.Camera`
Block in the API.

Another important argument is the *config* one. When enabled, a
:class:`~crappy.tool.camera_config.CameraConfig` window is displayed before the
main part of the script runs. In this window, the user can **interactively**
**tune the available settings** for the selected Camera object. The possible
settings can be viewed by looking at the documentation in the API, for example
in the *open* method of :class:`~crappy.camera.FakeCamera` for the Fake Camera.
If the config windows is disabled, the settings can still be adjusted by
providing them as *kwargs* to the Camera Block. Note that disabling the config
window makes some arguments mandatory, as detailed in the documentation of the
Camera Block.

To have a functional and clean example script, we still need to add a few
lines. In particular, unlike the :ref:`Generator` Block, the Camera does not
automatically stop after a condition is met. To allow the script to stop in a
proper way, a :ref:`Stop Button` Block should be added. It will display a
button, that will stop the execution of the script when clicked upon. It is
always possible to stop Crappy using CTRL+C, but this is not considered a
proper way of ending the script. After inserting the stop button, here's the
final runnable script :

.. literalinclude:: /downloads/tuto_camera.py
   :language: python
   :emphasize-lines: 14, 16

:download:`Download this Camera example </downloads/tuto_camera.py>` to run it
locally on your machine ! You can adjust the parameters to see what the effect
is. You'll find more information on the possible arguments and their effects in
the documentation. **The children of the Camera Block that perform image**
**processing work on the exact sample principle**, except they accept extra
arguments and can output data to downstream Blocks. More examples of the Camera
Block and its children can be found in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_.

2.c The Grapher Block
+++++++++++++++++++++

For displaying the data acquired or generated by a Block, the :ref:`Grapher`
Block is by far the most popular solution. It allows to **plot the received**
**data**, always one label against another one. In the first example, you can
see that the syntax for providing the labels is :py:`('label_x', 'label_y')`.
What is not shown in the first example, though, is that you can plot multiple
curves on a same graph. You also don't have to plot data against time, you can
plot any label against any other one as long as they are synchronized.

Just like any other Block, the Grapher also has a number of parameters that can
be adjusted. You can find the exact list in the API, at the
:class:`~crappy.blocks.Grapher` entry. Here is a modified version of the first
example, where the force is plotted against the position and where some extra
arguments of the Grapher Block are set :

.. literalinclude:: /downloads/tuto_grapher.py
   :language: python
   :emphasize-lines: 17-19, 24

Note that **there are other ways of displaying data in Crappy**, check the
:ref:`Dashboard` and the :ref:`Link Reader` Blocks for example. The Grapher
Block takes up quite much CPU and memory, so it is better not to have too many
of its instances in a script. You can :download:`download this Grapher example
</downloads/tuto_grapher.py>` to run it locally on your machine. Another
example of the Grapher Block can be found in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_.

2.d The Recorder Block
++++++++++++++++++++++

For saving the data acquired or generated by a Block, the preferred solution in
Crappy is to use the :ref:`Recorder` Block. It must be linked to one and only
one upstream Block, and will **save all the data it receives from it in a**
*.csv* **(or equivalent text format) file**. This Block is quite basic, so the
first example given above should be enough for you to understand its syntax.
Another example of the Recorder Block can be found in the `examples folder on
GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_. Note
that for recording streams, the :ref:`HDF Recorder` Block should be used
instead (see :ref:`this later section <4. Dealing with streams>`).

2.e The IOBlock Block
+++++++++++++++++++++

Along with the Camera and Actuator Blocks, the :ref:`IOBlock` is one of the
few Blocks in Crappy that can interact with hardware. It serves two purposes :
first, **it can acquire data from a device and send it to downstream Blocks**.
And second, **it can also receive commands from upstream Blocks and set them**
**on active hardware**. These two functions can be used simultaneously, for
hardware supporting it. To communicate with hardware, the IOBlock relies on the
:ref:`In / Out` objects, that each implement the communication with a different
device. Here's an example of code featuring an IOBlock for data acquisition :

.. literalinclude:: /downloads/tuto_ioblock.py
   :language: python
   :emphasize-lines: 7-9
   :lines: 1-6, 15, 17-23, 25-27

.. Note::
   To run this example, you'll need to have the *psutil* and *matplotlib*
   Python modules installed.

As you can see, the base syntax is quite simple for acquiring data with an
IOBlock. You first have to specify the :class:`~crappy.inout.InOut` that you
want to use for data acquisition. The :class:`~crappy.inout.FakeInOut` was
chosen here as it does not require any hardware to run. Then , you need to
indicate which labels will carry the acquired values. Refer to the API to know
what kind of data the chosen InOut outputs. The output data is here visualized
using a Grapher Block, and that's pretty much it ! The data that you can
visualize on the graph corresponds to the current RAM usage of your computer.
You can open or close a web browser to see it change consistently. Let's now
write another example where a command is set by an IOBlock :

.. literalinclude:: /downloads/tuto_ioblock.py
   :language: python
   :emphasize-lines: 15-17
   :lines: 1-16, 18, 23-24, 26-27

This time, the :py:`'cmd_labels'` argument must be set on the IOBlock to
indicate which label carries the command to set. The label carrying the command
can be generated by any type of Block, but for simplicity it is here output by
a Generator Block. When receiving a command, the
:class:`~crappy.inout.FakeInOut` tries to use the correct amount of RAM to
match the target value. Here, the command is a sine wave oscillating between 30
and 70% of RAM usage. You can visualize the effect of the script by opening a
RAM monitor, such as the Task Manager in Windows or *htop* in Linux. Finally,
it is possible to use both behaviors of the IOBlock simultaneously :

.. literalinclude:: /downloads/tuto_ioblock.py
   :language: python
   :emphasize-lines: 15-18

Notice how the two functionalities of the IOBlock integrate seamlessly into a
single common script. You can :download:`download this IOBlock example
</downloads/tuto_ioblock.py>` to run it locally on your machine. More examples
of the IOBlock can be found in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_. Note
that the *streamer* mode of the IOBlock is presented in :ref:`a dedicated
section <4. Dealing with streams>`, while the *make_zero* functionality is just
not presented in the tutorials. Directly check the documentation of the Block
to learn more about it.

2.f The Machine Block
+++++++++++++++++++++

3. Properly stopping a script
-----------------------------
