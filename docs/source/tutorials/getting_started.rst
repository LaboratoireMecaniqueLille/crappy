===========================================
Getting started: writing scripts in Crappy
===========================================

.. role:: py(code)
  :language: python
  :class: highlight

These tutorials introduce the basic structure of a Crappy test script. They
require introductory Python knowledge.

0. General concepts
-------------------

Before writing the first script, read :doc:`../concepts/blocks_links_labels`.
It defines the Blocks that perform each task, the directed Links that connect
them, and the labels used to identify values. For image data, the
:doc:`../concepts/regular_links_and_image_links` page explains when to use an
ImageLink instead of a regular Link.
1. Understanding Crappy's syntax
--------------------------------

This section builds a complete, hardware-free Crappy script. If it does not
work as described, report the problem as explained on the
:ref:`Troubleshooting <troubleshooting:troubleshooting>` page). Note that this first example script requires the
:mod:`matplotlib` Python module to run.

The first thing to do when writing a script for Crappy is to open a new *.py*
file. Start by importing Crappy:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 3
   :lines: 1-3

Then, depending on the requirements of your experimental setup, add Blocks and
link them together. For this first example, suppose
that we want to acquire both the position and the force signal from a tensile
test machine, plot the data against time and save it. So that everyone can run
this first example without requiring any hardware, let's use the
:ref:`Fake machine <crappy_docs/blocks:fake machine>` Block instead of a real machine. To add this Block to the
script, follow this syntax:

.. code-block:: python

  <chosen_name> = crappy.blocks.<Block_name>(<arguments>)

:py:`<chosen_name>` is the name of the instance of the Block. There can be
several instances of a same Block running simultaneously, for example several
:ref:`Grapher <crappy_docs/blocks:grapher>` Blocks plotting different labels. :py:`<Block_name>` is the
exact name of the Block, as given in Crappy's :ref:`API <api:api>`. The possible
:py:`<arguments>` differ for every Block. The API reference lists the accepted
arguments.

.. Note::
   To access the online documentation from a computer with internet access,
   type in a Python terminal:

     >>> import crappy
     >>> crappy.docs()

In the case of the Fake Machine Block, its description is given in the API at
:class:`crappy.blocks.FakeMachine`. Its arguments are
optional, and it outputs data over specific labels. Let's still specify the
:py:`cmd_label` argument. The code now looks as follows:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 7
   :lines: 1-6, 12-13

.. Warning::
   If you're not familiar with the :py:`if __name__ == '__main__':` statement,
   you can find technical documentation `here
   <https://docs.python.org/3/library/__main__.html>`_. Crappy might not run if
   the script does not use this entry-point guard.

In addition to the Fake Machine, we also need a :ref:`Recorder <crappy_docs/blocks:recorder>` Block for
saving the data, and two :ref:`Grapher <crappy_docs/blocks:grapher>` Blocks for plotting it. There will also
be a :ref:`Generator <crappy_docs/blocks:generator>` Block for driving the Fake Machine. The usage of the most
used Blocks is detailed in :ref:`the next section <tutorials/getting_started:2. the most used blocks>`.
In our specific example, the script could be as follows:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 7-10, 14-15, 17, 19
   :lines: 1-20

After instantiating the Blocks, link them so that they can share data. To link
two Blocks, add the following line to the script:

.. code-block:: python

  crappy.link(<block1>, <block2>)

Where :py:`<block1>` and :py:`<block2>` are the names you assigned to the
instances of the Blocks. In the example, we need the Generator to drive the
Fake Machine, and the Fake Machine has to transfer the data it acquired to both
Graphers and to the Recorder Block. Here's what the script becomes after adding
the Links:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 21, 23-25
   :lines: 1-25

The Generator produces a constant signal and sends it to the Fake Machine
under the label :py:`'input_speed'`. This is the target speed at
which the Fake Machine should operate for the fake tensile test, and its value
is *5 mm/min*. Notice the :py:`'delay=40'` condition, that indicates the
Generator Block to stop the test after 40s. As stated in the documentation, the
Fake Machine Block outputs the following labels:
:py:`'t(s)', 'F(N)', 'x(mm)', 'Exx(%)', 'Eyy(%)'`. They are all transmitted to
the Grapher and Recorder Blocks, that respectively plot and record only part of
these labels. The Recorder will save the received data to a :py:`'data.csv'`
file, at the same level as the script.

Block arguments often specify the labels to use as inputs or outputs. These
explicit labels show the data flow throughout the code.

There's only one final line to add before you can run this first example:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 27

You can now execute the file like any regular Python file:

.. code-block:: shell-session

  python crappy_syntax.py

As the script starts, two windows should appear and plot the data coming from
the Fake Machine Block. In the meantime, a *data.csv* file should appear at
the same level as the script that was just started. It contains the data being
acquired by the Recorder Block. As mentioned earlier, the execution of the
script will stop after 40s as specified to the Generator Block. The script can
also stop earlier if an error occurs (e.g. missing dependency), or if the user
hits :kbd:`Control-c`. Note that this last way of ending a script should only
be used in case something goes wrong, e.g. if the script crashes. You can find
more about the different ways to stop a script in Crappy in :ref:`a later
section <tutorials/getting_started:3. properly stopping a script>`.

You can :download:`download this first example
</downloads/getting_started/crappy_syntax.py>` to run it locally on your
computer. Continue with the following sections to configure the Blocks used in
this script.

2. The most used Blocks
-----------------------

This section introduces commonly used Crappy Blocks. A typical script contains
at least one of them. For a list of all implemented Blocks,
refer to the :ref:`Current functionalities <features:current functionalities>` section of the documentation.

2.a. The Generator Block and its Paths
++++++++++++++++++++++++++++++++++++++

Let's start this tour of the most used Blocks with the :ref:`Generator <crappy_docs/blocks:generator>`. It
generates a signal according to a predefined pattern and sends
it to downstream Blocks. It is mostly used for generating commands when driving
actuators or motors, but has actually many more possible applications (trigger
generation, target value for a PID, etc.). In the previous section, the
presented example already contained an instance of the Generator Block. So
let's take a closer look at it:

.. literalinclude:: /downloads/getting_started/crappy_syntax.py
   :language: python
   :emphasize-lines: 7-10
   :lines: 1-11

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

The first argument of the Generator is its *path*. It describes
the shape of the generated signal, and is the main parameter to set when
instantiating a Generator Block. It has to be an
:obj:`~collections.abc.Iterable` (like a :obj:`list` or a :obj:`tuple`), that
contains one or several :obj:`dict` with the correct keys. Each dictionary
represents one type of signal to generate, and these signals are generated
in the same order as the dictionaries are given. The moment when the Generator
switches to the next dictionary is usually determined by the :py:`'condition'`
argument of the current dictionary, if applicable. After finishing the last
dictionary, the default behavior for the Generator is to stop the current
script.

To know the available types of signals and their mandatory and optional
arguments, you'll need to refer to the :ref:`Generator Paths <crappy_docs/blocks:generator paths>` section of the
:ref:`API <api:api>` page. There are quite many options available, and if you have a very
specific need you can always
:ref:`create your own Generator Path <tutorials/complex_custom_objects:1. custom generator paths>`. The name of
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
the base Path, and :ref:`a tutorial section <tutorials/more_complexity:3. advanced generator condition>`
is dedicated to the advanced uses of this argument. For a number of
applications, setting it to :py:`'delay=xx'` (next Path after *xx* seconds) or
to :obj:`None` (never switches to next Path) is fine.

Let's now try to modify the previous example, so that the :ref:`Fake machine <crappy_docs/blocks:fake machine>`
is driven with a more complex pattern than just a constant speed. Let's say
that we now want to perform cyclic stretching and relaxation on the fake
sample, and then stretch it until failure. Compared to the previous example, we
can keep the :class:`~crappy.blocks.generator_path.Constant` Path, but we must
add a :class:`~crappy.blocks.generator_path.Cyclic` Path before to perform the
cyclic stretching. Here's how it looks:

.. literalinclude:: /downloads/getting_started/tuto_generator.py
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
same code as in the previous example:

.. literalinclude:: /downloads/getting_started/tuto_generator.py
   :language: python
   :emphasize-lines: 18, 20-21, 23, 25, 27, 29-31, 33

The script should run in the exact same way as the one of the previous section,
except this time there should be two cycles of stretching and relaxation before
the final step of stretching until failure. This reflects the changes to the
Generator Block's path. Just like previously, the script
will stop by itself. You can stop it earlier with :kbd:`Control-c`, but this is
not considered as a clean way to stop Crappy. :download:`Download this
Generator example </downloads/getting_started/tuto_generator.py>` to run it
locally on your machine.

Build other patterns by selecting Paths from the API and including them in the
Generator Block's *path* argument with the required parameters. As mentioned
earlier in this section, more information about the Generator Paths can be
found in :ref:`another tutorial section <tutorials/more_complexity:3. advanced generator condition>`.
More examples of the Generator Block can be found in the `examples folder on
GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
examples/blocks>`__.

2.b. Camera acquisition and VisionBlocks
++++++++++++++++++++++++++++++++++++++++

For this second example, let's build a basic image-acquisition pipeline. For
new scripts, the recommended approach is to combine independent
:class:`~crappy.blocks.vision.VisionBlock` objects. A :ref:`Camera Source <crappy_docs/blocks:camera source>` only
acquires images from a real or virtual :ref:`Camera <crappy_docs/cameras:camera>`, while separate
VisionBlocks can process, display, or record the images. These stages are
connected explicitly with :ref:`Image Links <crappy_docs/links:image link>`.

The minimal pipeline for acquiring and displaying images therefore contains a
:ref:`Camera Source <crappy_docs/blocks:camera source>` and an :ref:`Image Displayer <crappy_docs/blocks:image displayer>`:

.. literalinclude:: /downloads/getting_started/tuto_camera.py
   :language: python
   :emphasize-lines: 7-12
   :lines: 1-12

.. Note::
   To run this example, you'll need to have *Pillow* and either the
   *opencv-python* or :mod:`matplotlib` Python modules installed.

The first argument of :class:`~crappy.blocks.vision.CameraSource` is the name
of the :class:`~crappy.camera.Camera` object to use for acquiring the images.
In this demo, the :ref:`Fake Camera <crappy_docs/cameras:fake camera>` is used so that the code can run without
any hardware. The acquisition-loop frequency belongs to CameraSource, whereas
the maximum display framerate and the loop that checks for new frames belong
to :class:`~crappy.blocks.vision.ImageDisplayer`.

Another important argument is the *config* one. When enabled, a
:class:`~crappy.tool.camera_config.CameraConfig` window is displayed before the
main part of the script runs. In this window, the user can interactively tune
the available settings for the selected Camera object. The possible
settings can be viewed by looking at the documentation in the API, for example
in the *open* method of :class:`~crappy.camera.FakeCamera` for the Fake Camera.
If the config window is disabled, the settings can still be adjusted by
providing them as *kwargs* to CameraSource. In that case, ``img_shape`` and
``img_dtype`` must also be given so that the shared image buffer can be created
before acquisition starts. Some processing VisionBlocks can instead ask their
upstream CameraSource to open a specialized configuration window. These
requests are handled automatically before the test starts when ``config`` and
``allow_downstream_config`` are enabled.

The two VisionBlocks must now be connected. A regular :func:`crappy.link` is
not intended for image transport, so this pipeline uses
:func:`crappy.img_link`:

.. literalinclude:: /downloads/getting_started/tuto_camera.py
   :language: python
   :emphasize-lines: 7, 11, 16
   :lines: 1-16

The CameraSource owns one shared image buffer, and the ImageDisplayer copies
the newest image when it is ready. Consequently, setting a lower display
framerate does not slow down acquisition. Intermediate images may be skipped
by the displayer.

To have a functional and clean example script, we still need to add a few
lines. In particular, unlike the :ref:`Generator <crappy_docs/blocks:generator>` Block, the image source does
not automatically stop after a condition is met. To allow the script to stop
in a proper way, a :ref:`Stop Button <crappy_docs/blocks:stop button>` Block should be added. It will display a
button, that will stop the execution of the script when clicked upon. It is
always possible to stop Crappy using :kbd:`Control-c`, but this is not
considered a proper way of ending the script. After inserting the stop button,
here's the final runnable script:

.. literalinclude:: /downloads/getting_started/tuto_camera.py
   :language: python
   :emphasize-lines: 14, 16, 18

:download:`Download this CameraSource and ImageDisplayer example
</downloads/getting_started/tuto_camera.py>` to run it locally on your
machine. A more extensively commented version is available in the `vision
examples folder <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
master/examples/vision_blocks/camera_basic_display.py>`__.

The explicit architecture becomes especially useful as the workflow grows. An
:ref:`Image Recorder <crappy_docs/blocks:image recorder>` can be connected to the same CameraSource with a second
call to :func:`crappy.img_link`, without changing or slowing the displayer.
Similarly, the :ref:`DIC VE Processor <crappy_docs/blocks:dic ve processor>`, :ref:`DIS Correl Processor <crappy_docs/blocks:dis correl processor>`, and
:ref:`Video Extenso Processor <crappy_docs/blocks:video extenso processor>` can analyze the same source while sending their
small results and display overlays through regular Links. See the `basic
recording example <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
master/examples/vision_blocks/camera_basic_record.py>`__, the `software
trigger example <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
master/examples/vision_blocks/camera_software_trigger.py>`__, and the other
`VisionBlock examples <https://github.com/LaboratoireMecaniqueLille/crappy/
tree/master/examples/vision_blocks>`__ for complete pipelines.

The all-in-one :class:`~crappy.blocks.Camera` Block combines acquisition with
optional display, recording, and processing children inside a single Block, and
can still be convenient when this fixed architecture is exactly what is needed.
VisionBlocks are recommended for new image pipelines. The all-in-one Camera
Blocks remain supported and are not planned for deprecation. Their examples
remain available in the
`Camera examples folder <https://github.com/LaboratoireMecaniqueLille/crappy/
tree/master/examples/blocks/camera>`__.

2.c. The Grapher Block
++++++++++++++++++++++

For displaying the data acquired or generated by a Block, the :ref:`Grapher <crappy_docs/blocks:grapher>`
Block plots received data, one label against another. In the first example, you can
see that the syntax for providing the labels is :py:`('label_x', 'label_y')`.
What is not shown in the first example, though, is that you can plot multiple
curves on one graph. You also don't have to plot data against time. You can
plot any label against any other one as long as they are synchronized.

Just like any other Block, the Grapher also has a number of parameters that can
be adjusted. You can find the exact list in the API, at the
:class:`~crappy.blocks.Grapher` entry. Here is a modified version of the first
example, where the force is plotted against the position and where some extra
arguments of the Grapher Block are set:

.. literalinclude:: /downloads/getting_started/tuto_grapher.py
   :language: python
   :emphasize-lines: 17-19, 24

Other Blocks can display data. See the
:ref:`Dashboard <crappy_docs/blocks:dashboard>` and the :ref:`Link Reader <crappy_docs/blocks:link reader>` Blocks for example. The Grapher
Block takes up quite much CPU and memory, so it is better not to have too many
of its instances in a script. You can :download:`download this Grapher example
</downloads/getting_started/tuto_grapher.py>` to run it locally on your
machine. Another example of the Grapher Block can be found in the `examples
folder on GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/
master/examples/blocks>`__.

2.d. The Recorder Block
+++++++++++++++++++++++

The :ref:`Recorder <crappy_docs/blocks:recorder>` Block saves data acquired or
generated by a Block. It must be linked to one and only
one upstream Block and saves all received data in a ``.csv`` or equivalent
text file. The first example above demonstrates its syntax.
Another example of the Recorder Block can be found in the `examples folder on
GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
examples/blocks>`__. Note that for recording streams, the :ref:`HDF Recorder <crappy_docs/blocks:hdf recorder>`
Block should be used instead (see :ref:`this later section
<tutorials/more_complexity:4. dealing with streams>`).

2.e. The IOBlock Block
++++++++++++++++++++++

Along with the Camera and Actuator Blocks, the :ref:`IOBlock <crappy_docs/blocks:ioblock>` is one of the
few Blocks in Crappy that can interact with hardware. It can acquire data from
a device and send it to downstream Blocks. It can also receive commands from
upstream Blocks and set them on active hardware. Hardware that supports both
operations can use them simultaneously. To communicate with hardware, the IOBlock relies on the
:ref:`In / Out <crappy_docs/inouts:in / out>` objects, that each implement the communication with a different
device. Here's an example of code featuring an IOBlock for data acquisition:

.. literalinclude:: /downloads/getting_started/tuto_ioblock.py
   :language: python
   :emphasize-lines: 7-9
   :lines: 1-6, 15, 17-23, 25-27

.. Note::
   To run this example, you'll need to have the :mod:`psutil` and
   :mod:`matplotlib` Python modules installed.

To acquire data with an IOBlock, first specify the
:class:`~crappy.inout.InOut` that you
want to use for data acquisition. The :class:`~crappy.inout.FakeInOut` was
chosen here as it does not require any hardware to run. Then, you need to
indicate which labels will carry the acquired values. Refer to the API to know
what kind of data the chosen InOut outputs. The output data is here visualized
using a Grapher Block. The data that you can
visualize on the graph corresponds to the current RAM usage of your computer.
You can open or close a web browser to see it change consistently. Let's now
write another example where a command is set by an IOBlock:

.. literalinclude:: /downloads/getting_started/tuto_ioblock.py
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
it is possible to use both behaviors of the IOBlock simultaneously:

.. literalinclude:: /downloads/getting_started/tuto_ioblock.py
   :language: python
   :emphasize-lines: 15-18

The two IOBlock operations can run in the same script. You can
:download:`download this IOBlock example
</downloads/getting_started/tuto_ioblock.py>` to run it locally on your
machine. More examples of the IOBlock can be found in the `examples folder on
GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
examples/blocks>`_. Note that the *streamer* mode of the IOBlock is presented
in :ref:`a dedicated section <tutorials/more_complexity:4. dealing with streams>`, and same goes for the
:ref:`make_zero functionality <tutorials/complex_custom_objects:2. more about custom inouts>`. Directly check
the documentation of the IOBlock to learn more about it.

2.f. The Machine Block
++++++++++++++++++++++

Similar to the :ref:`IOBlock <crappy_docs/blocks:ioblock>`, the :ref:`Machine <crappy_docs/blocks:machine>` Block can send commands to
hardware and acquire data from it. The difference is that the IOBlock can send
any type of command and acquire any type of data, whereas the Machine Block can
only send speed or position commands and acquire speed and position data.
The Machine Block is designed to drive motors or comparable actuators. It
relies on the :ref:`Actuators <crappy_docs/actuators:actuators>` objects
for communicating with the hardware. The syntax of the arguments to provide to
the Machine Block is quite similar to that of the :ref:`Generator <crappy_docs/blocks:generator>` Block, as
demonstrated here:

.. literalinclude:: /downloads/getting_started/tuto_machine.py
   :language: python
   :emphasize-lines: 16-21

The Machine Block accepts an iterable of :obj:`dict` as its first argument.
Each dictionary describes one Actuator, so one Machine Block can drive several
Actuators. In each dictionary, the :py:`'type'`
key indicates the name of the Actuator to use. Then, other keys like
:py:`'mode'` or :py:`'cmd_label'` provide information on how to drive the
Actuator. The :py:`'speed_label'` key indicates which information to acquire
from the Actuator and under which label to return it. To have an overview of
the keys that are not Actuator-dependent and their effect, refer to the
documentation of the :class:`~crappy.blocks.Machine` Block in the API. Finally,
the arguments to pass to the Actuator should also be given in the dictionary,
here through the :py:`'kv'` key for example. The Actuator used here is the
:class:`~crappy.actuator.FakeDCMotor`, that does not require any hardware to
run. Check its documentation to get all the possible arguments it accepts.

.. Note::
   Driving several Actuators with one Machine Block is only recommended when
   these Actuators need to be synchronized, e.g. on a bi-motor machine.
   Otherwise, you should rather drive each Actuator with a different Machine
   Block.

You can :download:`download this Machine Block example
</downloads/getting_started/tuto_machine.py>` to run it locally on your
machine. It should last only 20s before stopping by itself. The Grapher window
that appears displays the current speed of the Actuator driven by the Machine
Block, and responding to the voltage command (treated as a speed by the
Machine) received from the Generator. More examples of the Machine Block can be
found in the `examples folder on GitHub <https://github.com/
LaboratoireMecaniqueLille/crappy/tree/master/examples/blocks>`__.

3. Properly stopping a script
-----------------------------

The previous sections presented several ways to stop a Crappy script. A clean
shutdown lets hardware integrations deinitialize their devices. For example, an
Actuator can stop a motor regardless of why the test ended. It also lets the
framework release the resources used by the test.

Do not abruptly close the terminal or force the application to close unless it
is unresponsive. These methods can prevent hardware and software cleanup.

Starting from version 2.0.0, hitting :kbd:`Control-c` to stop Crappy (i.e.
raising :exc:`KeyboardInterrupt`) is also considered as an invalid behavior.
Unlike more aggressive methods, :kbd:`Control-c` is still handled internally
and should lead to a proper termination of the Blocks. As it
might lead to unexpected behavior, and to deter users from using it, we chose
to have :kbd:`Control-c` raise an Exception once all the Blocks are correctly
stopped. This behavior can be tuned, see the :ref:`7. Advanced control over the
runtime <tutorials/more_complexity:7. advanced control over the runtime>` section of the tutorials. The default behavior therefore raises an error after
handling :kbd:`Control-c`, even when every Block stops correctly.

Use a lifecycle-aware stop mechanism during normal operation. Some Crappy
objects stop a script when their work is complete. The most common one is the
:ref:`Generator <crappy_docs/blocks:generator>`, which can stop a
script once its :ref:`Generator Paths <crappy_docs/blocks:generator paths>` are exhausted. Same goes for the
:ref:`File Reader <crappy_docs/cameras:file reader>` Camera object, that can stop a script once its images are
exhausted. In addition to these two Blocks, two other ones are specifically
designed to stop a test in a clean way. They are the :ref:`Stop Block <crappy_docs/blocks:stop block>` and
:ref:`Stop Button <crappy_docs/blocks:stop button>` Blocks, designed to respectively stop a test automatically
when a condition is met, and stop a test manually when the user clicks on a
button. For users integrating Crappy in a GUI, the :ref:`crappy.stop() <crappy_docs/aliases:crappy.stop()>` method
is also an option.

.. Note::
   A test in Crappy will also end if an unexpected Exception is raised anywhere
   in the module. In that case, all Blocks begin stopping and, just like
   with :kbd:`Control-c`, an error will be raised once all the Blocks are
   stopped. Crappy handles the unexpected Exception through its normal shutdown
   sequence.

When writing a script, first determine which termination way seems more
appropriate. Use the Stop Button Block if you need to stop the test at any
time, for example when sample properties make the duration unpredictable. Use
the Stop Block, or the Generator Block when applicable, if an objective
condition signals the end of the test. You can use both a Stop Button and a
Stop Block together. When
you want to trigger Crappy's termination from a GUI, use :ref:`crappy.stop() <crappy_docs/aliases:crappy.stop()>`
instead.
