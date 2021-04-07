=======================================
How to generate a signal with CRAPPY ?
=======================================

As in every new :ref:`CRAPPY<What is Crappy ?>` project, the first thing to do is to import it. So your first line of code will simply be::

   import crappy

Then, you can think about what you want your program to do.
Here we'll simply generate a signal, that we'll be able to send as a command or to plot afterwards.

So first let's choose what kind of signal we want.

Choose a signal
-------------------

There are 8 different types of signal, all described in the :ref:`generator path` folder.

.. note:: Every signal must be a dictionary providing the parameters to generate it.

Dictionary:
   A dictionary consists in a collection of key-value pairs. Each key-value pair maps the
   key to its associated value. Here is the syntax::

      d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}

   The order does not matter as each value is only associated with its key.

.. note::

   Each dictionary in the signal generator **MUST** have a ``type`` key.

   Every non cyclic path **MUST** have a ``condition`` key. (Cyclic paths **MUST** have
   ``condition1`` and ``condition2``.)

A condition can be:
   - A delay (str, in seconds): ``'delay=x'``
   - A condition on a label (str): ``'label<x'`` or ``'label>x'``. The label can be internal
     (t(s) or cmd_label) or be provided using a link.
   - ``None`` for a signal that never ends
   - A personalized condition: a fonction taking the dict of current labels as an argument 
     and returning ``True`` if the path ends.

Here are described the 4 most common types of signal:

1. Constant signals
++++++++++++++++++++

The simplest signal, it only has 3 keys:
   - ``type``: :ref:`constant`
   - ``value``: Whatever constant int or float value you want to give.
   - ``condition``: The condition that will indicate the end of the signal. For example a 
     position to reach or a time to wait are commonly used conditions.

Example:
   To get a constant signal sending the value 5 for 10 seconds, one should write::

      Signal1 = {'type': 'constant', 'value': 5, 'condition': 'delay=10'}

2. Ramp signals
++++++++++++++++

An other quite simple signal that just has 4 keys:
   - ``type``: :ref:`ramp`
   - ``condition``: same as the :ref:`constant signal<1. Constant signals>` condition, it will indicate the end of your signal when it's reached.
   - ``speed``: the slope of the ramp, in units per second.
   - ``cmd``: the starting value of the ramp. `Optional key`. (If not specified, the
     starting value will be the previous value.)

Example:
   To get a ramp signal increasing by 2 (mm/s for example) until it reaches 30
   (mm), one should write::

      Signal2 = {'type': 'ramp', 'speed': 2, 'condition': 'x(mm)>30')

.. note::

      Of course ``x(mm)`` must be a label containing the real-time position of
      whatever we are controlling. It should be provided to the Generator block
      through a link.

3. Sine signals
++++++++++++++++

Now a sine signal, that has 6 keys:
   - ``type``: :ref:`sine`
   - ``freq``: the frequency of the signal
   - ``amplitude``: the amplitude of the signal
   - ``offset``: adds an offset to the signal, the default offset is 0. `Optional key`.
   - ``phase``: adds a pahse to the signal, in unit of radians. The default phase is 0. `Optional key`.
   - ``condition``: same as the :ref:`constant signal<1. Constant signals>` condition, it will indicate the end of your signal when it's reached.

Example:
   To get a sine with a frequency of 0.5, an amplitude of 2, an offset of 1 and that
   stops after 25 seconds, one should write::

      Signal3 = {'type': 'sine', 'freq': .5, 'amplitude': 2, 'offset': 1,
      'condition': 'delay=25'}

   Now to get a cosine, with the same parameters as the ``Signal3``, then one should
   write::

      from math import pi
  
      Signal4 = {'type': 'sine', 'freq': .5, 'phase': pi/2, 'amplitude': 2, 'offset': 1,
      'condition': 'delay=25'}

.. note:: Ne number pi first has to be imported from the python module ``math``.

4. Cyclic ramp signals
+++++++++++++++++++++++

This type of signal is simply the combination of two simple :ref:`ramps<ramp>`, with the possibility to repeat them. So we've already detailed :ref:`how it works<2. Ramp signals>`!

It has 6 keys:
   - ``type``: :ref:`cyclic ramp`
   - ``condition1``: the condition to reach to stop the first ramp.
   - ``speed1``: the slope of the first ramp 
   - ``condition2``: the condition to reach to stop the second ramp.
   - ``speed2``: the slope of the second ramp
   - ``cycles``: number of repetitions of the two ramps. Can be 1. If 0, it will loop forever.

Example:
   To get a signal that goes up at a speed of 0.1 (mm/s) until it reach 5
   (mm), then goes down to 2 (mm) at a speed of 0.1 (mm/s), and is repeated 3 times, one should write::

      Signal5 = {'type': 'cyclic_ramp', 'condition1': 'x(mm)>5',
      'speed1': 0.1, 'condition2': 'x(mm)<2', 'speed2': -0.1, 'cycles': 3}

Apart from these 4 main types of signals, there's another one that can prove very useful.

5. Custom signals
++++++++++++++++++

This type allows to import any signal from a .csv file (hence the name `custom`).

It only has 2 key:
   - ``type``: :ref:`custom`
   - ``filename``: the path of the .csv file.

.. warning::

   The file must contain 2 columns: The first one with the time, and the second one with
   the value to send.

.. note::

   It will try to send at the right time every timestamp with the associated value.

Example:
   Do you really need it? ::

      Signal6 = {'type': 'custom', 'filename': 'my_custom_signal.csv'}

One the signal has been created, it's ready to be generated using a :ref:`Generator` crappy block.

Generate a signal
---------------------

Creating a :ref:`Generator` is as simple as that::

   OurGenerator = crappy.blocks.Generator([Signalx])

.. note::

      The :ref:`Generator` class is a block, so it's located in the
      folder :ref:`blocks<Blocks>` which is in :ref:`crappy<What is Crappy ?>`:
      ``crappy.blocks.[...]``

      Signalx can be replaced with the name of a signal you've already created, or
      directly with the explicit dictionary of the signal you want.

And here it is! Actually, that's not all. A :ref:`Generator` block in crappy must contain a list of dictionaries (hence the list: ``[]``).

Great, other signals can be added! ::

   OurGenerator = crappy.blocks.Generator([Signal1, Signal2, Signal3, Signal4, Signal5])

.. note::

   Once the end of a signal has been reached, the next one in the list begins immediately.
   Once the end of the list have been reached, the :ref:`Generator` stops the
   program.

Several options also allow to precise how the :ref:`Generator` should work:
   - ``cmd_label`` renames the output signal. The default name is 'cmd'. This feature is mostly useful when the program contains several Generators.
   - ``freq`` imposes the generator output frequency. The Generator will output commands at the given frequency even if that implies missing signal points.
   - ``repeat`` if True, the generator loops endlessly on the list and never ends the program.

Example:
   To generate Signal1 at 500 points per second and name it 's1', and also
   generate Signal2 and Signal3 without imposing a frequency and name it 's2',
   one should write::

      OurGenerator1 = crappy.blocks.Generator([{'type': 'constant',
      'value': 5, 'condition': 'delay=10'}], cmd_label='s1', freq=500)

      OurGenerator2 = crappy.blocks.Generator([Signal2,Signal3], cmd_label='s2')

As simple as that ! Now let's try plotting the signals.

Plot a signal
-----------------

To do so, first create a :ref:`Grapher` crappy block::

   crappy.blocks.Grapher((`Here everything that should be plotted on the graph`),
   Here the graph settings`)

Example:
   To plot Signal1, Signal2 and Signal3 at a frequency of 2 points
   per second on the same graph, and Signal1 only at a frequency of 10 points per
   second on another graph, one should write::

      Graph1 = crappy.blocks.Grapher(('t(s)', 's1'), ('t(s)', 's2'), freq=2)

      Graph2 = crappy.blocks.Grapher(('t(s)', 's1'), freq=10)

.. note:: Of course it won't work if all the signals haven't been generated before.

Finally, the last step is to link the :ref:`Generator<Generate your signal>` block with the :ref:`Grapher<Plot your signal>` block::

   crappy.link(`name_of_the_Generator`, `name_of_the_Grapher`)

.. note:: 

   For each signal to be plotted, the associated :ref:`Generator` should be linked to the :ref:`Grapher`.


Code Example
----------------

::

   import crappy

   # First: a constant value (2) for 5 seconds
   path1 = {'type':'constant','value':2,'condition':'delay=5'}
   # Second: a sine wave of amplitude 1, freq 1Hz for 5 seconds
   path2 = {'type':'sine','amplitude':1,'freq':1,'condition':'delay=5'}
   # Third: A ramp rising a 1unit/s until the command reaches 10
   path3 = {'type':'ramp','speed':1,'condition':'cmd>10'}
   # Fourth: cycles of ramps going down at 1u/s until cmd is <9
   # then going up at 2u/s for 1s. Repeat 5 times
   path4 = {'type':'cyclic_ramp','speed1':-1,'condition1':'cmd<9',
       'speed2':2,'condition2':'delay=1','cycles':5}

   # The generator: takes the list of all the paths to be generated
   # cmd_label specifies the name to give to the signal
   # freq : the target frequency in points/s
   # spam : Send the value even if it's identical to the previous one
   #   (so that the graph updates continuously)
   # verbose : display some information in the terminal
   gen = crappy.blocks.Generator([path1,path2,path3,path4],
       cmd_label='cmd',freq=50,spam=True,verbose=True)

   # The graph : we will plot cmd vs time
   graph = crappy.blocks.Grapher(('t(s)','cmd'))

   # Do not forget to link them or the graph won't be able to plot anything !
   crappy.link(gen,graph)

   # Let's start the program
   crappy.start()

Another example
-----------------

::

   import crappy
   # In this example, we would like to reach different levels of strain
   # and relax the sample (return to F=0) between each strain level

   speed = 5/60 # mm/sec

   path = [] # The list in which we'll put the paths to be followed

   # We will loop over the values we would like to reach
   # And add two paths for each loop: one for loading and one for unloading
   for exx in [.25,.5,.75,1.,1.5,2]:
     path.append({'type':'constant',
       'value':speed,
        'condition':'Exx(%)>{}'.format(exx)}) # Go up to this level
     path.append({'type':'constant',
       'value':-speed,
       'condition':'F(N)<0'}) # Go down to F=0N

   # Now we can simply give our list of paths as an argument to the generator
   generator = crappy.blocks.Generator(path=path)

   # This block will simulate a tensile test machine
   machine = crappy.blocks.Fake_machine()
   # The generator must be linked to the machine in order to control it
   crappy.link(generator,machine)
   # And the machine must be linked to the generator because we added 
   # conditions on force and strain, so the generator needs to access these 
   # values coming out of the machine
   # Remember : links are one way only !
   crappy.link(machine,generator)

   # Let's add two graphs to visualise in real time
   graph_def = crappy.blocks.Grapher(('t(s)','Exx(%)'))
   crappy.link(machine,graph_def)

   graph_f = crappy.blocks.Grapher(('t(s)','F(N)'))
   crappy.link(machine,graph_f)

   # And start the test
   crappy.start()
