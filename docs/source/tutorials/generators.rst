=======================================
How to generate a signal with CRAPPY ?
=======================================

As in every new :ref:`CRAPPY<What is Crappy ?>` project, the first thing to do is to import it. So your first line of code will simply be::

   import crappy

Then, you can think about what you want your program to do.
Here, we want to generate a signal and after that, we will be able send it into some motor or some card driver or simply just plot it.

So first let's choose what kind of signal we want.

Choose your signal
-------------------

Well, there are 8 different types of signal, all described in the :ref:`generator path` folder.

.. note:: Every signal must be a dictionary providing the parameters to generate it.

Dictionary:
   A dictionary consists of a collection of key-value pairs. Each key-value pair maps the
   key to its associated value. Here is the syntax::

      d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}

   Like that, you don't have to take care about the order because every value will be
   called by it's key.

.. note::

   Each dictionary **MUST** have a ``type`` key.

   Every non cyclic path **MUST** have a ``condition`` key. (Cyclic paths MUST have
   ``condition1`` and ``condition2``.)

A condition can be:
   - A delay (str, in seconds): ``'delay=x'``
   - A condition on a label (str): ``'label<x'`` or ``'label>x'``, it can be a condition
     on an intern label (t(s) or cmd_label), or a condition from a link.
   - ``None`` for a signal that never ends
   - A personalized condition: A fonction which takes the dict of actual labels and returns ``True`` if the path ends.

Here, we will see the 4 major types of signal:

1. Constant signals
++++++++++++++++++++

The simplest signal, it only has 3 keys:
   - ``type``: :ref:`constant`
   - ``value``: Whatever constant int or float value you want to give. It can be in a constant.
   - ``condition``: The condition that will indicate the end of the signal. It can be anything like a position to reach or a time to wait.

Example:
   If we want a constant signal that send the value 5 for 10 seconds, we can write::

      Signal1 = {'type': 'constant', 'value': 5, 'condition': 'delay=10'}

2. Ramp signals
++++++++++++++++

An other quite simple signal that just has 4 keys:
   - ``type``: :ref:`ramp`
   - ``condition``: same as the :ref:`constant signal<1. Constant signals>` condition, it will indicate the end of your signal when it's reached.
   - ``speed``: the speed of the ramp in unit per second (how fast the signal goes to the
     condition?)
   - ``cmd``: the starting value of the ramp. `Optional key`. (If not specified, the
     starting value will match the previous value.)

Example:
   If we want a ramp that goes up at a speed of 2 (mm/s for example) until it reach 30
   (mm), we can write::

      Signal2 = {'type': 'ramp', 'speed': 2, 'condition': 'x(mm)>30')

.. note::

      Of course, ``x(mm)`` must be a variable that contains the real time position of
      whatever we are controlling.

3. Sine signals
++++++++++++++++

Now a signal that has 6 keys:
   - ``type``: :ref:`sine`
   - ``freq``: the frequency of your signal
   - ``amplitude``: the amplitude of your signal
   - ``offset``: add an offset to your signal will move it up and down. `Optional key`.
   - ``phase``: add a phase to your signal will move it right and left. `Optional key`.
   - ``condition``: same as the :ref:`constant signal<1. Constant signals>` condition, it will indicate the end of your signal when it's reached.

Example:
   If we want a sine with a frequency of 0.5, an amplitude of 2, an offset of 1 and that
   stops after 25 seconds, we can write::

      Signal3 = {'type': 'sine', 'freq': .5, 'amplitude': 2, 'offset': 1,
      'condition': 'delay=25'}

   Now, if we want a cosine, with the same parameters as the ``Signal3``, then we can
   write::

      from math import pi
  
      Signal4 = {'type': 'sine', 'freq': .5, 'phase': pi/2, 'amplitude': 2, 'offset': 1,
      'condition': 'delay=25'}

.. note:: We have to import the the number pi, which is in the python module ``math``, in order to use it.

4. Cyclic ramp signals
+++++++++++++++++++++++

This type of signal seams a little bit more complicated, but it is just two simple :ref:`ramps<ramp>`, with the possibility to repeat them. So you already know :ref:`how it works<2. Ramp signals>`!

It has 6 keys:
   - ``type``: :ref:`cyclic ramp`
   - ``condition1``: the condition to reach to stop the first ramp.
   - ``speed1``: the speed of the first ramp (how fast the signal goes to the condition?)
   - ``condition2``: the condition to reach to stop the second ramp.
   - ``speed2``: the speed of the second ramp
   - ``cycles``: number of repetitions of the two ramps. Can be 1. If 0, it will loop forever.

Example:
   If we want a signal that goes up at a speed of 0.1 (mm/s) until it reach 5
   (mm) and then goes down to 2 (mm) at a speed of 0.1 (mm/s), and we want to repeat that
   3 times. Then we can write::

      Signal5 = {'type': 'cyclic_ramp', 'condition1': 'x(mm)>5',
      'speed1': 0.1, 'condition2': 'x(mm)<2', 'speed2': -0.1, 'cycles': 3}

So, we've seen the 4 major types of signal. Now I'll tell you about an other type that can be very useful in some case.

5. Custom signals
++++++++++++++++++

This type allows you to import any signal of your choice from a .csv file (hence the name `custom`).

It only has 2 key:
   - ``type``: :ref:`custom`
   - ``filename``: the name of the .csv file.

.. warning::

   The file must contain 2 columns: The first one with the time, and the second one with
   the value to send.

.. note::

   It will try to send at the right time every timestamp with the associated value.

Example:
   Do you really need it? ::

      Signal6 = {'type': 'custom', 'filename': 'my_custom_signal.csv'}

Once you've created your signal you can generate it by putting it in a :ref:`Generator` crappy block.

Generate your signal
---------------------

So now, we create our :ref:`Generator` just like that::

   OurGenerator = crappy.blocks.Generator([Signalx])

.. note::

      The :ref:`Generator` class is a block, so you find it under the
      folder :ref:`blocks<Blocks>` which is in :ref:`crappy<What is Crappy ?>`:
      ``crappy.blocks.[...]``

      You can replace Signalx with the name of a signal you've already created, or
      directly with the dictionary of the signal you want.

And here it is! Actually, it's not all. A :ref:`Generator` block in crappy must contain a list of dictionaries (hence the list: ``[]``).

Great, we can add other signals! ::

   OurGenerator = crappy.blocks.Generator([Signal1, Signal2, Signal3, Signal4, Signal5])

.. note::

   Once the end of a signal has been reached, the next one in the list begins immediately.
   Once the end of the list have been reached, the :ref:`Generator` stops the
   program.

After that you can add some options that will precise how the :ref:`Generator` should work:
   - ``cmd_label`` will give names to your signals after their generation so you can plot them easily, or match with a camera for example. Very useful when you have several generators.
   - ``freq`` will impose a frequency for the generation of the signal (how many points of the signal should it takes in 1 second ?)
   - ``repeat`` if True, the generator will go back to the beginning of the list endlessly instead of ending the program.

Example:
   If we want to generate our Signal1 with 500 points per second and name it 's1', and
   also generate Signal2 and Signal3 with no required frequency and name it 's2',
   we can write::

      OurGenerator1 = crappy.blocks.Generator([{'type': 'constant',
      'value': 5, 'condition': 'delay=10'}], cmd_label='s1', freq=500)

      OurGenerator2 = crappy.blocks.Generator([Signal2,Signal3], cmd_label='s2')

As simple as that ! Now we can try to show our signal on a graphic for example.

Plot your signal
-----------------

To do so, we will create a :ref:`Grapher` crappy block::

   crappy.blocks.Grapher((`Here, put everything you want to show on the graph`),
   Here, you can add some options`)

Example:
   If we want to plot the Signal1, the Signal2 and the Signal3 at a frequency of 2 points
   per second on the same graph, and the Signal1 only at a frequency of 10 points per
   second on an other graph, we will do something like::

      OurGraph1 = crappy.blocks.Grapher(('t(s)', 's1'), ('t(s)', 's2'), freq=2)

      OurGraph2 = crappy.blocks.Grapher(('t(s)', 's1'), freq=10)

.. note:: It will work only if you have generated all the signals before.

The last thing to do to make it work is to link :ref:`your Generator<Generate your signal>` block with :ref:`your Grapher<Plot your signal>` block::

   crappy.link(`name_of_the_Generator`, `name_of_the_Grapher`)

.. note:: 

   A :ref:`Grapher` should be linked to every :ref:`Generator` that generates the signals
   it want to plot.

   Also, if you send your signal into a :ref:`Machine`, don't forget the link Generator ->
   Machine. And if the :ref:`Generator` has a condition on a measured label (a position
   for example), don't forget the link Machine -> Generator too. 

Example of code
----------------

::

   import crappy

   # First part : a constant value (2) for 5 seconds
   path1 = {'type':'constant','value':2,'condition':'delay=5'}
   # Second: a sine wave of amplitude 1, freq 1Hz for 5 seconds
   path2 = {'type':'sine','amplitude':1,'freq':1,'condition':'delay=5'}
   # Third: A ramp rising a 1unit/s until the command reaches 10
   path3 = {'type':'ramp','speed':1,'condition':'cmd>10'}
   # Fourth : cycles of ramps: go down at 1u/s until cmd is <9
   # then go up at 2u/s for 1s. Repeat 5 times
   path4 = {'type':'cyclic_ramp','speed1':-1,'condition1':'cmd<9',
       'speed2':2,'condition2':'delay=1','cycles':5}

   # The generator: takes the list of all the paths to be generated
   # cmd_label specifies the name to give the signal
   # freq : the target of points/s
   # spam : Send the value even if nothing changed
   #   (so the graph updates continuously)
   # verbose : add some information in the terminal
   gen = crappy.blocks.Generator([path1,path2,path3,path4],
       cmd_label='cmd',freq=50,spam=True,verbose=True)

   # The graph : we will plot cmd over time
   graph = crappy.blocks.Grapher(('t(s)','cmd'))

   # Do not forget to link them or the graph will have nothing to plot !
   crappy.link(gen,graph)

   # Let's start the program
   crappy.start()

An other example
-----------------

::

   import crappy
   # In this example, we would like to reach different levels of strain
   # and relax the sample (return to F=0) between each strain level

   speed = 5/60 # mm/sec

   path = [] # We will put in this list all the paths to be followed

   # We will loop over the values we would like to reach
   # And add two paths for each loop: one for loading and one for unloading
   for exx in [.25,.5,.75,1.,1.5,2]:
     path.append({'type':'constant',
       'value':speed,
        'condition':'Exx(%)>{}'.format(exx)}) # Go up to this level
     path.append({'type':'constant',
       'value':-speed,
       'condition':'F(N)<0'}) # Go down to F=0N

   # Now we can simply give our list of paths to the generator
   generator = crappy.blocks.Generator(path=path)

   # This block will simulate a tensile testing machine
   machine = crappy.blocks.Fake_machine()
   # We must link the generator to the machine to give the command to the machine
   crappy.link(generator,machine)
   # But also the machine to the generator because we added conditions on force
   # and strain, so the generator needs these values coming out of the machine
   # Remember : links are one way only !
   crappy.link(machine,generator)

   # Let's add two graphs to visualise in real time
   graph_def = crappy.blocks.Grapher(('t(s)','Exx(%)'))
   crappy.link(machine,graph_def)

   graph_f = crappy.blocks.Grapher(('t(s)','F(N)'))
   crappy.link(machine,graph_f)

   # And start the experiment
   crappy.start()