===========================================
Creating and using custom objects in Crappy
===========================================

.. role:: py(code)
  :language: python
  :class: highlight

If you have read over the two first pages of the documentation, you should now
have a good understanding of how Crappy works and the possibilities it offers.
However, at that point, you are still limited by the functionalities that the
Blocks and other objects natively distributed with Crappy. **It is now time**
**for you to create your own objects in Crappy, to adapt your scripts to your**
**own needs** ! This page of the tutorials covers the basics of the
instantiation of custom objects, while the :ref:`next and last page of the
tutorials <More about custom objects in Crappy>` covers the advanced aspects of
custom object instantiation.

1. Custom Modifiers
-------------------

The first type of custom objects that we'll cover here are the
:ref:`Modifiers`, because they are by far the simplest objects ! **The**
**Modifiers come in use in a variety of situations, and it is often required**
**to write your own ones as the catalog of Crappy cannot cover every single**
**case**. Luckily, the only requirement for an object to qualify as a Modifier
is to be a :obj:`~collections.abc.Callable`, which means that functions can be
provided !

More precisely, a Modifier should accept a :obj:`dict` as its sole argument and
return a :obj:`dict` as well (:obj:`None` is also accepted). **This**
**dictionary is the representation of a chunk of data flowing through the**
:ref:`Link`, **and the Modifier has direct access to it** ! It can add keys,
delete others, change the value of a key, etc. Each key is a label, and has a
value it carries. In the end, all a Modifier does is to modify the incoming
dictionary and return it after modification. As usual, let's put these concepts
in application on a real runnable example !

For this first example, let's create a Modifier that simply doubles the value
of a given label. If you have understood the last paragraph, this Modifier will
basically only perform something like :py:`data['label'] = data['label'] * 2`.
Now, how to integrate it to your code in practice ? Start by defining the
function somewhere, preferably before the :py:`if __name__ == "__main__"`
statement. Then, simply pass it to the :py:`'modifier'` argument of the target
:class:`~crappy.links.Link`, and that's it ! Do not forget to return the
modified dictionary in the function !

.. literalinclude:: /downloads/custom_objects/custom_modifier.py
   :language: python
   :emphasize-lines: 6-9, 24
   :lines: 1-3, 5-12, 32-41, 43-45, 47-48

In this first example, you can see that instead of replacing the value of the
:py:`'cmd'` label with its double, it was chosen to store the double value in
the newly created :py:`'cmdx2'` label. A new label was added ! This is just how
powerful of a tool the Modifiers are ! Notice how the Modifier is added to the
Link between the :ref:`Generator` and the :ref:`Grapher`, the syntax couldn't
be more straightforward. If you need to change the name of the target label, or
the value of the multiplier, they can simply be modified in the definition of
the function. Alternatively, you could add arguments to you function and use
:obj:`functools.partial` when passing it to the Link, but that is quite an
advanced design already.

In the example, a basic function was passed as a Modifier. While functions are
very versatile and well-known even to beginners, there are many situations when
they will show strong limitations. For example, what if you want to store a
value between two chunks of data ? That is simply not possible with functions !
A concrete example of that is the :class:`~crappy.modifier.Integrate` Modifier,
that integrates a signal over time. It needs to store the integral between
consecutive chunks of data, and therefore cannot rely on a function.

To circumvent this limitation, we made it **possible to instantiate Modifiers**
**as classes** instead of functions. It is mentioned above that the Modifiers
need to be :obj:`~collections.abc.Callable` objects, but a class defining the
:py:`__call__` method is actually callable ! Here is the minimal template of a
Modifier as a class :

.. code-block:: python

   import crappy

   class MyModifier(crappy.modifier.Modifier):

       def __init__(self):
           super().__init__()

       def __call__(self, dic):
           return dic

Some aspects of the code are worth commenting. First, the class should be a
child of the base :class:`crappy.modifier.Modifier`, and initialize its parent
class during :py:`__init__` (via the call to :py:`super().__init__()`). And
second, it needs to define a :py:`__call__` method taking a :obj:`dict` as its
sole argument and returning a :obj:`dict`. The :py:`__call__` method plays the
same role as the function in the previous example, but the class structure
makes it possible to store attributes and to achieve much more complex
behaviors ! To demonstrate that, let's recreate a simpler version of the
Integrate Modifier :

.. literalinclude:: /downloads/custom_objects/custom_modifier.py
   :language: python
   :emphasize-lines: 7-23, 37
   :lines: 1-6, 13-39, 42-44, 46-48

As you can see, compared to the template, several features have been added.
First, the Modifier takes one argument at instantiation, that indicates the
name of the label to integrate over time. This label is indeed provided in the
line where the Modifier is given as an argument to the Link. And then, several
attributes are defined in the :py:`__init__` method to handle the calculation
of the integral during :py:`__call__`. This ability to store values between
consecutive calls is exactly what was desired when using classes as Modifiers !
The two examples presented in this section can finally be merged into a single
big one :

.. literalinclude:: /downloads/custom_objects/custom_modifier.py
   :language: python

You can :download:`download this custom Modifier example
</downloads/custom_objects/custom_modifier.py>` to run it locally on your
machine. An extra example of a custom Modifier can also be found in the
`examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/examples/custom_objects>`_. Except fot what was detailed in this
section, there is actually not much more to know about the definition of
custom Modifiers ! They stand after all among the simplest objects in Crappy.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Modifier, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.modifier.Modifier.log` method provided by the parent
   :class:`~crappy.modifier.Modifier` class. This way, the log messages are
   included in the log file and handled in a nicer way by Crappy.

2. Custom Actuators
-------------------

Creating custom actuators presents no particular challenge once you've read the
two previous sections. All actuators must inherit from the :ref:`Actuator`
object, and must implement the ``open``, ``close``, ``stop`` and either
``set_position`` or ``set_speed`` methods. It is possible to define both.
Additionally, the ``get_speed`` and ``get_position`` methods can be defined.

- ``open`` is meant to perform any action required before starting the assay,
  like initializing hardware and setting parameters.
- ``close`` is meant to perform actions once the assay ends, like switching
  hardware off or closing a bus.
- ``stop`` should instantly stop a device, preferably as fast as possible since
  this method is only called in case an error happens.
- ``set_speed`` and ``set_position`` should make the actuator reach a target
  speed or position.
- ``get_speed`` and ``get_position`` should return the current speed or the
  current position of the actuator.

When an actuator is driven by the :ref:`Machine` block, is repeatedly calls
either ``set_speed`` or ``set_position`` according to the chosen driving mode
and with the input command as argument. If a ``get_speed`` or ``get_position``
exists, it is also repeatedly called according to the chosen mode and a value is
returned. Otherwise no value is returned.

For the sake of the example, let's create a fake actuator that doesn't
necessitate any actual hardware. It will just emulate the behavior of a stepper
motor controlled by a conditioner, i.e. try to reach the target speed or
position and then maintain the target as long as no new command is sent. An
argument allows to tune the refreshment rate for the position calculation.

So let's get to work ! Here's the very minimal actuator class, that does
nothing. It can only be driven in position, but we could simply replace position
by speed.

.. code-block:: python

   import crappy

   class My_actuator(crappy.actuator.Actuator):

       def __init__(self):
           super().__init__()

       def open(self):
           pass

       def set_position(self, pos, speed=3):
           pass

       def stop(self):
           pass

       def close(self):
           pass

Notice that the ``set_position`` method takes the target position as an
argument, but can also take a speed. See the :ref:`Machine` block for details.
Here we'll consider the default speed to be 3 mm/s. Now for the sake of the
example let's add the optional methods and the argument :

.. code-block:: python
   :emphasize-lines: 5-6,14-21

   import crappy

   class My_actuator(crappy.actuator.Actuator):

       def __init__(self, refresh):
           self.t = 1 / refresh
           super().__init__()

       def open(self):
           pass

       def set_position(self, pos, speed=3):
           pass

       def set_speed(self, speed):
           pass

       def get_position(self):
           return 0

       def get_speed(self):
           return 0

       def stop(self):
           pass

       def close(self):
           pass

We're going to use a `threading.Thread <https://docs.python.org/3/library/
threading.html#threading.Thread>`_ to emulate the behavior of the stepper motor.
If you're not familiar with it, check out `this tutorial <https://realpython.
com/intro-to-python-threading/>`_ from RealPython which is complete, accessible
and very well-writen. Or to keep it short, simply consider that two flows of
execution will run in parallel: the regular one handling the user inputs, and
another one exclusively dedicated to emulating the motor. The thread will loop
at a tunable frequency, and simply update the position according to the target
and the current speed. So we also need variables to store the current speed,
position, and position target if any. Without going further into detail, after
adding the thread the code looks this way :

.. code-block:: python
   :emphasize-lines: 2,3,10-17,20,38-58

   import crappy
   import time
   from threading import Thread, RLock

   class My_actuator(crappy.actuator.Actuator):

       def __init__(self, refresh):
           self.t = 1 / refresh
           super().__init__()
           self.position = 0
           self.speed = 0
           self.target_pos = None

           self.stop_thread = False

           self.lock = RLock()
           self.thread = Thread(target=self.run)

       def open(self):
           self.thread.start()

       def set_position(self, pos, speed=3):
           pass

       def set_speed(self, speed):
           pass

       def get_position(self):
           return 0

       def get_speed(self):
           return 0

       def stop(self):
           pass

       def close(self):
           self.stop_thread = True
           self.thread.join()

       def run(self):
           while not self.stop_thread:
               self.lock.acquire()
               if self.target_pos is not None:
                   if self.target_pos < self.position:
                       if self.position - self.speed * self.t < self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position -= self.speed * self.t
                   elif self.target_pos > self.position:
                       if self.position + self.speed * self.t > self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position += self.speed * self.t
               else:
                   self.position += self.speed * self.t
               self.lock.release()
               time.sleep(self.t)

Now the motor emulation is functional, but it doesn't take into account the user
inputs. So now all that's left to do is write the ``get`` and ``set`` methods
and the block will be ready !

.. code-block:: python
   :emphasize-lines: 23-25,28-29,32-33,36-45,48

   import crappy
   import time
   from threading import Thread, RLock

   class My_actuator(crappy.actuator.Actuator):

       def __init__(self, refresh):
           self.t = 1 / refresh
           super().__init__()
           self.position = 0
           self.speed = 0
           self.target_pos = None

           self.stop_thread = False

           self.lock = RLock()
           self.thread = Thread(target=self.run)

       def open(self):
           self.thread.start()

       def set_position(self, pos, speed=3):
           with self.lock:
               self.target_pos = pos
               self.speed = speed

       def set_speed(self, speed):
           with self.lock:
               self.speed = speed

       def get_position(self):
           with self.lock:
               return self.position

       def get_speed(self):
           with self.lock:
               if self.target_pos is None:
                   return self.speed
               else:
                   if self.target_pos < self.position:
                       return -self.speed
                   if self.target_pos > self.position:
                       return self.speed
                   else:
                       return 0

       def stop(self):
           self.set_speed(0)

       def close(self):
           self.stop()

           self.stop_thread = True
           self.thread.join()

       def run(self):
           while not self.stop_thread:
               self.lock.acquire()
               if self.target_pos is not None:
                   if self.target_pos < self.position:
                       if self.position - self.speed * self.t < self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position -= self.speed * self.t
                   elif self.target_pos > self.position:
                       if self.position + self.speed * self.t > self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position += self.speed * self.t
               else:
                   self.position += self.speed * self.t
               self.lock.release()
               time.sleep(self.t)

Now we can integrate our custom actuator in a Crappy script in order to test it.
We'll simply drive it in position, and plot the position and speed.

.. code-block:: python
   :emphasize-lines: 75-103

   import crappy
   import time
   from threading import Thread, RLock

   class My_actuator(crappy.actuator.Actuator):

       def __init__(self, refresh):
           self.t = 1 / refresh
           super().__init__()
           self.position = 0
           self.speed = 0
           self.target_pos = None

           self.stop_thread = False

           self.lock = RLock()
           self.thread = Thread(target=self.run)

       def open(self):
           self.thread.start()

       def set_position(self, pos, speed=3):
           with self.lock:
               self.target_pos = pos
               self.speed = speed

       def set_speed(self, speed):
           with self.lock:
               self.speed = speed

       def get_position(self):
           with self.lock:
               return self.position

       def get_speed(self):
           with self.lock:
               if self.target_pos is None:
                   return self.speed
               else:
                   if self.target_pos < self.position:
                       return -self.speed
                   if self.target_pos > self.position:
                       return self.speed
                   else:
                       return 0

       def stop(self):
           self.set_speed(0)

       def close(self):
           self.stop()

           self.stop_thread = True
           self.thread.join()

       def run(self):
           while not self.stop_thread:
               self.lock.acquire()
               if self.target_pos is not None:
                   if self.target_pos < self.position:
                       if self.position - self.speed * self.t < self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position -= self.speed * self.t
                   elif self.target_pos > self.position:
                       if self.position + self.speed * self.t > self.target_pos:
                           self.position = self.target_pos
                       else:
                           self.position += self.speed * self.t
               else:
                   self.position += self.speed * self.t
               self.lock.release()
               time.sleep(self.t)

   if __name__ == '__main__':

       mot = crappy.blocks.Machine([{'type': 'My_actuator',
                                     'mode': 'position',
                                     'cmd': 'target_position',
                                     'pos_label': 'position',
                                     'speed_label': 'speed',
                                     'refresh': 200}])

       gen = crappy.blocks.Generator([{'type': 'constant',
                                       'value': 0,
                                       'condition': 'delay=5'},
                                      {'type': 'constant',
                                       'value': 10,
                                       'condition': 'delay=5'},
                                      {'type': 'constant',
                                       'value': -10,
                                       'condition': 'delay=10'},
                                      {'type': 'constant',
                                       'value': 0,
                                       'condition': 'delay=5'}],
                                     cmd_label='target_position')

       graph = crappy.blocks.Grapher(('t(s)', 'position'), ('t(s)', 'speed'))

       crappy.link(gen, mot)
       crappy.link(mot, graph)

       crappy.start()

Simply switch the ``'mode'`` key from ``'position'`` to ``'speed'`` to drive
the motor in speed rather than in position !

3. Custom InOuts
----------------

Just like the actuators we've just covered, creating custom inouts is fairly
easy. They must inherit from the :ref:`InOut` object, and implement the
following methods: ``open``, ``close``, and either ``set_cmd`` or ``get_data``.
Note that it is possible to implement both.

- ``open`` is meant to perform any action required before starting the assay,
  like initializing hardware and setting parameters.
- ``close`` is meant to perform actions once the assay ends, like switching
  hardware off or closing a bus.
- ``set_cmd`` takes one or several arguments, and does something with it.
  Usually it is used to set the output of a DAC or to control hardware that
  doesn't fit in the actuator category. But it can actually perform any action.
- ``get_data`` takes no arguments but returns one or several values. Usually it
  returns values read from sensors or ADCs, but again it can actually be any
  kind of data.

**Do not** define ``set_cmd`` or ``get_data`` if not needed, even if the method
does nothing. Crappy could then have issues finding your object in its database.
During the main part of the assay, Crappy will repeatedly call ``set_cmd`` or
``get_data`` depending on what is defined and how the :ref:`IOBlock` is linked
to the other blocks.

For the example we'll use the capacity every computer has to monitor the
real-time memory usage, that will be the value returned by the ``get_data``
method. There's also a way to influence the memory usage by creating big Python
objects, so the ``set_cmd`` method will try to reach a target memory usage. All
memory usages will be given as a percentage.

First let's start from a minimal inout object possessing both the ``set_cmd``
and ``get_data`` methods :

.. code-block:: python

   import crappy
   import time

   class My_inout(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           pass

       def get_data(self):
           return [time.time(), 0]

       def set_cmd(self, cmd):
           pass

       def close(self):
           pass

Note that if the class only uses ``get_data`` or ``set_cmd``, the unused method
should be removed. Now we'll use the :mod:`psutil` module to monitor the memory
consumption. This will only affect the ``get_data`` method :

.. code-block:: python
   :emphasize-lines: 3,14

   import crappy
   import time
   import psutil

   class My_inout(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           pass

       def get_data(self):
           return [time.time(), psutil.virtual_memory().percent]

       def set_cmd(self, cmd):
           pass

       def close(self):
           pass

Now we need to add a structure for adding or removing memory. We'll create a
:obj:`list` containing a variable amount of other (huge) :obj:`list`, what will
allow us to influence the memory usage. We'll also add an argument for setting
a maximal memory usage that shouldn't be reached :

.. code-block:: python
   :emphasize-lines: 7,9,12,18-26,29

   import crappy
   import time
   import psutil

   class My_inout(crappy.inout.InOut):

       def __init__(self, max_mem):
           super().__init__()
           self.max_mem = max_mem

       def open(self):
           self.buf = list()

       def get_data(self):
           return [time.time(), psutil.virtual_memory().percent]

       def set_cmd(self, cmd):
           if cmd > self.max_mem:
               cmd = self.max_mem
           if cmd > psutil.virtual_memory().percent:
               self.buf.append([0] * 1024*1024)
           elif cmd < psutil.virtual_memory().percent:
               try:
                   del self.buf[-1]
               except IndexError:
                   return

       def close(self):
           del self.buf

Now we simply need to integrate out custom inout in a script, that will simply
send a memory usage command and display the current memory usage :

.. code-block:: python
   :emphasize-lines: 31-53

   import crappy
   import time
   import psutil

   class My_inout(crappy.inout.InOut):

       def __init__(self, max_mem):
           super().__init__()
           self.max_mem = max_mem

       def open(self):
           self.buf = list()

       def get_data(self):
           return [time.time(), psutil.virtual_memory().percent]

       def set_cmd(self, cmd):
           if cmd > self.max_mem:
               cmd = self.max_mem
           if cmd > psutil.virtual_memory().percent:
               self.buf.append([0] * 1024*1024)
           elif cmd < psutil.virtual_memory().percent:
               try:
                   del self.buf[-1]
               except IndexError:
                   return

       def close(self):
           del self.buf

   if __name__ == '__main__':

       gen = crappy.blocks.Generator([{'type': 'constant',
                                       'value': 50,
                                       'condition': 'delay=10'},
                                      {'type': 'constant',
                                       'value': 10,
                                       'condition': 'delay=10'},
                                      {'type': 'constant',
                                       'value': 90,
                                       'condition': 'delay=10'}
                                      ], spam=True)

       inout = crappy.blocks.IOBlock('My_inout', labels=['t(s)', 'Memory'],
                                     cmd_labels=['cmd'], spam=True, max_mem=90)

       graph = crappy.blocks.Grapher(('t(s)', 'Memory'))

       crappy.link(inout, graph)

       crappy.link(gen, inout)

       crappy.start()

4. Custom Cameras
-----------------

Adding cameras, and all the other Crappy objects, actually follows the same
scheme as adding blocks but with different rules. Consequently we'll go over it
a bit quicker than for the blocks.

As you may have guessed, custom cameras must inherit from the :ref:`Camera
<Meta Camera>` object (not the :ref:`Camera` block !). They must also initialize
their parent object during ``__init__``. Their mandatory methods are
``get_image``, ``open`` and ``close``, with ``get_image`` returning the current
time and an array. So the very minimal camera would look like that :

.. code-block:: python

   import crappy
   import numpy as np
   import time

   class My_camera(crappy.camera.Camera):

       def __init__(self):
           super().__init__()

       def open(self, **kwargs):
           pass

       def get_image(self):
           return time.time(), np.array([0])

       def close(self):
           pass

Notice the ``**kwargs`` argument in the ``open`` method. When instantiating a
camera block it is possible to specify setting values to the camera object,
we'll cover it later on.

All the methods automatically called by Crappy are there, there's no optional
one like for the blocks. ``open`` is called during Crappy's ``prepare`` and
should be used to initialize streams, open buses, etc. ``close`` is called
during ``finish`` and should be used to close streams, buses, etc. ``get_image``
is called by a ``loop`` during the main part of the program, and should grab a
frame and return it along with the associated timestamp.

Now it is difficult to illustrate how a frame can be grabbed in this example
that mustn't require any hardware, so if you want real examples you should go
over the existing cameras. What can however be explained here is how the
settings can be added and tuned in Crappy. If you never tried to use a camera
in Crappy and your computer has a webcam, you should run the displayer example
to see how the graphical interface allows tuning the settings. To actually start
the test don't forget to close the setting window !

Settings must be added during ``__init__`` using the ``self.add_setting``
method. It takes as arguments the name, a getter method, a setter method, the
limits and the default value. This means that a getter and a setter method have
to be defined for each setting added. The getter method should return the
current value of the setting, (most likely) as returned by the hardware. The
setter method should (most likely) send a command to the hardware in order to
set the parameter. There's a specific syntax for the limits according to the
type:

- A :obj:`bool` indicates that the possible values are :obj:`True` and
  :obj:`False`. A checkbox will be displayed in the interface.
- A :obj:`dict` will have its keys displayed in the graphical interface among
  which the user has to pick one, and the values of the :obj:`dict` correspond
  to the value of the setting actually used in the program.
- A :obj:`tuple` of two elements indicates that the possible values are in the
  range between the first and the second element. If it is a tuple of :obj:`int`
  the possible values will be :obj:`int`, and if it is a :obj:`tuple` of
  :obj:`float` the possible values will be :obj:`float`. In both cases a slider
  will be displayed in the interface.
- :obj:`None` indicates that this setting is not accessible to the user, not
  the most interesting option !

And the default argument simply indicates the default value of the setting,
which should of course be one of the values allowed by the specified type.

So now to illustrate this, let's create a custom camera object that will take a
given image and animate it. We'll add a setting to activate or not the
animation, a setting to tune the animation speed, and one to choose the
orientation. This way we'll cover all the setting types of interest.

The image is distributed in Crappy's package, stored in
``crappy.resources.ve_markers``. To animate it, we'll simply fill a variable
portion of it with black. First we create the structure :

.. code-block:: python
   :emphasize-lines: 9-17, 20-23, 31-47

   import crappy
   import numpy as np
   import time

   class My_camera(crappy.camera.Camera):

       def __init__(self):
           super().__init__()
           self.add_setting('Enable animation',
                            self.get_anim, self.set_anim,
                            True, True)
           self.add_setting('Speed (img/s)',
                            self.get_speed, self.set_speed,
                            (0.5, 2), 1.)
           self.add_setting('Orientation',
                            self.get_orientation, self.set_orientation,
                            {'Vertical': 1, 'Horizontal': 0}, 'Vertical')

       def open(self, **kwargs):
           self.orient = 1
           self.speed = 1.
           self.anim = True
           self.set_all(**kwargs)

       def get_image(self):
           return time.time(), np.array([0])

       def close(self):
           pass

       def get_speed(self):
           return self.speed

       def set_speed(self, speed):
           self.speed = speed

       def get_orientation(self):
           return self.orient

       def set_orientation(self, orient):
           self.orient = orient

       def get_anim(self):
           return self.anim

       def set_anim(self, anim):
           self.anim = anim

Notice the ``self.set_all(**kwargs)`` call during ``open``. It's at this very
moment that the default settings are applied.

Now let's play a bit with the image. We're going to use the timestamp to
determine how blacked the image is. Every ``speed`` seconds the image has
to be completely black, and the mask should then disappear in a linear way. The
displayed array is simply made of the part of the image we keep plus the other
part that's filled with black :

.. code-block:: python
   :emphasize-lines: 23, 27-47

   import crappy
   import numpy as np
   import time

   class My_camera(crappy.camera.Camera):

       def __init__(self):
           super().__init__()
           self.add_setting('Enable animation',
                            self.get_anim, self.set_anim,
                            True, True)
           self.add_setting('Speed (s/img)',
                            self.get_speed, self.set_speed,
                            (1., 5.), 2.)
           self.add_setting('Orientation',
                            self.get_orientation, self.set_orientation,
                            {'Vertical': 1, 'Horizontal': 0}, 'Vertical')

       def open(self, **kwargs):
           self.orient = 1
           self.speed = 1.
           self.anim = True
           self.frame = crappy.resources.ve_markers
           self.set_all(**kwargs)

       def get_image(self):
           t = time.time()
           num_row = int((t % self.get_speed()) *
                         self.frame.shape[0] / self.get_speed())
           num_column = int((t % self.get_speed()) *
                            self.frame.shape[1] / self.get_speed())
           row_mask = np.array([True] * num_row +
                               [False] * (self.frame.shape[0] - num_row))
           column_mask = np.array([True] * num_column +
                                  [False] * (self.frame.shape[1] -
                                             num_column))
           if self.get_anim():
               if self.get_orientation():
                   mask = row_mask
                   return t, np.concatenate((self.frame[mask, :],
                                             self.frame[~mask, :] * 0))
               else:
                   mask = column_mask
                   return t, np.concatenate((self.frame[:, mask],
                                             self.frame[:, ~mask] * 0),
                                            axis=1)
           return time.time(), self.frame

       def close(self):
           pass

       def get_speed(self):
           return self.speed

       def set_speed(self, speed):
           self.speed = speed

       def get_orientation(self):
           return self.orient

       def set_orientation(self, orient):
           self.orient = orient

       def get_anim(self):
           return self.anim

       def set_anim(self, anim):
           self.anim = anim

There's no need to do anything special at exit, so the ``close`` method remains
as it was. Now we'll simply write a short program displaying our animated image.
To do so we only need a Displayer block, and of course our custom camera.
Notice that the argument for choosing a camera object in the :ref:`Camera` block
is a :obj:`str`, you should give the name not the object. We'll also set the
frame rate to 50, because the camera may loop way too fast for the screen to
follow. In the end, here's the working code :

.. code-block:: python
   :emphasize-lines: 70-78

   import crappy
   import numpy as np
   import time

   class My_camera(crappy.camera.Camera):

       def __init__(self):
           super().__init__()
           self.add_setting('Enable animation',
                            self.get_anim, self.set_anim,
                            True, True)
           self.add_setting('Speed (s/img)',
                            self.get_speed, self.set_speed,
                            (1., 5.), 2.)
           self.add_setting('Orientation',
                            self.get_orientation, self.set_orientation,
                            {'Vertical': 1, 'Horizontal': 0}, 'Vertical')

       def open(self, **kwargs):
           self.orient = 1
           self.speed = 1.
           self.anim = True
           self.frame = crappy.resources.ve_markers
           self.set_all(**kwargs)

       def get_image(self):
           t = time.time()
           num_row = int((t % self.get_speed()) *
                         self.frame.shape[0] / self.get_speed())
           num_column = int((t % self.get_speed()) *
                            self.frame.shape[1] / self.get_speed())
           row_mask = np.array([True] * num_row +
                               [False] * (self.frame.shape[0] - num_row))
           column_mask = np.array([True] * num_column +
                                  [False] * (self.frame.shape[1] -
                                             num_column))
           if self.get_anim():
               if self.get_orientation():
                   mask = row_mask
                   return t, np.concatenate((self.frame[mask, :],
                                             self.frame[~mask, :] * 0))
               else:
                   mask = column_mask
                   return t, np.concatenate((self.frame[:, mask],
                                             self.frame[:, ~mask] * 0),
                                            axis=1)
           return time.time(), self.frame

       def close(self):
           pass

       def get_speed(self):
           return self.speed

       def set_speed(self, speed):
           self.speed = speed

       def get_orientation(self):
           return self.orient

       def set_orientation(self, orient):
           self.orient = orient

       def get_anim(self):
           return self.anim

       def set_anim(self, anim):
           self.anim = anim

   if __name__ == '__main__':

       cam = crappy.blocks.Camera('My_camera')

       disp = crappy.blocks.Displayer(framerate=50)

       crappy.link(cam, disp)

       crappy.start()

5. Custom Blocks
----------------

Mandatory methods
Useful attributes
Sending data
Receiving data

Depending on your research field, it is possible that the hardware you're using
or planning to use is not yet implemented in Crappy. Don't worry, Crappy's been
written so that you can easily add new hardware or blocks and use it right
away ! In this first part of the tutorial we'll see how to create custom Crappy
objects directly in a test script. This is the most flexible way to go, but the
objects won't truly be part of Crappy.

What makes a Python class part of Crappy's framework is basically just
inheriting from one of Crappy's base classes. These classes are :ref:`Block`,
:ref:`InOut`,
:ref:`Camera`, :ref:`Actuator` or :ref:`Modifier`, and allow creating
:ref:`blocks`, :ref:`inouts <In / Out>`, :ref:`cameras`, :ref:`actuators` and
:ref:`modifiers` respectively. If you're not familiar with inheritance in
Python you'll find more info `here <https://docs.python.org/3/tutorial/classes.
html#inheritance>`_. Additionally to inheritance, there are also a few specific
rules to follow for each type of object. They will be detailed now for each
type, and illustrated with examples.

To create a block, you first need to instantiate a class inheriting from the
:ref:`Block` class :

.. code-block:: python

   import crappy
  
   class my_block(crappy.blocks.Block):

For your block to integrate within Crappy's framework, it is then necessary to
initialize the parent class :

.. code-block:: python
   :emphasize-lines: 5,6

   import crappy
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self):
           super().__init__()

The last constraint is then to add a ``loop`` method to your class, otherwise an
error will be raised. During the main part of the test (i.e. not when
initializing or closing) the ``loop`` method will be called repeatedly by
Crappy, all you have to do is to define it. You can make it perform any desired
action, but keep in mind that to ensure a smooth termination of the test the
``loop`` method mustn't be blocking (e.g. if it waits for a certain event a
timeout should be given). So here's the **minimal block object**, that literally
does nothing :

.. code-block:: python
   :emphasize-lines: 8,9

   import crappy
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self):
           super().__init__()
  
       def loop(self):
           pass

Apart from the ``loop`` method, several other special methods will be
automatically called by Crappy. Except for ``__init__`` they're however optional
and will not do anything if you don't define them yourself :

- ``__init__`` is called when the class is instantiated, even before
  ``crappy.start()`` is called. Here you should handle the block arguments (if
  it takes any), and declare most of the instance attributes.

- ``prepare`` is called after ``crappy.start()``, i.e. after Crappy truly
  starts but before the actual test is launched. Here you should perform any
  action needed to prepare the test, like creating a data structure or
  initializing hardware.

- ``begin`` is called when the test actually starts, but unlike ``loop`` it is
  only called once. It allows performing a special action on startup, like
  sending a trigger signal to a device. Once it returns, ``loop`` will be called
  repeatedly until the end of the test.

- ``finish`` is called when the assay stops (either in a normal way or due to an
  error). It is meant to perform any action needed before leaving, like
  switching off a device.

In addition to these methods that will be automatically called, you're of course
free to define as many other methods as you need.

There's also one aspect we didn't talk about: the interaction of your block with
the others. So first, the links pointing towards your blocks will be accessible
in the ``self.inputs`` :obj:`list`. You don't have to create it, Crappy handles
it for you. Once you have accessed a link object - we'll call it ``link`` - you
can access the waiting data by calling ``link.recv_chunk()``. It returns a
:obj:`dict`, whose keys are the labels and whose values are :obj:`list`
containing all the values received since the last ``recv`` call. Alternatively,
``link.recv_last()`` returns a :obj:`dict` whose keys are the labels and values
are the last value received in the link (only the last one is kept, others are
discarded). ``link.recv_last()`` might return :obj:`None`, while
``link.recv_chunk()`` is blocking and waits for at least one value to return.
If you're a bit confused no worries, the example will probably make it all
clearer !

Now what about sending data to downstream blocks ? It's much simpler than
receiving data ! The data should first be organized in a :obj:`dict` whose keys
are labels and values are whatever you want to send. Preferably the values
should be :obj:`int`, :obj:`float`, :obj:`bool` or :obj:`str` and not
:obj:`list` or :obj:`dict` for compatibility with the other Crappy blocks. It
means that if your block generates several values for the same label, you should
send them separately and not together in a same :obj:`list`. Once your
:obj:`dict` is created, let's call it ``out``, just call ``self.send(out)``.
That's it ! Again, it will probably be much clearer in an example.

So now to illustrate what was just explained, let's build a block performing
logical operations on signals. This block will take as many logical inputs as
desired, and output the AND, OR and XOR results on all values at once. Since the
values from different blocks may not come at the same frequency, the last
received value is stored for each input and considered to be the current value.
Inputs that didn't send a value yet are all considered either :obj:`True` or
:obj:`False` according to the user's choice. Now let's get to work !

We're starting from the minimal template given previously. What arguments does
the user need to provide ? First the labels to consider as inputs and then the
label of the outputs. We also decided that the user could provide the default
value for labels that do not have a value yet. For simplicity let's say that
only one label should be provided for the output, to which the suffixes
``'_AND', '_OR', '_XOR'`` will be added. So if we stick to the essentials the
``__init__`` method should be pretty concise :

.. code-block:: python
   :emphasize-lines: 5,7-9

   import crappy
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self, cmd_labels, label='logical', default=False):
           super().__init__()
           self.cmd_labels = cmd_labels
           self.out_label = label
           self.default = default
  
       def loop(self):
           pass

Now we need to build a data structure before startup, so let's write a
``prepare`` method. We simply need to define one variable per label, which will
store the last received value or the default value if no value was received.
A :obj:`dict` is well-suited for that. We'll keep the syntax understandable to
everyone even though it's not the optimal :

.. code-block:: python
   :emphasize-lines: 11-14

   import crappy
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self, cmd_labels, label='logical', default=False):
           super().__init__()
           self.cmd_labels = cmd_labels
           self.out_label = label
           self.default = default
  
       def prepare(self):
           self.values = {}
           for label in self.cmd_labels:
               self.values[label] = self.default
  
       def loop(self):
           pass

Now the main part that will be run again and again during the test. We actually
simply need to get the last received value for each label, calculate the 3
logical outputs and send the results with the right labeling. For each link
we'll try to receive values, if there's any we'll go through the labels to check
if there are ones matching with the ``cmd_labels``, and if so we'll write the
corresponding value to our ``self.values`` structure. The logical values
calculations may be a bit too straightforward depending on your level in Python,
but it's not the important part. We must not forget to add the time to the
output. All of this should be pretty quick :

.. code-block:: python
   :emphasize-lines: 2,18-35

   import crappy
   import time
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self, cmd_labels, label='logical', default=False):
           super().__init__()
           self.cmd_labels = cmd_labels
           self.out_label = label
           self.default = default
  
       def prepare(self):
           self.values = {}
           for label in self.cmd_labels:
               self.values[label] = self.default
  
       def loop(self):
           for link in self.inputs:
               recv_dict = link.recv_last()
               if recv_dict is not None:
                   for label in recv_dict:
                       if label in self.cmd_labels:
                           self.values[label] = recv_dict[label]
  
           log_and = all(log_value for log_value in self.values.values())
           log_or = any(log_value for log_value in self.values.values())
           val_list = list(self.values.values())
           log_xor = any(log_1 ^ log_2 for log_1, log_2 in
                         zip(val_list[:-1], val_list[1:]))
  
           out = {'t(s)': time.time() - self.t0,
                  self.out_label + '_AND': log_and,
                  self.out_label + '_OR': log_or,
                  self.out_label + '_XOR': log_xor}
           self.send(out)

There's no particular need to perform any action before program termination, so
a ``finish`` method is not needed. Our custom block is then finished ! Now for
using it like a regular Crappy object, all you need to do is to instantiate it.
Here's an example code that will allow us to test it :

.. code-block:: python
   :emphasize-lines: 37-65

   import crappy
   import time
  
   class my_block(crappy.blocks.Block):
  
       def __init__(self, cmd_labels, label='logical', default=False):
           super().__init__()
           self.cmd_labels = cmd_labels
           self.out_label = label
           self.default = default
  
       def prepare(self):
           self.values = {}
           for label in self.cmd_labels:
               self.values[label] = self.default
  
       def loop(self):
           for link in self.inputs:
               recv_dict = link.recv_last()
               if recv_dict is not None:
                   for label in recv_dict:
                       if label in self.cmd_labels:
                           self.values[label] = recv_dict[label]
  
           log_and = all(log_value for log_value in self.values.values())
           log_or = any(log_value for log_value in self.values.values())
           val_list = list(self.values.values())
           log_xor = any(log_1 ^ log_2 for log_1, log_2 in
                         zip(val_list[:-1], val_list[1:]))
  
           out = {'t(s)': time.time() - self.t0,
                  self.out_label + '_AND': log_and,
                  self.out_label + '_OR': log_or,
                  self.out_label + '_XOR': log_xor}
           self.send(out)
  
   if __name__ == '__main__':
  
       gen_1 = crappy.blocks.Generator([{'type': 'constant',
                                         'value': 0,
                                         'condition': 'delay=10'},
                                        {'type': 'constant',
                                         'value': 1,
                                         'condition': 'delay=5'}],
                                        cmd_label='cmd_1')
  
       gen_2 = crappy.blocks.Generator([{'type': 'constant',
                                         'value': 0,
                                         'condition': 'delay=5'},
                                        {'type': 'constant',
                                         'value': 1,
                                         'condition': 'delay=10'}],
                                        cmd_label='cmd_2')
  
       logic = my_block(cmd_labels=['cmd_1', 'cmd_2'])
  
       graph = crappy.blocks.Grapher(('t(s)', 'logical_AND'),
                                     ('t(s)', 'logical_OR'),
                                     ('t(s)', 'logical_XOR'))
  
       crappy.link(gen_1, logic)
       crappy.link(gen_2, logic)
       crappy.link(logic, graph)
  
       crappy.start()

This is it ! See how straightforward it was to use the block we just created.
Note that it can easily be reused elsewhere without copy/pasting by just
importing it, see the corresponding `documentation on imports <https://docs.
python.org/3/reference/import.html>`_.
