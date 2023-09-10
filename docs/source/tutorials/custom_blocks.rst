=============================================
Using custom blocks and adding them to Crappy
=============================================

1. Using custom blocks in scripts
---------------------------------

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

1.a. blocks
+++++++++++

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
python.org/3/reference/import.html>`_. Alternatively, it can also be permanently
added, see :ref:`the second section of this tutorial <2. Permanently adding
custom blocks to Crappy>`

1.b. cameras
++++++++++++

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

1.c. actuators
++++++++++++++

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

1.d. inouts
+++++++++++

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

1.e. modifiers
++++++++++++++

The last type of Crappy object we have to go over is the modifier. The syntax
is much freer than for the previous objects, since modifiers can actually be
either classes or just functions. Using functions is the easiest way to go, and
that's what we recommend. In most cases, classes are necessary either if you
need to store data between loops, or if you want to easily instantiate similar
modifiers but with a varying parameter.

So let's begin with the functions. It is really straightforward since any
function will be accepted as a modifier. For it to work properly, functions
should take a :obj:`dict` as only parameter, and return only a :obj:`dict`. This
:obj:`dict` will contain the data coming from the upstream block. Its keys are
the different labels, and to each key is associated a single value. The
available labels and the type of the values depend on the kind of block the link
comes from.

Inside the function, you actually have a direct access to the data flowing
through the links. You can add keys, delete others or modify their values, it's
all up to you ! So as an example, let's say we want to invert the value of the
``'lab'`` label (and leave it to 0 if it's 0). We'll create three functions for
that: one modifying the label, one adding a new label ``'lab_inv'`` and keeping
the original one, and one adding the new label but deleting the original one :

.. code-block:: python

   def modify(dic):
       if dic['lab'] != 0:
           dic['lab'] = 1 / dic['lab']
       return dic

   def add(dic):
       if dic['lab'] != 0:
           dic['lab_inv'] = 1 / dic['lab']
       else:
           dic['lab_inv'] = 0
       return dic

   def add_del(dic):
       if dic['lab'] != 0:
           dic['lab_inv'] = 1 / dic['lab']
       else:
           dic['lab_inv'] = 0
       dic.pop('lab')
       return dic

Now if you need to create a class for your modifier, there's only one condition:
the class must define an ``evaluate`` method. This method should be similar to
the functions we defined previously: take only a :obj:`dict` as argument (except
for the ``self`` argument), and return a :obj:`dict`. The only difference with
the functions is that the ``evaluate`` method has access to class and instance
attributes, opening more possibilities. Also, your class can (but doesn't need
to) inherit from the :ref:`Modifier` object. This may become mandatory in a
future release.

To fully demonstrate the use of a modifier as a class, let's create one that
sends a different signal if the difference between the maximum and the minimum
values ever recorded is higher than a given threshold. The user has to specify
the label to listen to, and the label on which the signal will be sent. This
would be impossible with a function as it cannot store the successive values of
the signal. The labels could also not simply be given as arguments. Let's start
from the minimal template :

.. code-block:: python

   import crappy

   class My_modifier(crappy.modifier.Modifier):

       def __init__(self):
           super().__init__()

       def evaluate(self, dic):
           return dic

Let's now add our specific features :

.. code-block:: python
   :emphasize-lines: 5,7-11,14-27

   import crappy

   class My_modifier(crappy.modifier.Modifier):

       def __init__(self, label_in, label_out, threshold):
           super().__init__()
           self.label_in = label_in
           self.label_out = label_out
           self.threshold = threshold
           self.max = None
           self.min = None

       def evaluate(self, dic):
           if self.max is None:
               self.max = dic[self.label_in]
           if self.min is None:
               self.min = dic[self.label_in]

           if dic[self.label_in] > self.max:
               self.max = dic[self.label_in]
           if dic[self.label_in] < self.min:
               self.min = dic[self.label_in]

           if self.max - self.min > self.threshold:
               dic[self.label_out] = 1
           else:
               dic[self.label_out] = 0

           return dic

We can now test our modifiers in a simple script. It will just generate a signal
and display it along with the modified signals.

.. code-block:: python
   :emphasize-lines: 5,7-11,14-27

   import crappy

   class My_modifier(crappy.modifier.Modifier):

       def __init__(self, label_in, label_out, threshold):
           super().__init__()
           self.label_in = label_in
           self.label_out = label_out
           self.threshold = threshold
           self.max = None
           self.min = None

       def evaluate(self, dic):
           if self.max is None:
               self.max = dic[self.label_in]
           if self.min is None:
               self.min = dic[self.label_in]

           if dic[self.label_in] > self.max:
               self.max = dic[self.label_in]
           if dic[self.label_in] < self.min:
               self.min = dic[self.label_in]

           if self.max - self.min > self.threshold:
               dic[self.label_out] = 1
           else:
               dic[self.label_out] = 0

           return dic

   def add(dic):
       if dic['signal'] != 0:
           dic['signal_inv'] = 1 / dic['signal']
       else:
           dic['signal_inv'] = 0
       return dic

   if __name__ == '__main__':

       gen = crappy.blocks.Generator([{'type': 'sine',
                                       'freq': 1/3,
                                       'amplitude': 1,
                                       'offset': 1,
                                       'condition': 'delay=15'},
                                      {'type': 'ramp',
                                       'speed': 1/3,
                                       'condition': 'delay=12.5'}],
                                     cmd_label='signal')

       graph_inv = crappy.blocks.Grapher(('t(s)', 'signal'),
                                         ('t(s)', 'signal_inv'))

       graph_thresh = crappy.blocks.Grapher(('t(s)', 'signal'),
                                            ('t(s)', 'signal_thresh'))

       crappy.link(gen, graph_inv, modifier=[add])
       crappy.link(gen, graph_thresh, modifier=[My_modifier('signal',
                                                            'signal_thresh',
                                                            3)])

       crappy.start()

2. Permanently adding custom blocks to Crappy
---------------------------------------------

You can either add an object locally or to the entire project. If it's locally,
you'll be the only one having access to the modifications but you're free to do
whatever you want. Any modification to the entire project requires an approval
and is subject to few rules, but then everyone will be able to use your object.
**We always recommend you to add any improvement to the entire project, the more
contributions the better !** Here are the different possibilities :

- **Adding your object locally** :

  - If Crappy was installed using ``git``, simply copy a ``.py`` file
    containing your block or your object into the right folder. The class
    inheritance changes compared with an in-script object definition. Refer to
    objects that are already implemented for the appropriate syntax. For example
    if you had :

    .. code-block:: python

       import crappy

       class My_block(crappy.blocks.Block):

    Now you should have :

    .. code-block:: python

       from .block import Block

       class My_block(Block)

    Then modify the ``__init__.py`` file of the folder in which you placed your
    new object. For example if the block mentioned a few lines above is
    contained in ``my_block.py``, you should write in
    ``crappy/blocks/__init__.py`` :

    .. code-block:: python

       from .my_block import My_block

    If you included docstrings in your file and you wish to include them in a
    local documentation, add your object in the corresponding ``.rst`` file in
    the ``/docs/source/crappydocs/`` folder. Again the syntax should be
    self-explanatory. Still following the same example, here we should write in
    ``/docs/source/crappydocs/blocks.rst`` :

    .. code-block:: rst

       My Block
       --------
       .. automodule:: crappy.blocks.my_block
          :members:

    Now simply reinstall Crappy (see :ref:`Installation`, the syntax slightly
    differs according to your OS) and that's it, you can freely use your object
    in scripts !

  - If Crappy was installed using ``pip``,  the quick-and-dirty way is to do
    almost the same steps as in the previous point, except now Crappy's folder
    may be harder to find. If it is installed in a virtualenv you should find it
    easily, otherwise you can open a Python terminal, and type :

      >>> import crappy
      >>> crappy

    This will display the location of Crappy's files. Now like in the previous
    point add your ``.py`` file to the right folder with the right import and
    inheritance modifications, change the corresponding ``__init__.py`` file,
    and that's it ! Next time you import Crappy your object should be available.

      .. Important::
         It's likely that your modifications will be discarded if you then
         update Crappy using ``pip`` !

- **Adding your object to the Crappy project** : see the
  :ref:`Developers information` section. There are a few rules to respect, but
  if your pull request is accepted then all the Crappy users will be able to use
  your object !