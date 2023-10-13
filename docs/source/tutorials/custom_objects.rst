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

After introducing how custom Modifiers work in the first section, this second
section will focus on the use of custom :ref:`Actuators`. Knowing how to add
and use your own :class:`~crappy.actuator.Actuator` objects in Crappy is
critical for anyone who wants to drive their own tests, as **the equipment**
**that we already integrated in the module will likely not match the one you**
**have at your disposal**. In this situation, you'll have no choice but to
implement it yourself in Crappy !

Unlike Modifiers, Actuators usually communicate with hardware. The code for
driving a custom Actuator is therefore quite different than the one for a
Modifier that simply handles data. That being said, what are the steps for
writing code for your Actuator ? First, **make sure that the hardware you**
**want to use can be driven with Python** ! If that is not the case, you
unfortunately won't be able to drive it with Crappy. There are usually many
different ways to drive hardware from Python, so **just because there is no**
**Python library for your device doesn't mean you cannot drive it from**
**Python** ! Here are a few ways to drive hardware from Python :

- Use a Python library provided by the manufacturer or a third party
- Get the correct communication syntax and protocol from the datasheet, and
  code the communication yourself over the right protocol (serial, USB, I2C,
  SPI, etc.)
- Send commands in the terminal from Python, if the manufacturer provides a way
  to drive hardware from the console
- Write Python bindings for a C/C++ library, if the manufacturer provides one

Note that all these solutions require various levels of expertise in Python,
some are very simple and others much more difficult ! For now, let's assume
that you have found a way to drive your device from Python. The next step is to
**write a draft code completely independent from Crappy, and in which you're**
**able to initialize the connection to the device, set its parameters, set**
**its speed or target position, acquire its current speed or position, and**
**properly de-initialize the device and the connection**. We advise you to
write this independent draft so that in a first time you don't have to bother
with Crappy's syntax.

Once you have a working draft, it is time to integrate it in Crappy ! Just like
the Modifier previously, there is also a template for the
:class:`~crappy.actuator.Actuator` objects :

.. code-block:: python

   import crappy

   class MyActuator(crappy.actuator.Actuator):

       def __init__(self):
           super().__init__()

       def open(self):
           ...

       def set_position(self, pos, speed):
           ...

       def set_speed(self, speed):
           ...

       def get_position(self):
           ...

       def get_speed(self):
           ...

       def stop(self):
           ...

       def close(self):
           ...

This template looks much bigger than the one for the Modifier, but actually
part of the methods are optional. Out of the :py:`set_position`,
:py:`set_speed`, :py:`get_position` and :py:`get_speed` methods, you only need
to implement at least one. You could even get away with implementing none, but
the interest is limited. Let's review what each method is intended for :

- :meth:`~crappy.actuator.Actuator.__init__` is where you should initialize the
  Python objects that your class uses. It is also where the class accepts its
  arguments, that are given in the dictionary passed to the :ref:`Machine`
  Block. Avoid interacting with hardware already in this method. Also, don't
  forget to initialize the parent class with :py:`super().__init__()` !
- In :meth:`~crappy.actuator.Actuator.open` you should perform any action
  required for configuring the device. That includes opening the communication
  with it, configuring its parameters, or maybe energizing it.
- :meth:`~crappy.actuator.Actuator.set_speed` and
  :meth:`~crappy.actuator.Actuator.set_position` are for sending respectively a
  target speed or position command to the device. It is possible to implement
  both if the device supports it, or only one, or even none if the device is
  only used as a sensor in Crappy. These methods take as an argument the target
  speed and position respectively, and do not return anything. As you may have
  guessed, :meth:`~crappy.actuator.Actuator.set_speed` is called if the
  Actuator is driven in *speed* mode, and
  :meth:`~crappy.actuator.Actuator.set_position` is called if the Actuator is
  driven in *position* mode. These methods are only called if the Machine Block
  receives commands via an incoming :ref:`Link`. Note that the
  :meth:`~crappy.actuator.Actuator.set_position` method always accepts a second
  :py:`speed` argument, that may be equal to :obj:`None`. You'll find more
  about it in :ref:`a dedicated section on the next page
  <3. More about custom Actuators>`.
- In a similar way, :meth:`~crappy.actuator.Actuator.get_speed` and
  :meth:`~crappy.actuator.Actuator.get_position` are for acquiring the current
  speed or position of the device. These methods do not take any argument, and
  return the acquired speed or position as a :obj:`float`. Again, it is
  possible to define both methods, or only one, or none. They can be called no
  matter what the driving mode is, provided that the :py:`position_label`
  and/or :py:`speed_label` keys are provided as arguments in the dictionary
  passed to the Machine Block. The data is only sent to downstream Blocks if
  the Machine Block has outgoing Links.
- :meth:`~crappy.actuator.Actuator.stop` should stop the device in the fastest
  and more durable possible way. It is called if a problem occurs, and at the
  very end of the test. If there is no other way to stop the device than
  setting its speed to 0, this method doesn't need to be defined.
- In :meth:`~crappy.actuator.Actuator.close` you should perform any action
  required for properly de-initializing the device. For example, this is where
  you put a device to sleep mode or close the connection to it.

Also note how the class inherits from the parent
:class:`crappy.actuator.Actuator` class. That must not be forgotten, otherwise
the Actuator won't work ! At that point, you should **use your working draft**
**to fill in the corresponding methods in the template**. You should be able to
obtain a working Actuator in no time! To give you a better idea of what the
result could look like, here's an example inspired from the custom Actuator in
the `examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/examples/custom_objects>`_ :

.. literalinclude:: /downloads/custom_objects/custom_actuator.py
   :language: python
   :emphasize-lines: 7-36
   :lines: 1-37

As you can see, the :py:`__init__` method takes on argument and initializes
various objects used elsewhere in the class. The :py:`open` method does not
much, as this example emulates hardware and does not interact with any
real-world device. For the same reason, the :py:`close` and :py:`stop` methods
are missing. This Actuator can only be driven in speed, so the
:py:`set_position` method is also missing. The :py:`set_speed` and
:py:`get_speed` methods are present for setting the target speed and measuring
the current one, as well as the :py:`get_position` method since the position
is also measurable. Now that the Actuator is defined, it is time to add some
context to make it run :

.. literalinclude:: /downloads/custom_objects/custom_actuator.py
   :language: python

You can :download:`download this custom Actuator example
</downloads/custom_objects/custom_actuator.py>` to run it locally on your
machine. **The concepts presented in this section will be re-used for all the**
**other types of custom objects**, so make sure to understand them well ! You
can also have a look at the `Actuators distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/src/crappy/actuator>`_ to
see how the implementation of real-life Actuators looks like.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Actuator, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.actuator.Actuator.log` method provided by the parent
   :class:`~crappy.actuator.Actuator` class. This way, the log messages are
   included in the log file and handled in a nicer way by Crappy.

3. Custom InOuts
----------------

Creating custom :ref:`In / Out` objects is extremely similar to creating custom
:ref:`Actuators`, so make sure to first read and understand the previous
section first ! Just like for Actuators, **anyone who wants to drive their**
**own setups with Crappy will surely need at some point to create their own**
:class:`~crappy.inout.InOut` **objects**. This section covers the specificities
of creating new InOuts.

3.a Regular mode
++++++++++++++++

First, let's cover the similarities with the creation of Actuator objects, in
the case of a regular usage. The case of the *streamer* mode is covered in
:ref:`the next sub-section <3.b Streamer mode>`. Just like for an Actuator,
you'll need to write your class before the :py:`if __name__ == "__main__"`
statement, or to import it from another file. You should also start from a
working draft in which you're able to drive your device in Python. And in both
cases, creating your custom class can be simply achieved by filling in a
template ! That being said, the objects and methods to manipulate will of
course differ, here's how the template for an InOut looks like :

.. code-block:: python

   import crappy

   class MyInOut(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           ...

       def get_data(self):
           ...

       def set_cmd(self, cmd):
           ...

       def close(self):
           ...

As you can see, there are two main differences. First, the parent class from
which your InOut must inherit is now :class:`crappy.inout.InOut`, and second
you now have to define the :py:`get_data` and/or :py:`set_cmd` methods. The
:meth:`~crappy.inout.InOut.__init__`, :meth:`~crappy.inout.InOut.open` and
:meth:`~crappy.inout.InOut.close` methods serve the same purpose as for the
Actuators. The new methods are :

- :meth:`~crappy.inout.InOut.get_data`, that takes no argument and should
  return the data acquired by the device. The first returned value must be the
  timestamp of the acquisition, as returned by :obj:`time.time`. Then, you can
  return as many values as you want, usually corresponding to different
  channels you device can acquire. The number of returned values should always
  be the same, and for each value a label should be given in the :py:`labels`
  argument of the :ref:`IOBlock`. The data will only be acquired if the IOBlock
  has outgoing :ref:`Links` !
- :meth:`~crappy.inout.InOut.set_cmd` takes one or several arguments, and does
  not return anything. Instead, the arguments it receives should be used to set
  commands on the device to drive. The number of arguments this method receives
  only depends on the number of labels given as the :py:`cmd_labels` argument
  to the IOBlock. The order of the arguments is also the same as the one of the
  labels in :py:`cmd_labels`.

Once again, let's switch to practice by writing a custom InOut class. We'll
keep it very basic, you can browse the `InOuts distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/src/crappy/inout>`_ to
have an overview of what a real-life InOut looks like.

.. literalinclude:: /downloads/custom_objects/custom_inout.py
   :language: python
   :emphasize-lines: 7-22
   :lines: 1-23

In this example, the InOut simply stores two values. When :py:`get_data` is
called, it simply returns these two values as well as a timestamp. When,
:py:`set_cmd` is called, it expects two arguments and sets their values as the
new stored values. Let's now integrate the InOut into a runnable code :

.. literalinclude:: /downloads/custom_objects/custom_inout.py
   :language: python

In order to obtain two commands from a single :ref:`Generator`, a
:ref:`Modifier` is added to create a new label. In the IOBlock, the two labels
carrying the commands are indicated in the :py:`cmd_labels` argument. The
values acquired by the :py:`get_data` method are transmitted to the
:ref:`Grapher` Block over the labels indicated in the :py:`labels` argument of
the IOBlock. And in the end it all works fine together ! You can
:download:`download this custom InOut example
</downloads/custom_objects/custom_inout.py>` to run it locally on your
machine, and have a look at the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/custom_objects>`_
to find more examples of custom InOut objects.

.. Note::
   If you want to have debug information displayed in the terminal from your
   InOut, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.inout.InOut.log` method provided by the parent
   :class:`~crappy.inout.InOut` class. This way, the log messages are
   included in the log file and handled in a nicer way by Crappy.

3.b Streamer mode
+++++++++++++++++

If you want to be able to use your custom InOut object in *streamer* mode, the
methods described above will not be sufficient. Instead, **there is a**
**particular framework to follow that is detailed in this sub-section**. For
more details on how to use the *streamer* mode, refer to the :ref:`Dealing with
streams section <4. Dealing with streams>` of the tutorials. Getting straight
to the point, here's how the template for an InOut supporting the *streamer*
mode looks like :

.. code-block:: python

   import crappy

   class MyStreamerInOut(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           ...

       def get_data(self):
           ...

       def set_cmd(self, cmd):
           ...

       def start_stream(self):
           ...

       def get_stream(self):
           ...

       def stop_stream(self):
           ...

       def close(self):
           ...

It is the exact same as the one for the regular InOuts, except there are three
additional methods. **You can still define the**
:meth:`~crappy.inout.InOut.get_data` **and**
:meth:`~crappy.inout.InOut.set_cmd` **methods, so that your InOut can be used**
**both in regular and streamer mode** depending on the value of the
:py:`streamer` argument of the :ref:`IOBlock` ! Now, what are the new methods
supposed to do ?

- :meth:`~crappy.inout.InOut.start_stream` should perform any action required
  to start the acquisition of a stream on the device. It can for example
  configure the device, or send a specific command. It is fine not to define
  this method if no particular action is required. The actions performed in
  this method must be specific to the *streamer* mode, the general
  initialization commands should still be executed in the
  :meth:`~crappy.inout.InOut.open` method.
- :meth:`~crappy.inout.InOut.get_stream` is where the stream data is acquired.
  This method does not take any parameter, and should return two objects. The
  first one is a :mod:`numpy` array of shape `(m,)`, and the second another
  :mod:`numpy` array of shape `(m, n)`, where `m` is the number of timestamps
  and `n` the number of channels of the InOut. The first array contains only
  one column with all the timestamps at which data was acquired. It is
  equivalent to the timestamp value in :meth:`~crappy.inout.InOut.get_data`,
  except here there are several timestamps to return. The second array is a
  table containing for each timestamp and each label the acquired value.
  Instead of returning one value per channel like in the
  :meth:`~crappy.inout.InOut.get_data`, only one object contains all the
  values.
- :meth:`~crappy.inout.InOut.stop_stream` should perform any action required
  for stopping the acquisition of the stream. It is fine not to define this
  method if no particular action is required. The actions performed in this
  method must be specific to the *streamer* mode, the general de-initialization
  commands should still be executed in the :meth:`~crappy.inout.InOut.close`
  method.

At that point, the syntax of the objects to return in the
:meth:`~crappy.inout.InOut.get_stream` method might still not be very clear to
you. A short example should help you understand it, it's not as difficult as it
first seems ! The following example is the continuation of the one presented in
the previous sub-section :

.. literalinclude:: /downloads/custom_objects/custom_inout_streamer.py
   :language: python
   :emphasize-lines: 5, 25-35, 52, 54-55, 61-62

The first difference is that the module :mod:`numpy` must be used, but that is
not a problem since it is a requirement of Crappy. Then, the
:meth:`~crappy.inout.InOut.get_stream` method is defined. The structure of the
returned arrays should not be too difficult to understand if you're familiar
with :mod:`numpy`. Note that here the returned arrays are built iteratively,
but for real-life InOuts they are usually derived directly from a big message
sent by the driven device. Just like previously, the
:meth:`~crappy.inout.InOut.start_stream`,
:meth:`~crappy.inout.InOut.stop_stream`,
:meth:`~crappy.inout.InOut.open` and :meth:`~crappy.inout.InOut.close` methods
don't need to be defined. At the IOBlock level, the :py:`streamer` argument is
now set to :obj:`True`, and the :py:`labels` argument has also been updated.
Finally, a :ref:`Demux` Modifier is now needed on the :ref:`Link` from the
IOBlock to the Grapher in order for the data to be displayed.

You can :download:`download this custom streamer InOut example
</downloads/custom_objects/custom_inout_streamer.py>` to run it locally on your
machine. The only real difficulty with the instantiation of custom InOuts
supporting the *streamer* mode is building the arrays to return, but you can
find an additional example of a custom InOut in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/custom_objects>`_
and in the `InOuts distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/src/crappy/inout>`_.

4. Custom Cameras
-----------------

Now that you're getting familiar with the instantiation of custom objects in
Crappy, adding your own :ref:`Cameras` to Crappy should not present any
particular difficulty. **The camera management is one of the big strengths of**
**Crappy, as Crappy handles the parallelization of the acquisition, display,**
**recording and processing of the images.** The result is a very high
performance when dealing with images. Also, **Crappy comes with a variety of**
**Blocks for performing advanced and optimized image processing**, that you can
use with your own :class:`~crappy.camera.Camera` objects (see the
:ref:`DIS Correl`, :ref:`DIC VE` or :ref:`Video Extenso` Blocks for example).
For these reasons, integrating your own cameras into Crappy might prove very
advantageous.

The first step for integrating a camera in Crappy is to check whether it can be
read by one of the existing :ref:`Cameras`. The :ref:`Camera OpenCV` and
:ref:`Camera GStreamer` objects in particular are designed to be compatible
with a wide range of cameras, using the :mod:`opencv-python` and GStreamer
modules respectively, so it might be worth testing them on your hardware
first ! If your camera is not compatible, you'll have to write your own
:class:`~crappy.camera.Camera` object.

Just like in the previous sections, there is a template for the Camera
objects :

.. code-block:: python

   import crappy

   class MyCamera(crappy.camera.Camera):

       def __init__(self):
           super().__init__()

       def open(self, **kwargs):
           ...

       def get_image(self):
           ...

       def close(self):
           ...

The base class from which each Camera must inherit is
:class:`crappy.camera.Camera`. The :meth:`~crappy.camera.Camera.open` and
:meth:`~crappy.camera.Camera.close` methods are, as usual, meant for
(de-)initializing the camera and the connection to it. A big difference with
the custom classes that were defined in the previous sections is that here the
:meth:`~crappy.camera.Camera.__init__` method does not accept any argument.
Instead, all the arguments to pass to the Camera will be given as *kwargs* to
the :meth:`~crappy.camera.Camera.open` method. Why did we choose a different
implementation ? As detailed below, it is possible to define settings of the
Camera objects that can be adjusted interactively in a nice
:ref:`Camera Configurator` interface. Because the settings are handled in a
special way and applied during :meth:`~crappy.camera.Camera.open`, it then
makes sense to catch the arguments here !

The method unique to the Camera objects is
:meth:`~crappy.camera.Camera.get_image`, that should acquire one image at a
time, normally by communicating with the hardware. This method does not accept
any argument, and should return two values. The first one is the timestamp at
which the image was acquires, as returned by :obj:`time.time`. The second one
is the acquired image, as a :mod:`numpy` array or numpy-compatible object. The
image can be an array of dimension two, if it is a grey level image, or of
dimension three if it is a color image. It can also be encoded over 8 or 16
bits indifferently.

As always, let's write a basic example to make it clear how the implementation
should look like in an actual script, inspired from an `example available on
GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/examples/
custom_objects>`_ :

.. literalinclude:: /downloads/custom_objects/custom_camera.py
   :language: python
   :emphasize-lines: 8-41
   :lines: 1-17, 52-76, 86-97

In this first example, the Camera object generates a random image with several
settings that can be adjusted in the :meth:`~crappy.camera.Camera.__init__`
method. If you run the script, you'll however notice that the settings cannot
be interactively tuned in the configuration window. The possibility to do so
will be introduced in the next paragraphs. Except for the part that customizes
the output image, the syntax is fairly simple. Once the image is acquired, it
just has to be returned by the :meth:`~crappy.camera.Camera.get_image` method
along with a timestamp. Here, the :meth:`~crappy.camera.Camera.open` and
:meth:`~crappy.camera.Camera.close` methods don't need to be defined as there
is no interactive setting defined nor any hardware to (de-)initialize.

In the previous example, we've seen that the settings couldn't be interactively
adjusted in the configuration window. To enable this feature, a set of specific
methods has to be used instead of managing the settings ourselves. These
methods are :

- :meth:`~crappy.camera.Camera.add_bool_setting`, that allows to add a setting
  taking a boolean value (:obj:`True` or :obj:`False`). It is accessible in the
  configuration window as a checkbox that can be checked and unchecked.
- :meth:`~crappy.camera.Camera.add_choice_setting`, for adding a setting that
  takes one :obj:`str` value out of a given set of possible values. It is
  accessible in the configuration window as a menu in which you choose one out
  of several possible values.
- :meth:`~crappy.camera.Camera.add_scale_setting`, that adds a setting taking
  an :obj:`int` or :obj:`float` value within given limits. It is accessible in
  the configuration window as a horizontal slider that the user can adjust.

There are actually more methods available, but they are covered in :ref:`a
dedicated section <4. More about custom Cameras>` on the next page. By calling
any of the presented methods, you'll add a
:class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` that manages
automatically the integration of your setting in the configuration window. It
also ensures that any value you would try to set is valid, and manages the
communication with hardware provided that you indicate a getter and a setter
method as arguments. Otherwise, the value of the setting is simply stored
internally like any other attribute. Every setting can be accessed by calling
:py:`self.name`, with :py:`name` the name of the setting in plain text, or
:py:`getattr(name, self)` with the name as a :obj:`str` if the name contains
spaces. Let's now modify the first example to include a better setting
management :

.. literalinclude:: /downloads/custom_objects/custom_camera.py
   :language: python
   :emphasize-lines: 13, 15-42, 69-71, 73-75
   :lines: 1-12, 21-97

After the changes, notice that the :meth:`~crappy.camera.Camera.get_image`
method remains unchanged. The values of the settings, that were previously
defined as attributes in the :meth:`~crappy.camera.Camera.__init__` method, are
still accessed the same way, because the same names were given when adding the
settings. An :meth:`~crappy.camera.Camera.open` method is now defined, in which
the settings are instantiated and where their initial value can be provided as
arguments. What happens is that :meth:`~crappy.camera.Camera.set_all` will call
the setter of each setting, effectively setting it on the device with the
indicated value. If :meth:`~crappy.camera.Camera.set_all` is not called, the
setter is never called and there is no interaction with the hardware until you
modify a setting in the configuration window.

Here, there is no actual hardware to drive so there is no need for getters and
setters. However, an example is still provided for the :py:`high` setting to
show you how it works. The getter and the setter are usually methods of the
class, in which you communicate with the camera. When changing the value of the
setting, the setter will first be called, followed by the getter to check if
the setting was set to the correct value. It is possible to only provide a
setter, or only a getter.

You can :download:`download this custom Camera example
</downloads/custom_objects/custom_camera.py>` to run it locally on your
machine. Cameras are quite complex objects, so **there's much more to**
**discover by reading the documentation of the** :class:`~crappy.camera.Camera`
in the API. You can also have a look at the `Cameras distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/src/crappy/inout>`_ to see
how they are implemented.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Camera, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.camera.Camera.log` method provided by the parent
   :class:`~crappy.camera.Camera` class. This way, the log messages are
   included in the log file and handled in a nicer way by Crappy.

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
