===========================================
Creating and using custom objects in Crappy
===========================================

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

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

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

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

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

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

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

As you can see, compared to the template, several features have been added.
First, the Modifier takes one argument at instantiation, that indicates the
name of the label to integrate over time. This label is indeed provided in the
line where the Modifier is given as an argument to the Link. And then, several
attributes are defined in the :py:`__init__` method to handle the calculation
of the integral during :py:`__call__`. This ability to store values between
consecutive calls is exactly what was desired when using classes as Modifiers !
The two examples presented in this section can finally be merged into a single
big one :

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_modifier.py
      :language: python

|

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

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

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

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_actuator.py
      :language: python

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

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

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

Creating custom :ref:`In / Out` objects is extremely similar to creating custom
:ref:`Actuators`, so make sure to first read and understand the previous
section first ! Just like for Actuators, **anyone who wants to drive their**
**own setups with Crappy will surely need at some point to create their own**
:class:`~crappy.inout.InOut` **objects**. This section covers the specificities
of creating new InOuts.

3.a. Regular mode
+++++++++++++++++

First, let's cover the similarities with the creation of Actuator objects, in
the case of a regular usage. The case of the *streamer* mode is covered in
:ref:`the next sub-section <3.b. Streamer mode>`. Just like for an Actuator,
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
keep it very basic, you can browse the `collection of InOuts distributed with
Crappy <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/src/
crappy/inout>`_ to have an overview of what a real-life InOut looks like.

.. literalinclude:: /downloads/custom_objects/custom_inout.py
   :language: python
   :emphasize-lines: 7-22
   :lines: 1-23

In this example, the InOut simply stores two values. When :py:`get_data` is
called, it simply returns these two values as well as a timestamp. When,
:py:`set_cmd` is called, it expects two arguments and sets their values as the
new stored values. Let's now integrate the InOut into a runnable code :

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_inout.py
      :language: python

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

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

3.b. Streamer mode
++++++++++++++++++

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

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_inout_streamer.py
      :language: python
      :emphasize-lines: 5, 25-35, 52, 54-55, 61-62

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

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

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

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

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_camera.py
      :language: python
      :emphasize-lines: 8-41
      :lines: 1-17, 52-76, 86-97

|

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

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_camera.py
      :language: python
      :emphasize-lines: 13, 15-42, 69-71, 73-75
      :lines: 1-12, 21-97

|

.. Note::
   To run this example, you'll need to have the *opencv-python*,
   :mod:`matplotlib` and *Pillow* Python modules installed.

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

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

For the last section of this tutorial page, we are going to **cover the most**
**difficult but also most interesting and powerful object that you can**
**customize in Crappy : the** :ref:`Block`. Unlike the other objects introduced
on this page, Blocks are much more complex and as a user you are only supposed
to tune a very small part of it for your application. The rest of the code
should remain untouched, as it is the one that allows Crappy to run smoothly.
If you are able to define your own :class:`~crappy.blocks.Block`, **you**
**should be able to highly customize your scripts in Crappy and to drive**
**almost any experimental setup** ! Remember that the Blocks are usually not
meant to directly interact with hardware, the helper classes like the
:ref:`Actuators` and the :ref:`Cameras` are here for that. Instead, Blocks
usually create data, perform processing on existing data, interact with the
system, display data, etc.

5.a. Methods of the Block
+++++++++++++++++++++++++

First, all the Blocks must be children of the base :class:`crappy.blocks.Block`
parent class. Let's now see what methods you can define when instantiating your
own Blocks :

.. code-block:: python

   import crappy
  
   class MyBlock(crappy.blocks.Block):
  
       def __init__(self):
           super().__init__()

       def prepare(self):
           ...

       def begin(self):
           ...
  
       def loop(self):
           ...

       def finish(self):
           ...

There are actually not that many methods for you to fill in, and almost all of
them are optional ! For each method, here's what it does and how it should be
used :

- :meth:`~crappy.blocks.Block.__init__` should be used for initializing the
  Python objects that will be used in your Block. Avoid doing too much in this
  method, as there is no mechanism for properly de-initializing what you do
  there in case Crappy crashes very early. This method is also where your
  Block accepts arguments.
- :meth:`~crappy.blocks.Block.prepare` is where you should perform the
  initialization steps necessary for your Block to run. That can include
  starting a :obj:`~threading.Thread`, creating a file, populating an object,
  connecting to a website, etc. The actions performed here will be properly
  de-initialized by the :meth:`~crappy.blocks.Block.finish` method in case
  Crappy crashes. It is fine not to define this method if no particular setup
  action is required.
- :meth:`~crappy.blocks.Block.begin` is equivalent to to the first call of
  :meth:`~crappy.blocks.Block.loop`. It is the moment where the Block starts
  being allowed to send and receive data to/from other Blocks, and performs its
  main task. For the very first loop, you might want to do something special,
  like sending a trigger to another application. If so, you should use this
  method. Otherwise, this method doesn't need to be defined.
- :meth:`~crappy.blocks.Block.loop` is a method that will be called repeatedly
  during the execution of the script. It is where your Block performs its main
  task, and can send and receive data to/from other Blocks. This method does
  not take any argument, and also doesn't return anything.
- :meth:`~crappy.blocks.Block.finish` should perform the de-initialization
  steps necessary to properly stop your Block before the script ends. This
  method *should* always be called, even in case something goes wrong in your
  script. It is fine not to define this method if no particular action is
  required in your Block before exiting.

.. Important::
  Avoid including any call or structure that would prevent a method of your
  Block from returning ! For example, avoid using blocking calls without a
  short timeout (at most a few seconds), and do not use infinite loops that
  could never end. That is because in the smooth termination scenarios, the
  Blocks are only told to terminate once their current method call returns.
  Otherwise, you'll have to use :kbd:`Control-c` to stop your script, which is
  now considered an invalid way to stop Crappy.

Now that the possible methods have been described, it is time to put them into
application in an example. However, as the Block object is quite complex, such
an example needs to include aspects described in the next sub-sections. So,
instead of building and improving an example iteratively over the sub-sections,
we'll simply comment the relevant parts of one complete example in each
sub-section.

For this example, we have created a fully functional Block that can send and/or
receive data to/from network sockets. It can be useful for communicating with
remote devices over a network, although the :ref:`Client Server` Block already
provides this functionality using MQTT. The demo Block is really not advanced
enough to be distributed with Crappy, but it will do just fine for this
tutorial ! Here is the full code :

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_block.py
      :language: python
      :emphasize-lines: 13-22, 46, 97, 127

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

In this Block, only the :meth:`~crappy.blocks.Block.begin` method is not
defined. That is not a big deal, most Blocks do not need to define this method,
especially for beginners. Overall, the Block can send the value of a given
input label to a given output network address along with a timestamp. It can
also receive a value and a timestamp from a given input network address and
send it to downstream Blocks over a given output label. It can thus basically
receive and/or send data over the network. Let's review its methods one by
one :

- :meth:`~crappy.blocks.Block.__init__` only sets attributes, and accepts
  arguments. It also instantiates two sockets, which is fine since
  instantiation alone does not actually trigger any connection to anything. In
  your own Blocks, you can define as many arguments as you want to provide the
  desired level of granularity, but this comes of course at the cost of
  complexity. You can see that some attributes have a leading underscore in
  their name, this is discussed in the
  :ref:`next sub-section <5.b. Useful attributes of the Block>`.
- In :meth:`~crappy.blocks.Block.prepare`, quite a lot of initialization is
  performed. There are two parts in the implementation : one executed if the
  Block has input Links, the other if it has output Links. If there are input
  Links, the Block tries to connect to the provided port at the provided
  address. If there are output Links, the Block waits for an external
  connection on the desired address and port, and accepts one connection. If
  any of these operations fail, an exception is raised and the Block stops.
- In :meth:`~crappy.blocks.Block.loop`, incoming data is first received. Then,
  if the data contains all the necessary information, the timestamp and the
  value are cast to bytes and sent over the network. If there are output Links,
  the Block then checks if data is ready to be read from the network. If so, it
  unpacks the timestamp and the value and sends them to downstream Blocks.
- :meth:`~crappy.blocks.Block.finish` simply closes all the opened network
  sockets, in order to free the associated resources.

You'll need to have a closer look at the code if you want to understand every
single line, but you should already have a rough idea of how it works. More
details about the methods and attributes that are used are given in the next
sub-sections. You can :download:`download this custom Block example
</downloads/custom_objects/custom_block.py>` to run it locally on your machine.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Block, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.blocks.Block.log` method provided by the parent
   :class:`~crappy.blocks.Block` class. This way, the log messages are
   included in the log file and handled in a nicer way by Crappy.

5.b. Useful attributes of the Block
+++++++++++++++++++++++++++++++++++

While writing your own Blocks, you are free to use whatever names you want for
the attributes you define. Any name, really ? Actually, **a few attribute**
**names are already used by the parent** :class:`~crappy.blocks.Block`
**class, and provide some very useful functionalities**. This sub-section lists
them, as well as their meaning and effect when applicable. In the general case,
none of the attributes presented here is mandatory to use. Nothing bad can
happen if you choose not to use them, what happens if you override them is a
different story !

.. Note::
   When defining your own attributes, you can put a leading underscore in their
   names to indicate that an attribute is for internal use only and should not
   be accessed or modified by any external user or program.

Here is the exhaustive list of all the attributes you can access and their
meaning :

- :py:`outputs` is a :obj:`list` containing the reference to all the incoming
  Links. It is useful for checking whether the Block has input Links or not. It
  should not be modified !
- :py:`inputs` is a :obj:`list` containing the reference to all the outgoing
  Links. It is useful for checking whether the Block has output Links or not.
  It should not be modified ! It is sometimes used to put a limit on the number
  of incoming Links (for example the :class:`~crappy.blocks.Recorder` Block
  raises an error if it has more than one incoming Link).
- :py:`niceness` can be set during :meth:`~crappy.blocks.Block.__init__`, and
  the corresponding niceness value will be set for the Process by
  :meth:`~crappy.blocks.Block.renice_all`. It is only relevant on Linux, and
  barely used. Most users can ignore it.
- :py:`freq` sets the target looping frequency for the Block. It can be set to
  any positive value, or to :obj:`None` to switch to free-run mode. If a value
  is given, the Block will *try* to reach it but this is not guaranteed. It
  can be set anytime, but is usually set during
  :meth:`~crappy.blocks.Block.__init__`. Depending on the application, a
  reasonable value for this attribute is usually somewhere between 20 and 200.
- :py:`display_freq` is a :obj:`bool` that enables the display of the achieved
  looping frequency of the Block. If set to :obj:`True`, the looping frequency
  is displayed in the terminal every two seconds. It can be set anytime, but is
  usually set during :meth:`~crappy.blocks.Block.__init__`.
- :py:`debug` can be either :obj:`True`, :obj:`False`, or :obj:`None`. If set
  to :obj:`False` (the default), it only displays a limited amount of
  information in the terminal. If set to :obj:`True`, additional debug
  information is displayed for this Block. When the debug mode is enabled,
  there is usually way too much information displayed to follow ! The extra
  information is useful for debugging, for skilled enough users. The last
  option is to set :py:`debug` to :obj:`None`, in which case no information is
  displayed at all for the Block. That is not advised in the general case. This
  attribute must be set during :meth:`~crappy.blocks.Block.__init__`.
- :py:`labels` contains the names of the labels to send to downstream Blocks.
  When given, the values to send can be given as a :obj:`tuple` (for example),
  rather than as a :obj:`dict` containing both the names of the labels and the
  values. More about it in :ref:`the next section
  <5.c. Sending data to other Blocks>`. This attribute can be set at any moment.
- :py:`t0` contains the timestamp of the exact moment when all the Blocks start
  looping together. It is useful for obtaining the timestamp of the current
  moment relative to the beginning of the test. This attribute can only be
  read starting from :meth:`~crappy.blocks.Block.begin`, and must not be
  modified !
- :py:`name` contains the unique name attributed to the Block by Crappy. It can
  be read at any time, and even modified. This name is only used for logging,
  and appears in the log messages for identifying where a message comes from.

In the presented example, you may have recognized a few of the presented
attributes. They are highlighted here for convenience :

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_block.py
      :language: python
      :emphasize-lines: 27-29, 49, 74
      :lines: 1-96

|

There is not much more to say about the available attributes of the Block that
you can use, you'll see for yourself which ones you need and which ones you
don't when developing !

5.c. Sending data to other Blocks
+++++++++++++++++++++++++++++++++

A very important aspects of Blocks is how they communicate with each other. You
normally already know that for two Blocks to exchange data, they must be linked
by a :class:`~crappy.links.Link`. But when writing your own Blocks, how to tell
Crappy what to send exactly ? That is the topic of this sub-section !

**Sending data to downstream Blocks in Crappy is extremely simple, because**
**there is only one way to achieve it : you have to use the**
:meth:`~crappy.blocks.Block.send` **method**. This method accepts only one
argument, either a :obj:`dict` or an :obj:`~collections.abc.Iterable` (like a
:obj:`list` or a :obj:`tuple`) of values to send (usually the values are
:obj:`float` or :obj:`str`). If a dictionary is given, its keys are the names
of the labels to send. For each label, a single value must be provided, and the
same labels should be sent throughout a given test. If the values are given in
an Iterable without labels, then the :py:`labels` attribute of the Block must
have been set beforehand. The dictionary to send will be reconstructed from the
labels and the given values. There should, of course, be as many given values
as there are labels. And that's basically all there is to know about sending
data to downstream Blocks in Crappy !

.. Note::
   The dictionary sent through the Links are exactly the same that the
   :ref:`Modifiers` can access and modify. See the :ref:`dedicated section
   <1. Custom Modifiers>` for more information.

The line in the example where the data gets sent is outlined below :

.. literalinclude:: /downloads/custom_objects/custom_block.py
   :language: python
   :emphasize-lines: 29
   :lines: 97-126

As you can see, it was chosen to send a dictionary here, but a solution using
the :py:`labels` attribute would also have worked. The dictionary is built in
a quite elegant way, using the :obj:`zip` method. You can see that the labels
to send are :py:`'t(s)'` fo the time, and the chosen output label for the
transferred value. The values to send are given by the :obj:`~struct.unpack`
function, that returns two :obj:`float` from binary data.

5.d. Receiving data from other Blocks
+++++++++++++++++++++++++++++++++++++

Now that the method for sending data has been covered, it is time to describe
the complementary methods that allow a Block to receive data from upstream
Blocks. Things get a bit more complex at that point, because **there are no**
**less than four possible methods that you can use** ! Each of them serves a
different purpose, let's review them all in this sub-section :

- :meth:`~crappy.blocks.Block.recv_data` is by far the simplest method. It
  creates an empty :obj:`dict`, that it updates with **one** message (i.e. one
  sent :obj:`dict`) from each of the incoming :ref:`Links`, and then returns
  it. This means that some data might be lost if several Links carry a same
  label, which is very often the case with the time label ! Also, only the
  first available message of each Link is read, meaning that if there are
  several incoming messages in the queue, only one is queued out. For this
  reason, this method is barely used, but it is still implemented for whoever
  would find an application to it !
- :meth:`~crappy.blocks.Block.recv_last_data` is based on the same principle as
  the previous method, except it includes a loop that updates the dictionary
  to return with **all** the queued messages. In the end, only the latest
  received value for each label is present in the returned dictionary, hence
  the name of the method. A :py:`fill_missing` argument allows to control
  whether the last known value of each label is included if no newer value is
  available, thus returning the latest known value of **all** the known labels
  (not just the ones whose values were recently received). Just like the
  previous method, this one doesn't keep the integrity of the time information
  if there are several incoming Links, and only returns one value per label
  even if several messages were received.
- :meth:`~crappy.blocks.Block.recv_all_data` allows to keep and return
  multiple values for each label, if several messages were received from
  upstream Links. To do so, it returns a :obj:`dict` whose keys are the
  received labels, but whose values are :obj:`list` containing for each label
  all the successive values that were received. This way, the history of each
  label is preserved, which is crucial for certain applications (integration
  for example). However, just like the previous ones, this method isn't safe in
  case several Links carry a same label. Therefore, it also doesn't preserve
  the time information. Note that this method possesses two arguments for
  acquiring data continuously over a given delay, but you'll need to check the
  API for more information about them.
- :meth:`~crappy.blocks.Block.recv_all_data_raw` is by far the most complex way
  of receiving data ! This method returns **everything** that flows into each
  single incoming Link, with no possible loss ! However, the returned object is
  of course much more complex. Basically, it is equivalent to a call to
  :meth:`~crappy.blocks.Block.recv_all_data` on each Link taken separately. All
  the results are then put together in one list, so this method returns a
  :obj:`list` of :obj:`dict` (one per Link) whose keys are :obj:`str` (labels)
  and values are :obj:`list` of all the received data for the given label.
  **Using this method is mandatory when you have to retrieve all the exact**
  **timestamps for several labels that can come from different Links**. You can
  check the :class:`~crappy.blocks.Grapher` or the
  :class:`~crappy.blocks.Multiplexer` Blocks for examples of usage.

There is quite much choice, but choosing the right method for your Block is
really not that difficult. Put aside :meth:`~crappy.blocks.Block.recv_data`
that is almost never used, you can decide for the right method in one or two
steps. Can you work with only the latest value of each label ? If so, go for
:meth:`~crappy.blocks.Block.recv_last_data`. Otherwise, will you use the
history of the time label **and** does your Block accept multiple incoming
Links ? If so, no other choice than using
:meth:`~crappy.blocks.Block.recv_all_data_raw`. Else,
:meth:`~crappy.blocks.Block.recv_all_data` will do just fine !

.. Note::
   An additional :meth:`~crappy.blocks.Block.data_available` method allows
   checking for the availability of new data in the incoming Links. It returns
   a :obj:`bool` indicating whether new data is available or not. It can be
   useful to avoid useless calls to *recv* methods.

.. literalinclude:: /downloads/custom_objects/custom_block.py
   :language: python
   :emphasize-lines: 4
   :lines: 97-126

In the custom Block example, you can see that we opted for the
:meth:`~crappy.blocks.Block.recv_last_data` method. The handling of the
returned data is then fairly simple, as a single :obj:`dict` with single values
is returned. This dictionary must still be checked before processing it, as it
might be empty or not contain all the necessary data at each loop !

Finally, that's it ! If you have been through this entire page of tutorials
and the previous ones, **you should now be ready to use Crappy at a**
**reasonably high level to drive all kinds of experimental setups**. We know
that the module is far from being simple, and that there are many things to
keep in mind when using it. This is why we're trying to keep the documentation
as extensive as possible, and we provide a wide variety of `ready-to-run
examples  <https://github.com/LaboratoireMecaniqueLille/crappy/examples>`_. At
that point of the tutorials, there are still a few uncovered topics only
relevant to advanced users. You can check them on the :ref:`next and last page
of the tutorials <More about custom objects in Crappy>`.
