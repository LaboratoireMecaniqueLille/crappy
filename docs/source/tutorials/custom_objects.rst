===========================================
Creating and using custom objects in Crappy
===========================================

.. role:: py(code)
  :language: python
  :class: highlight

This page explains how to create Modifiers, Actuators, InOuts, Camera objects,
and Blocks for functionality that the built-in objects do not provide. It
covers the basic lifecycle and registration requirements. The :ref:`next page
of the
tutorials <tutorials/complex_custom_objects:more about custom objects in crappy>` covers the advanced aspects of
custom object instantiation.

Use :doc:`../concepts/choosing_custom_object_type` if you are unsure which base
class fits a task. The :doc:`../concepts/lifecycle_shutdown` page defines the
shared Block lifecycle referenced by the examples below.

1. Custom Modifiers
-------------------

Custom :ref:`Modifiers <crappy_docs/modifiers:modifiers>` implement focused
transformations that the built-in catalog does not provide. A Modifier must be
a :obj:`~collections.abc.Callable`, so it can be a function or a class.

More precisely, a Modifier should accept a :obj:`dict` as its sole argument and
return a :obj:`dict` as well (:obj:`None` is also accepted). The dictionary is
a chunk of data flowing through the :ref:`Link <crappy_docs/links:link>`. The Modifier can add keys,
delete others, change the value of a key, etc. Each key is a label, and has a
value it carries. In the end, all a Modifier does is to modify the incoming
dictionary and return it after modification. The following runnable example
demonstrates this operation.

For this first example, let's create a Modifier that simply doubles the value
of a given label. If you have understood the last paragraph, this Modifier will
basically only perform something like :py:`data['label'] = data['label'] * 2`.
Define the
function somewhere, preferably before the :py:`if __name__ == "__main__"`
statement. Then, pass it to the :py:`'modifier'` argument of the target
:class:`~crappy.links.Link`. The function must return the modified dictionary.

.. literalinclude:: /downloads/custom_objects/custom_modifier.py
   :language: python
   :emphasize-lines: 6-9, 24
   :lines: 1-3, 5-12, 32-41, 43-45, 47-48

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

In this first example, you can see that instead of replacing the value of the
:py:`'cmd'` label with its double, it was chosen to store the double value in
the newly created :py:`'cmdx2'` label. The Modifier is added to the Link between
the :ref:`Generator <crappy_docs/blocks:generator>` and the
:ref:`Grapher <crappy_docs/blocks:grapher>`. To change the target label or
the value of the multiplier, modify them in the definition of
the function. Alternatively, you could add arguments to you function and use
:obj:`functools.partial` when passing it to the Link, but that is quite an
advanced design already.

In the example, a basic function was passed as a Modifier. While functions are
concise, but an ordinary function cannot retain state between calls without an
enclosing closure or another external store. The
:class:`~crappy.modifier.Integrate` Modifier, for example, stores an integral
between consecutive chunks of data. A Modifier class expresses reusable
stateful behavior more clearly.

To circumvent this limitation, instantiate Modifiers as classes instead of
functions. It is mentioned above that the Modifiers
need to be :obj:`~collections.abc.Callable` objects, and a class defining the
:py:`__call__` method is callable. Here is the minimal template of a
Modifier as a class:

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
makes it possible to store attributes between calls. The next example
implements a reduced version of the
Integrate Modifier:

.. literalinclude:: /downloads/custom_objects/custom_modifier.py
   :language: python
   :emphasize-lines: 7-23, 37
   :lines: 1-6, 13-39, 42-44, 46-48

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

Compared with the template, the Modifier takes one argument at instantiation
that indicates the
name of the label to integrate over time. This label is indeed provided in the
line where the Modifier is given as an argument to the Link. And then, several
attributes are defined in the :py:`__init__` method to handle the calculation
of the integral during :py:`__call__`. This ability to store values between
consecutive calls is the stateful behavior provided by the class.
The two examples presented in this section can finally be merged into a single
big one:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_modifier.py
      :language: python

|

You can :download:`download this custom Modifier example
</downloads/custom_objects/custom_modifier.py>` to run it locally on your
machine. An extra example of a custom Modifier can also be found in the
`examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/tree/master/examples/custom_objects>`__. These examples cover the
required interface for custom Modifiers.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Modifier, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.modifier.Modifier.log` method provided by the parent
   :class:`~crappy.modifier.Modifier` class. This way, the log messages are
   included in the log file and handled by Crappy's centralized logging.

2. Custom Actuators
-------------------

After introducing how custom Modifiers work in the first section, this second
section will focus on the use of custom :ref:`Actuators <crappy_docs/actuators:actuators>`. Knowing how to add
and use your own :class:`~crappy.actuator.Actuator` objects in Crappy is
necessary when the built-in integrations do not support the equipment in a
test.

Unlike Modifiers, Actuators usually communicate with hardware. The code for
driving a custom Actuator is therefore different from the code for a
Modifier that handles data. First, verify that Python can control the hardware.
Crappy cannot integrate a device that is inaccessible from Python. A dedicated
Python library is not always required. The available approaches include:

- Use a Python library provided by the manufacturer or a third party
- Get the correct communication syntax and protocol from the datasheet, and
  code the communication yourself over the right protocol (serial, USB, I2C,
  SPI, etc.)
- Send commands in the terminal from Python, if the manufacturer provides a way
  to drive hardware from the console
- Write Python bindings for a C/C++ library, if the manufacturer provides one

These approaches require different levels of Python and protocol expertise.
Before integrating the device with Crappy, write an independent program that
initializes the connection, configures the device, sends supported commands,
acquires its state, and reliably closes the device and connection. This
separates device communication problems from framework integration problems.

After validating the independent program, integrate it with Crappy. As with
the Modifier previously, there is also a template for the
:class:`~crappy.actuator.Actuator` objects:

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
the interest is limited. Let's review what each method is intended for:

- :meth:`~crappy.actuator.Actuator.__init__` is where you should initialize the
  Python objects that your class uses. It is also where the class accepts its
  arguments, that are given in the dictionary passed to the :ref:`Machine <crappy_docs/blocks:machine>`
  Block. Avoid interacting with hardware already in this method. Also, don't
  initialize the parent class with :py:`super().__init__()`.
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
  receives commands via an incoming :ref:`Link <crappy_docs/links:link>`. Note that the
  :meth:`~crappy.actuator.Actuator.set_position` method always accepts a second
  :py:`speed` argument, that may be equal to :obj:`None`. You'll find more
  about it in :ref:`a dedicated section on the next page
  <tutorials/complex_custom_objects:3. more about custom actuators>`.
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
  required for properly deinitializing the device. For example, this is where
  you put a device to sleep mode or close the connection to it.

The class must inherit from :class:`crappy.actuator.Actuator`. Transfer the
corresponding operations from the independent program into the template's
lifecycle methods. The following example is based on the custom Actuator in
the `examples folder on GitHub <https://github.com/LaboratoireMecaniqueLille/
crappy/tree/master/examples/custom_objects>`__:

.. literalinclude:: /downloads/custom_objects/custom_actuator.py
   :language: python
   :emphasize-lines: 7-36
   :lines: 1-37

The :py:`__init__` method takes one argument and initializes objects used
elsewhere in the class. The :py:`open` method has no hardware-specific work
because this example emulates hardware and does not interact with any
real-world device. For the same reason, the :py:`close` and :py:`stop` methods
are missing. This Actuator can only be driven in speed, so the
:py:`set_position` method is also missing. The :py:`set_speed` and
:py:`get_speed` methods are present for setting the target speed and measuring
the current one, as well as the :py:`get_position` method since the position
is also measurable. Now that the Actuator is defined, it is time to add some
context to make it run:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_actuator.py
      :language: python

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

You can :download:`download this custom Actuator example
</downloads/custom_objects/custom_actuator.py>` to run it locally on your
machine. The following sections apply the same lifecycle pattern to other
custom objects. The `Actuators distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/src/crappy/
actuator>`_ to see how the implementation of real-life Actuators looks like.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Actuator, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.actuator.Actuator.log` method provided by the parent
   :class:`~crappy.actuator.Actuator` class. This way, the log messages are
   included in the log file and handled by Crappy's centralized logging.

3. Custom InOuts
----------------

Creating custom :ref:`In / Out <crappy_docs/inouts:in / out>` objects follows
the same lifecycle pattern as custom :ref:`Actuators <crappy_docs/actuators:actuators>`. Read the previous
section first. This section describes the InOut-specific methods.

3.a. Regular mode
+++++++++++++++++

First, let's cover the similarities with the creation of Actuator objects, in
the case of a regular usage. The case of the *streamer* mode is covered in
:ref:`the next sub-section <tutorials/custom_objects:3.b. streamer mode>`. Just like for an Actuator,
you'll need to write your class before the :py:`if __name__ == "__main__"`
statement, or to import it from another file. You should also start from a
working draft in which you're able to drive your device in Python. And in both
cases, create the custom class by filling in a template. The InOut template is:

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

There are two main differences. First, the parent class from
which your InOut must inherit is now :class:`crappy.inout.InOut`, and second
you now have to define the :py:`get_data` and/or :py:`set_cmd` methods. The
:meth:`~crappy.inout.InOut.__init__`, :meth:`~crappy.inout.InOut.open` and
:meth:`~crappy.inout.InOut.close` methods serve the same purpose as for the
Actuators. The new methods are:

- :meth:`~crappy.inout.InOut.get_data`, that takes no argument and should
  return the data acquired by the device. The first returned value must be the
  timestamp of the acquisition, as returned by :obj:`time.time`. Then, you can
  return as many values as you want, usually corresponding to different
  channels you device can acquire. The number of returned values should always
  be the same, and for each value a label should be given in the :py:`labels`
  argument of the :ref:`IOBlock <crappy_docs/blocks:ioblock>`. The data will only be acquired if the IOBlock
  has outgoing :ref:`Links <crappy_docs/links:links>`.
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

In this example, the InOut stores two values. When :py:`get_data` is called,
it returns these two values and a timestamp. When
:py:`set_cmd` is called, it expects two arguments and sets their values as the
new stored values. Let's now integrate the InOut into a runnable code:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_inout.py
      :language: python

|

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` Python module
   installed.

In order to obtain two commands from a single :ref:`Generator <crappy_docs/blocks:generator>`, a
:ref:`Modifier <crappy_docs/modifiers:modifier>` is added to create a new label. In the IOBlock, the two labels
carrying the commands are indicated in the :py:`cmd_labels` argument. The
values acquired by the :py:`get_data` method are transmitted to the
:ref:`Grapher <crappy_docs/blocks:grapher>` Block over the labels indicated in the :py:`labels` argument of
the IOBlock. You can
:download:`download this custom InOut example
</downloads/custom_objects/custom_inout.py>` to run it locally on your
machine, and have a look at the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
custom_objects>`_ to find more examples of custom InOut objects.

.. Note::
   If you want to have debug information displayed in the terminal from your
   InOut, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.inout.InOut.log` method provided by the parent
   :class:`~crappy.inout.InOut` class. This way, the log messages are
   included in the log file and handled by Crappy's centralized logging.

3.b. Streamer mode
++++++++++++++++++

If you want to be able to use your custom InOut object in *streamer* mode, the
methods described above are not sufficient. Follow the streaming lifecycle
described in this subsection. For
more details on how to use the *streamer* mode, refer to the
:ref:`streaming-acquisition tutorial <tutorial-streaming-acquisition>`.
Getting straight
to the point, here's how the template for an InOut supporting the *streamer*
mode looks like:

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

It is the same as the regular InOut template, with three additional methods.
You can still define :meth:`~crappy.inout.InOut.get_data` and
:meth:`~crappy.inout.InOut.set_cmd` so that the InOut supports both regular and
streamer modes, selected through the :py:`streamer` argument of the
:ref:`IOBlock <crappy_docs/blocks:ioblock>`. The streaming methods have these responsibilities:

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
  method must be specific to the *streamer* mode. The general deinitialization
  commands should still be executed in the :meth:`~crappy.inout.InOut.close`
  method.

The following example extends the previous InOut and demonstrates the arrays
returned by :meth:`~crappy.inout.InOut.get_stream`:

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
Finally, a :ref:`Demux <crappy_docs/modifiers:demux>` Modifier is now needed on the :ref:`Link <crappy_docs/links:link>` from the
IOBlock to the Grapher in order for the data to be displayed.

You can :download:`download this custom streamer InOut example
</downloads/custom_objects/custom_inout_streamer.py>` to run it locally on your
machine. The only real difficulty with the instantiation of custom InOuts
supporting the *streamer* mode is building the arrays to return, but you can
find an additional example of a custom InOut in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
custom_objects>`__ and in the `InOuts distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/src/crappy/
inout>`_.

4. Custom Cameras
-----------------

Custom :class:`crappy.camera.Camera` objects integrate unsupported camera
hardware with Crappy's acquisition pipelines. Acquisition, display, recording,
and processing can run independently. Built-in processing Blocks can consume
images from a custom Camera object. See :doc:`../concepts/image_pipelines` for
the two supported pipeline architectures, and see the
:ref:`DIS Correl <crappy_docs/blocks:dis correl>`,
:ref:`DIC VE <crappy_docs/blocks:dic ve>`, and
:ref:`Video Extenso <crappy_docs/blocks:video extenso>` Blocks for examples.

The first step for integrating a camera in Crappy is to check whether it can be
read by one of the existing :ref:`Cameras <crappy_docs/cameras:cameras>`. The :ref:`Camera OpenCV <crappy_docs/cameras:camera opencv>` and
:ref:`Camera GStreamer <crappy_docs/cameras:camera gstreamer>` objects in particular are designed to be compatible
with a wide range of cameras, using the :mod:`opencv-python` and GStreamer
modules respectively. Test them with the hardware first. If neither is
compatible, write a custom
:class:`~crappy.camera.Camera` object.

Just like in the previous sections, there is a template for the Camera
objects:

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
the :meth:`~crappy.camera.Camera.open` method. Camera settings can be adjusted
interactively in the
:ref:`Camera Configurator <crappy_docs/tools:camera configurator>` interface. Because the settings are handled in a
special way and applied during :meth:`~crappy.camera.Camera.open`, their values
are accepted by this method.

The method unique to the Camera objects is
:meth:`~crappy.camera.Camera.get_image`, that should acquire one image at a
time, normally by communicating with the hardware. This method does not accept
any argument, and should return two values. The first one is the timestamp at
which the image was acquired, as returned by :obj:`time.time`. The second one
is the acquired image, as a :mod:`numpy` array or numpy-compatible object. The
image can be a two-dimensional array for a grayscale image, or a
dimension three if it is a color image. It can also be encoded over 8 or 16
bits indifferently.

The following basic implementation is based on an `example available on
GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
examples/custom_objects>`_:

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
will be introduced in the next paragraphs. The
:meth:`~crappy.camera.Camera.get_image` method returns the generated image and
its timestamp. Here, the :meth:`~crappy.camera.Camera.open` and
:meth:`~crappy.camera.Camera.close` methods don't need to be defined as there
is no interactive setting defined nor any hardware to (de-)initialize.

In the previous example, we've seen that the settings couldn't be interactively
adjusted in the configuration window. To enable this feature, a set of specific
methods has to be used instead of managing the settings ourselves. These
methods are:

- :meth:`~crappy.camera.Camera.add_bool_setting`, which adds a setting
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
dedicated section <tutorials/complex_custom_objects:4. more about custom cameras>` on the next page. By calling
any of the presented methods, you'll add a
:class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` that manages
automatically the integration of your setting in the configuration window. It
also ensures that any value you would try to set is valid, and manages the
communication with hardware provided that you indicate a getter and a setter
method as arguments. Otherwise, the value of the setting is stored
internally like any other attribute. Every setting can be accessed by calling
:py:`self.name`, with :py:`name` the name of the setting in plain text, or
:py:`getattr(name, self)` with the name as a :obj:`str` if the name contains
spaces. Let's now modify the first example to include a better setting
management:

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
machine. See :class:`crappy.camera.Camera` in the API for the complete
interface. You can also inspect the `Camera objects distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/src/crappy/
inout>`_ to see how they are implemented.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Camera object, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.camera.Camera.log` method provided by the parent
   :class:`~crappy.camera.Camera` class. This way, the log messages are
   included in the log file and handled by Crappy's centralized logging.

5. Custom Blocks
----------------

The :ref:`Block <crappy_docs/blocks:block>` is the most general base class
covered on this page. A custom Block overrides a small set of lifecycle methods
from :class:`crappy.blocks.Block`. Blocks are usually not
meant to directly interact with hardware, the helper classes like the
:ref:`Actuators <crappy_docs/actuators:actuators>` and the :ref:`Cameras <crappy_docs/cameras:cameras>` are here for that. Instead, Blocks
usually create data, perform processing on existing data, interact with the
system, display data, etc.

.. Note::
   A custom Block that sends or receives images should normally inherit from
   :class:`~crappy.blocks.vision.VisionBlock`, rather than setting
   :attr:`~crappy.blocks.Block.is_vision_block` itself. The
   :ref:`custom VisionBlock tutorial <tutorials/complex_custom_objects:5. custom visionblocks>` describes the
   shared-memory lifecycle and image methods, and points to a complete runnable
   example.

5.a. Methods of the Block
+++++++++++++++++++++++++

All custom Blocks must inherit from :class:`crappy.blocks.Block`. A custom
Block can define these methods:

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

Most of these methods are optional:

- :meth:`~crappy.blocks.Block.__init__` should be used for initializing the
  Python objects that will be used in your Block. Avoid doing too much in this
  method, as there is no mechanism for properly deinitializing what you do
  there in case Crappy crashes very early. This method is also where your
  Block accepts arguments.
- :meth:`~crappy.blocks.Block.prepare` is where you should perform the
  initialization steps necessary for your Block to run. That can include
  starting a :obj:`~threading.Thread`, creating a file, populating an object,
  connecting to a website, etc. The actions performed here will be properly
  deinitialized by the :meth:`~crappy.blocks.Block.finish` method if
  Crappy crashes. It is fine not to define this method if no particular setup
  action is required.
- :meth:`~crappy.blocks.Block.begin` is equivalent to the first call of
  :meth:`~crappy.blocks.Block.loop`. It is the moment where the Block starts
  being allowed to send and receive data to/from other Blocks, and performs its
  main task. For the very first loop, you might want to do something special,
  like sending a trigger to another application. If so, you should use this
  method. Otherwise, this method doesn't need to be defined.
- :meth:`~crappy.blocks.Block.loop` is a method that will be called repeatedly
  during the execution of the script. It is where your Block performs its main
  task, and can send and receive data to/from other Blocks. This method does
  not take any argument, and also doesn't return anything.
- :meth:`~crappy.blocks.Block.finish` should perform the deinitialization
  steps necessary to properly stop your Block before the script ends. This
  method *should* always be called, even in case something goes wrong in your
  script. It is fine not to define this method if no particular action is
  required in your Block before exiting.

.. Important::
  Avoid including any call or structure that would prevent a method of your
  Block from returning. For example, avoid using blocking calls without a
  short timeout (at most a few seconds), and do not use infinite loops that
  could never end. That is because in the smooth termination scenarios, the
  Blocks are only told to terminate once their current method call returns.
  Otherwise, you'll have to use :kbd:`Control-c` to stop your script, which is
  now considered an invalid way to stop Crappy.

Now that the possible methods have been described, it is time to put them into
application in an example. However, as the Block object is quite complex, such
an example needs to include aspects described in the next sub-sections. So,
instead of building and improving an example iteratively over the sub-sections,
each subsection comments the relevant part of one complete example.

For this example, we have created a fully functional Block that can send and/or
receive data to/from network sockets. It can be useful for communicating with
remote devices over a network, although the :ref:`Client Server <crappy_docs/blocks:client server>` Block already
provides this functionality using MQTT. The demo is tutorial code rather than a
built-in implementation. Here is the full code:

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
one:

- :meth:`~crappy.blocks.Block.__init__` only sets attributes, and accepts
  arguments. It also instantiates two sockets, which is fine since
  instantiation alone does not actually trigger any connection to anything. In
  your own Blocks, you can define as many arguments as you want to provide the
  desired level of granularity, but this increases complexity. Some attributes
  have a leading underscore in their name, as discussed in the
  :ref:`next sub-section <tutorials/custom_objects:5.b. useful properties and attributes of the block>`.
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
- :meth:`~crappy.blocks.Block.finish` closes all opened network
  sockets, in order to free the associated resources.

You'll need to have a closer look at the code if you want to understand every
single line, but you should already have a rough idea of how it works. More
details about the methods and attributes that are used are given in the next
sub-sections. You can :download:`download this custom Block example
</downloads/custom_objects/custom_block.py>` to run it locally on your machine.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Block, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.blocks.Block.log` method provided by the parent
   :class:`~crappy.blocks.Block` class. This way, the log messages are
   included in the log file and handled by Crappy's centralized logging.

5.b. Useful properties and attributes of the Block
+++++++++++++++++++++++++++++++++++++++++++++++++++

Custom Blocks can define their own attributes, but several names belong to the
parent :class:`~crappy.blocks.Block` class. Most execution settings are
public properties: they are accessed just like attributes, but their setters
validate assignments and may update internal variables. This sub-section
lists them, as well as their meaning and effect when applicable.

.. Note::
   When defining your own attributes, you can put a leading underscore in their
   names to indicate that an attribute is for internal use only and should not
   be accessed or modified by any external user or program. In particular, do
   not bypass the Block properties by assigning their private backing
   attributes (such as ``_freq`` or ``_labels``).

Here is the exhaustive list of the relevant properties and attributes you can
access and their meaning:

- :py:`outputs` is a :obj:`list` containing the references to all the outgoing
  Links. It is useful for checking whether the Block has output Links or not.
  Do not modify it.
- :py:`inputs` is a :obj:`list` containing the references to all the incoming
  Links. It is useful for checking whether the Block has input Links or not.
  Do not modify it. It is sometimes used to put a limit on the number
  of incoming Links (for example the :class:`~crappy.blocks.Recorder` Block
  raises an error if it has more than one incoming Link).
- :attr:`~crappy.blocks.Block.niceness` is an integer property from `-20` to
  `19`. It can be set during :meth:`~crappy.blocks.Block.__init__`, and the
  corresponding runtime priority will be requested by
  :meth:`~crappy.blocks.Block.renice_all`. It is only relevant on Linux, and
  barely used. Most users can ignore it.
- :attr:`~crappy.blocks.Block.freq` sets the target looping frequency for the
  Block. It accepts a positive integer or floating-point value, which is stored
  as a :obj:`float`, or :obj:`None` to switch to free-run mode. If a value is
  given, the Block will *try* to reach it but this is not guaranteed. It can be
  set anytime, but is usually set during
  :meth:`~crappy.blocks.Block.__init__`. Select the value according to the
  operation cost and timing requirements, then measure the achieved frequency.
- :attr:`~crappy.blocks.Block.display_freq` is a :obj:`bool` property that
  enables the display of the achieved looping frequency of the Block. If set
  to :obj:`True`, the looping frequency is displayed in the terminal every two
  seconds. It can be set anytime, but is usually set during
  :meth:`~crappy.blocks.Block.__init__`.
- :attr:`~crappy.blocks.Block.debug` can be either :obj:`True`, :obj:`False`,
  or :obj:`None`. If set to :obj:`False` (the default), it only displays a
  limited amount of information in the terminal. If set to :obj:`True`,
  additional debug information is displayed for this Block. Debug mode is
  verbose and is intended for diagnosing Block behavior. The
  last option is to set ``debug`` to :obj:`None`, in which case no information
  is displayed at all for the Block. That is not advised in the general case.
  This property must be set during :meth:`~crappy.blocks.Block.__init__`.
- :attr:`~crappy.blocks.Block.labels` contains the unique names of the labels
  to send to downstream Blocks. It accepts :obj:`None` or a sequence containing
  only strings. When given, the values to send can be given as a :obj:`tuple`
  (for example) at runtime, rather than as a :obj:`dict` containing both the
  names of the labels and the values. More about it in :ref:`the next section
  <tutorials/custom_objects:5.c. sending data to other blocks>`. This property can be set at any moment.
- :attr:`~crappy.blocks.Block.t0` is a read-only property containing the
  timestamp of the exact moment when all the Blocks start looping together. It
  is useful for obtaining the timestamp of the current moment relative to the
  beginning of the test. It can only be read starting from
  :meth:`~crappy.blocks.Block.begin`.
- :attr:`~crappy.blocks.Block.name` contains the unique, non-empty name
  attributed to the Block by Crappy. It can be read at any time. It should only
  be modified in :meth:`~crappy.blocks.Block.__init__`. The name identifies the
  Block both in log messages and in Crappy's :class:`~crappy.links.LinkGraph`.
  Renaming it before the test starts also updates all already-created Links
  connected to it. Renaming it while the Block is running raises
  :exc:`RuntimeError`.
- :attr:`~crappy.blocks.Block.pausable` is a :obj:`bool` property indicating
  whether the Block is affected when a pause is started by a
  :class:`~crappy.blocks.Pause` Block. By default, most Blocks are affected
  except for the ones managing the test flow (like the
  :class:`~crappy.blocks.StopButton` Block).
- :attr:`~crappy.blocks.Block.is_vision_block` is a :obj:`bool` property used
  when validating :class:`~crappy.links.ImageLink`. It is normally set by
  image-oriented Block base classes and should rarely need to be changed in a
  custom Block.

In the presented example, you may have recognized a few of the presented
attributes. They are highlighted here for convenience:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/custom_objects/custom_block.py
      :language: python
      :emphasize-lines: 27-29, 49, 74
      :lines: 1-96

|

Invalid assignments to these properties raise :exc:`TypeError` or
:exc:`ValueError` immediately, which helps catch invalid custom Block
definitions before the test starts.

5.c. Sending data to other Blocks
+++++++++++++++++++++++++++++++++

A central aspect of a Block is how it communicates with other Blocks. Two
Blocks must be connected by a :class:`~crappy.links.Link` to exchange labeled
data.

Use :meth:`~crappy.blocks.Block.send` to send data to downstream Blocks. This
method accepts one
argument, either a :obj:`dict` or an :obj:`~collections.abc.Iterable` (like a
:obj:`list` or a :obj:`tuple`) of values to send (usually the values are
:obj:`float` or :obj:`str`). If a dictionary is given, its keys are the names
of the labels to send. For each label, a single value must be provided, and the
same labels should be sent throughout a given test. If the values are given in
an Iterable without labels, then the :py:`labels` attribute of the Block must
have been set beforehand. The dictionary to send will be reconstructed from the
labels and the given values. There must be as many given values
as there are labels.

.. Note::
   The dictionary sent through the Links are exactly the same that the
   :ref:`Modifiers <crappy_docs/modifiers:modifiers>` can access and modify. See the :ref:`dedicated section
   <tutorials/custom_objects:1. custom modifiers>` for more information.

The line in the example where the data gets sent is outlined below:

.. literalinclude:: /downloads/custom_objects/custom_block.py
   :language: python
   :emphasize-lines: 29
   :lines: 97-126

The example sends a dictionary, but it could instead use the :py:`labels`
attribute. It builds the dictionary with :obj:`zip`. The labels are
:py:`'t(s)'` for the time and the chosen output label for the
transferred value. The values to send are given by the :obj:`~struct.unpack`
function, that returns two :obj:`float` from binary data.

5.d. Receiving data from other Blocks
+++++++++++++++++++++++++++++++++++++

Now that the method for sending data has been covered, it is time to describe
the complementary methods that allow a Block to receive data from upstream
Blocks. Four methods cover different message-history and Link-separation
requirements:

- :meth:`~crappy.blocks.Block.recv_data` reads one message from each Link. It
  creates an empty :obj:`dict`, that it updates with **one** message (i.e. one
  sent :obj:`dict`) from each of the incoming :ref:`Links <crappy_docs/links:links>`, and then returns
  it. This means that some data might be lost if several Links carry a same
  label, which is often the case with the time label. Also, only the
  first available message of each Link is read, meaning that if there are
  several incoming messages in the queue, only one is queued out. For this
  reason, use this method only when one message per Link and merged labels are
  sufficient.
- :meth:`~crappy.blocks.Block.recv_last_data` is based on the same principle as
  the previous method, except it includes a loop that updates the dictionary
  to return with **all** the queued messages. In the end, only the latest
  received value for each label is present in the returned dictionary, hence
  the name of the method. A :py:`fill_missing` argument controls
  whether the last known value of each label is included if no newer value is
  available, thus returning the latest known value of **all** the known labels
  (not just the ones whose values were recently received). Just like the
  previous method, this one doesn't keep the integrity of the time information
  if there are several incoming Links, and only returns one value per label
  even if several messages were received.
- :meth:`~crappy.blocks.Block.recv_all_data` retains and returns
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
- :meth:`~crappy.blocks.Block.recv_all_data_raw` preserves the messages from
  each incoming Link separately. It is equivalent to a call to
  :meth:`~crappy.blocks.Block.recv_all_data` on each Link taken separately. All
  the results are then put together in one list, so this method returns a
  :obj:`list` of :obj:`dict` (one per Link) whose keys are :obj:`str` (labels)
  and values are :obj:`list` of all the received data for the given label.
  Use this method to retrieve the exact timestamps for labels that can come
  from different Links. You can
  check the :class:`~crappy.blocks.Grapher` or the
  :class:`~crappy.blocks.Multiplexer` Blocks for examples of usage.

Choose :meth:`~crappy.blocks.Block.recv_last_data` when only the latest value
of each label is needed. Choose
:meth:`~crappy.blocks.Block.recv_all_data_raw` when timestamp history must stay
separate across multiple incoming Links. Otherwise, use
:meth:`~crappy.blocks.Block.recv_all_data`.

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
returned data is a single :obj:`dict` with single values. Check this dictionary
before processing it because it might be empty or lack required data during a
given loop.

The `ready-to-run
examples <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
examples>`_ demonstrate additional configurations. Continue with the
:ref:`advanced custom-object guide
<tutorials/complex_custom_objects:more about custom objects in crappy>` for
Generator Paths, VisionBlocks, advanced Camera objects, and all-in-one Camera
processing.
