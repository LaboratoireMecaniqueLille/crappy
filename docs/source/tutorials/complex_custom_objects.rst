===================================
More about custom objects in Crappy
===================================

.. role:: py(code)
  :language: python
  :class: highlight

**This last page of the tutorials covers various advanced topics related to**
**the creation of custom objects in Crappy**. Unlike for the three previous
pages, the content of this fourth page will not be of interest for all users.
It is still interesting to go over it for users wanting to have a deeper
understanding of the module, or users with a specific need.

1. Custom Generator Paths
-------------------------

Starting from version 2.0.0, **it is now possible for users to create their**
**own** :ref:`Generator Paths` ! There are two reasons why this possibility was
added so late in the module. First, we're not certain that there is a need for
it. But since only a few modifications were needed to allow the creation of
custom Paths, it was decided to make it possible anyway. And second, the
implementation is a bit messier than for other custom objects. It should still
be accessible for most users though, don't worry !

Just like for the other custom objects, there is a template for creating
custom Paths and the Paths have to be children of
:class:`crappy.blocks.generator_path.meta_path.Path` :

.. code-block:: python

   import crappy

   class MyPath(crappy.blocks.generator_path.meta_path.Path):

       def __init__():
           super().__init__()

       def get_cmd(self, data):
           ...

As you can see, there are only two methods to define ! Just like for the other
custom objects, :meth:`~crappy.blocks.generator_path.meta_path.Path.__init__`
should initialize the parent class. It can also accept arguments, that will
correspond to the keys and values given in the :obj:`dict` passed to the
:ref:`Generator` Block. Note that in addition to these arguments, the value of
the last command sent by the Generator and the moment when it was sent are
accessible through the :py:`self.t0` and :py:`self.last_cmd` attributes.

The :meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method is for
generating the next command for the Generator to send. It must return the next
command as a :obj:`float` (:obj:`None` is also acceptable is there's no new
command to send). It accepts one argument, which is the :obj:`dict` returned by
the :meth:`~crappy.blocks.Block.recv_all_data` method of the Generator, and
that contains all the data recently received over incoming Links. It allows to
handle the case when Generator Paths have stop conditions based on the value of
a label, described in :ref:`this tutorials section
<3. Advanced Generator condition>`.

But how to handle the stop conditions ? And how to signal the Generator that a
stop condition was met ? This is where things get a bit trickier ! To indicate
that a stop condition is met, the
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method simply has
to raise a :exc:`StopIteration` exception. That can be done anytime, based on
any arbitrary criterion. However, to make it so that conditions like
:py:`'delay=10'` can be used, a
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method is
provided by the base :class:`~crappy.blocks.generator_path.meta_path.Path`
class. It takes a :obj:`str` or a :obj:`~collections.abc.Callable` or
:obj:`None` as its single argument, and always returns a Callable out of it.
This Callable accepts one argument, which is the :obj:`dict` that is passed as
an argument to :meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd`,
and it returns a :obj:`bool` indicating whether the stop condition is met or
not.

So, to summarize, if your custom Path does not accept a :py:`'condition'` or
equivalent argument, you're free to raise :exc:`StopIteration` whenever you
want to switch to the next Path based on arbitrary criteria. If you do have a
:py:`'condition'` or equivalent argument, you should first parse it during
:meth:`~crappy.blocks.generator_path.meta_path.Path.__init__` using the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method. It
will output a Callable, that you should store as a variable. Then, in the
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method, you should
call this variable with the :obj:`dict` from
:meth:`~crappy.blocks.Block.recv_all_data` as an argument. If it returns
:obj:`True` the condition is met and you should raise :exc:`StopIteration`.
Otherwise, you should return a value for the Generator to send.

It is definitely not the most straightforward implementation, but it is very
flexible and should fit most situations. Let's write a short example to make it
clearer how to create a custom Generator Path and how to handle the
conditions. This example generates a square wave, whose duty cycle can be
either fixed or controlled by the value of an input label :

.. literalinclude:: /downloads/complex_custom_objects/custom_path.py
   :language: python
   :emphasize-lines: 35, 40-41, 49, 52-53

.. Note::
   To run this example, you'll need to have the :mod:`matplotlib` and *scipy*
   Python modules installed.

This example contains all the ingredients described above. The parent class is
initialized, then the :py:`condition` argument is parsed with
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition`. In
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd`, the given
condition is checked based on the latest received data from upstream Blocks,
and raises :exc:`StopIteration` if needed. This method also returns
:obj:`float` values as expected, and the :py:`t0` attribute is used for
calculating the value to return.

The exact way the custom Path works won't be detailed here, but it should be
self-explanatory by just reading the code and the comments. You can
:download:`download this custom Path example
</downloads/complex_custom_objects/custom_path.py>` to run it locally on your
machine. You should see that the duty cycle of the generated square signal
varies according to the target duty cycle, as expected. In the `examples on
GitHub  <https://github.com/LaboratoireMecaniqueLille/crappy/examples/
custom_objects>`_, you'll find another example of a custom Generator Path.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Path, do not use the :func:`print` function ! Instead, use the
   :meth:`~crappy.blocks.generator_path.meta_path.Path.log` method provided by
   the parent :class:`~crappy.blocks.generator_path.meta_path.Path` class. This
   way, the log messages are included in the log file and handled in a nicer
   way by Crappy.

There's one more very specific point that we'd like to outline about the use of
Generator Paths in Crappy. Earlier, it was mentioned that the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method of
the base Path object accepts :obj:`~collections.abc.Callable`. More precisely,
it accepts Callables that take as only argument a :obj:`dict` whose keys are
:obj:`str` and values are :obj:`list`, and that return a :obj:`bool` value.
This means that it is actually possible to pass a Callable as the value for
the :py:`condition` argument, not just a :obj:`str` or :obj:`None` ! This
possibility is not often used, but at least you now know that it exists ! It
could for instance come in use if you want to use an existing Path, but you
have an unusual stop condition (e.g. one that depends on the values of two
labels).

2. More about custom InOuts
---------------------------

In addition to what was described in the tutorial section about :ref:`how to
create custom InOut objects <3. Custom InOuts>`, there is one more minor
feature that the :ref:`In / Out` possess and that is worth describing in the
tutorials. That is **the ability for an InOut to acquire data before a test**
**starts, and to use this data to offset the channels to zero**. To do so, the
script must match two conditions. First, the :py:`make_zero_delay` argument of
the :ref:`IOBlock` must be set to a positive value. And second, the used InOut
must have its :meth:`~crappy.inout.InOut.get_data` method defined (it cannot be
a pure stream class). If both of these conditions are met, then the InOut will
acquire data using :meth:`~crappy.inout.InOut.get_data` during
:meth:`~crappy.blocks.IOBlock.prepare` for the specified delay, and create
offsets so that for each acquired channel its value starts from zero at the
beginning of the test. It also works for streams, provided that the number of
channels acquired in *streamer* mode is the same as the number of channels
acquired by :meth:`~crappy.inout.InOut.get_data`.

**Thing get a bit trickier when the hardware can handle and tune offsets for**
**its channels** ! In such a case, it might be advantageous to set the zeroing
offsets directly on the device rather than relying on Crappy. To achieve that,
the :meth:`~crappy.inout.InOut.make_zero` method of the base
:class:`~crappy.inout.InOut` has to be overriden in the child InOut class, and
the way it is performed depends on the capabilities of the hardware. What is
usually done is that the :meth:`~crappy.inout.InOut.make_zero` method of the
base class calculates the offset values, and the one of the child class sets
these values on the hardware and resets the offsets on Crappy's side. This
kind of implementation can be found in the :ref:`Labjack T7` or the
:ref:`Comedi` InOuts. Check their code to see how it looks ! There is also a
very basic example of offsetting in the `examples on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/custom_objects>`_
where the method is overriden and the offsets are simply doubled.

There is no need for a specific example in this sub-section, it is mostly
included to signal the existence of the zeroing feature and the possibility for
users to override it.

3. More about custom Actuators
------------------------------

In the tutorial section about :ref:`how to create custom Actuator objects
<2. Custom Actuators>`, then entire speed management aspect in :py:`position`
mode was left out. **In this section, we're going to cover in more details**
**the possibilities for driving the speed in** :py:`position` mode, **and how**
**to write a** :meth:`~crappy.actuator.Actuator.set_position` **method**
**accordingly**.

In the :obj:`dict` containing information about the
:class:`~crappy.actuator.Actuator` to drive, there are two optional keys that
allow tuning the target speed in :py:`position` mode. They can both be set, or
only one, or none. These keys are :

- :py:`'speed'`, that sets a target speed value from the beginning of the test.
  This value might be overriden if :py:`'speed_cmd_label'` is given. If it is
  not overriden, it persists forever.
- :py:`'speed_cmd_label'`, that provides the name of a label carrying the
  target speed values. As soon as a value is received over this label, the
  previous target value is overriden and the new one is set.

If no target speed value is set, i.e. if none of the two possible keys is
provided or if :py:`'speed'` is not set and no target speed has been received
over the :py:`'speed_cmd_label'` so far, the target speed is set to
:obj:`None`.

Now, how is that reflected on your code when creating a custom Actuator ?
First, note that it only influences the
:meth:`~crappy.actuator.Actuator.set_position` method, all the other ones are
unaffected. The target speed value is always passed to the Actuator as the
second argument of the :meth:`~crappy.actuator.Actuator.set_position` method.
It is passed no matter its value, so it might be equal to :obj:`None` ! It is
your duty to handle the two situations when it has or hasn't an actual value.
For hardware that doesn't support speed adjustment when operated in position
mode, this argument can always be ignored. You can have a look at the
`Actuators distributed with Crappy
<https://github.com/LaboratoireMecaniqueLille/crappy/src/crappy/actuator>`_
to see how the various :meth:`~crappy.actuator.Actuator.set_position` methods
implement the speed management in position mode. Also, an example of a
:ref:`Machine` Block with a variable target speed can be found in the `examples
folder on GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/
examples/blocks>`_.

4. More about custom Cameras
----------------------------

Because image acquisition is such a complex topic, the
:class:`~crappy.camera.Camera` object is by far the richest of the classes
interfacing with hardware in Crappy. For that reason, not all of its features
could be presented in the previous tutorial sections. The missing ones are
introduced here instead. Note that they are clearly secondary compared to the
other features already presented !

4.a. Pre-defined settings
+++++++++++++++++++++++++

4.a.1. Trigger setting
""""""""""""""""""""""

:ref:`On the previous page <4. Custom Cameras>`, the three methods allowing to
instantiate a :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`
were presented. While these methods cover a wide range of situations, we found
that they were not always well-suited to manage the trigger setting that some
cameras possess. Indeed, when a camera is switched to external trigger mode, it
will only acquire images when receiving an external signal. But if this signal
is itself issued by a device controlled from Crappy, then the camera cannot
acquire images for display in the
:class:`~crappy.tool.camera_config.CameraConfig` window, as the
:class:`~crappy.inout.InOut` used for generating the signal will only do so
once the configuration window closes ! To address this problem, **a new**
**method was introduced specifically for instantiating a trigger setting :**
**the** :meth:`~crappy.camera.Camera.add_trigger_setting` **method** !

When calling this method, a new
:class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting` is
instantiated with the name :py:`'trigger'`. Its possible choices are
:py:`'Free run'`, :py:`'Hdw after config'` and :py:`'Hardware'`, and its
default is :py:`'Free run'`. The only arguments left for the user to set are
thus the getter and the setter methods. This trigger setting appears in the
configuration window just like any other setting, and can be accessed and
modified in the code as well. It is really just a normal setting, but with a
pre-determined name and choices !

When set to :py:`'Free run'` mode, the camera should acquire images without
needing an external trigger. When set to :py:`'Hardware'`, the camera should
only acquire images when receiving a hardware trigger. What is more interesting
is definitely the :py:`'Hdw after config'` mode : when set, the camera stays in
free run mode as long as the configuration window is opened, but switches to
hardware trigger mode as soon as the window is closed ! **This way, you can**
**adjust the various settings interactively in the configuration window, but**
**still use the hardware trigger mode for the test** !

As mentioned above, the user still has to define the getter and setter methods.
For the setter, both the :py:`'Free run'` and :py:`'Hdw after config'` settings
should set the camera to free run mode, and the :py:`'Hardware'` setting should
set the camera to hardware trigger mode. For the getter now, it should return
:py:`'Hardware'` is the camera is in hardware trigger mode, and either
:py:`'Free run'` or :py:`'Hdw after config'` otherwise, depending on the last
value set by the setter. It is not the most straightforward getter to
implement, we know ! This aspect should be improved in future releases, but for
now you'll have to cope with it. You can get inspiration from the :ref:`Xi API`
Camera that implements it already.

4.a.2. Software ROI setting
"""""""""""""""""""""""""""

In addition to the trigger setting, another improvement was brought to make
user's life easier : the :meth:`~crappy.camera.Camera.add_software_roi` method.
**It allows to crop the acquired images to the desired dimension**, so that
they take less space when recorded, or can be processed faster. The remaining
Region Of Interest should of course only contain the area relevant to your
test. Unlike the hardware ROI setting that some cameras might possess, this
setting does not influence the image acquisition, and thus does not improve the
acquisition rate.

Under the hood, the :meth:`~crappy.camera.Camera.add_software_roi` method
instantiates four
:class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting` managing
the position and size of the ROI. These settings are :py:`'ROI_x'`,
:py:`'ROI_y'`, :py:`'ROI_width'` and :py:`'ROI_height'`, and their arguments
are inaccessible to the user. The only values that the user has to provide are
the width and the height of the acquired images, as arguments to the
:meth:`~crappy.camera.Camera.add_software_roi` method.

The application of the software ROI to the acquired images is not automatic,
you have to run the :meth:`~crappy.camera.Camera.apply_soft_roi` on the
acquired image in order for it to be effective. It returns the cropped image,
or :obj:`None` if there's nothing left to display (shouldn't happen). You can
find examples of usage for the software ROI in
:class:`~crappy.camera.CameraOpencv`, or in the `examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/examples/blocks>`_.

4.b. Reload slider and choice settings
++++++++++++++++++++++++++++++++++++++

The software ROI setting described in the previous sub-section sure is nice,
but what happens to it when the size of the acquired images change because of
another setting that controls the image format ? After all, the limits of the
sliders that it creates depend on the image size given by the user, and once
the :meth:`~crappy.camera.Camera.open` method of :class:`~crappy.camera.Camera`
returns, there's no way to re-instantiate the settings. To address this
problem, and all the similar ones that users might face, we added the
possibility to "reload" the
:class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting` and the
:class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting`.
**Reloading a setting means either adjusting the limits of the slider, or**
**changing the labels and/or the number of choices, depending on the type of**
**setting**.

In practice, each setting (except for the boolean ones) possess a
:meth:`~crappy.camera.meta_camera.camera_setting.CameraSetting.reload` method,
that allows to reload it. The arguments to provide depend on the type of
setting. The calls to
:meth:`~crappy.camera.meta_camera.camera_setting.CameraSetting.reload` should
be placed in the relevant getter or setter methods, so that when the value of a
setting changes it adjusts the other settings accordingly. It is totally not
mandatory to do so, and most Cameras won't ever need to reload any setting. For
the specific case of the software ROI setting, the
:class:`~crappy.camera.Camera` class defines a specific
:meth:`~crappy.camera.Camera.reload_software_roi` method for reloading it. You
can check the :class:`~crappy.camera.CameraOpencv` Camera to see an example of
a class implementing a setting reload.

.. Important::
   The possibility to reload settings is still recent, and might not be fully
   stable. If you have trouble using it, please report it (see the
   :ref:`Troubleshooting` page).

4.c. Manage the metadata of the images
++++++++++++++++++++++++++++++++++++++

For the last feature of th :class:`~crappy.camera.Camera` objects presented in
the tutorials, let's introduce **the possibility to include metadata in the**
**information returned by a Camera** ! So far, it was always mentioned that the
first value that the :meth:`~crappy.camera.Camera.open` method of Cameras
should return is the timestamp of the acquired image. That is actually
incorrect, since it is also possible to return a :obj:`dict` containing
metadata about the acquired image ! This option is only interesting if the
used camera can return metadata, such as the frame number, the aperture, the
exposure time, etc.

The returned dictionary should replace the bare timestamp value, and must
contain at least two keys. The :py:`'t(s)'` key contains the timestamp of the
image, as given by the :obj:`time.time`. And the :py:`'ImageUniqueID'` key
should contain an integer allowing to identify the image, like the index of
the acquired frame. In the case when only a timestamp is returned (and not a
metadata :obj:`dict`), the frame index is calculated automatically by Crappy
based on the images it sees, but might not correspond to the real frame index
of the camera.

Apart from these two mandatory keys, the user is free to include any other key
carrying any other type of information. Relevant information in the context of
experimental research could be the moment when the image was captured
(different from the moment when it was transmitted to Crappy), the exposure
time, etc. All the data included in the returned dictionary is meant to be
written in a *metadata.csv* file saved along with the recorded images, that
contains for each image its metadata. For each key of the dictionary that is a
valid EXIF tag, the metadata will also be embedded in the recorded images if
the :mod:`PIL` backend is used for recording. The :py:`'ImageUniqueID'` is
already a valid EXIF tag, and the time information is split and recorded over
the :py:`'DateTimeOriginal'` and :py:`'SubsecTimeOriginal'` tags. For now, none
of the Cameras implemented in Crappy return metadata as a :obj:`dict`, but that
will change in future releases !

5. Custom Camera Blocks
-----------------------

On the previous tutorial page, :ref:`a section <5. Custom Blocks>` was
dedicated to the instantiation of custom :ref:`Blocks`. Always moving one step
further into customization, we're going to see in this section how you can
create your own subclass of a particular subclass of Block, namely the
:class:`~crappy.blocks.Camera` Block !

Basically, the Camera Block provides three functionalities. First, it acquires
images by driving a :ref:`Camera` object. Then, it can optionally display the
acquired images in a dedicated window. And third, it can optionally record the
acquired images. The great advantage of this Block is that it can perform these
three operations in parallel, and therefore optimize the framerate for each
functionality. The counterpart is that these three operations must be embedded
into a single Block, rather than performed separately by three different
Blocks. More details about the implementation of the Camera Block can be found
in the :ref:`Developers <Camera-related Blocks>` section of the documentation.

In the Camera Block, some lines of code provide the possibility to perform a
fourth operation in parallel : image processing on the acquired images. While
the Camera Block itself does not make use of this possibility, children of
Camera can use it very easily and implement parallelized image processing. For
instance, the :ref:`Video Extenso` and the :ref:`DIC VE` Blocks are children of
Camera that implement real-time video-extensometry on the acquired images. So,
in most cases, the Camera Block should be subclassed by users wishing to
implement their own custom image processing method in Crappy.

Now, in practice, how to write your own subclass of Camera ? As mentioned
above, the base Camera Block already handles the acquisition, the display, and
the recording of the images. All that's left for you to define is how to
correctly process the images, and what results to send to downstream Blocks.
But remember that just like the other functionalities, the processing is also
parallelized ! This means that it cannot be performed directly in the custom
Camera Block, but rather in another object : a
:class:`~crappy.blocks.camera_processes.CameraProcess`. So, anyone who wants to
implement their own image processing in Crappy must create two new classes :
one child of :class:`~crappy.blocks.Camera`, and one child of
:class:`~crappy.blocks.camera_processes.CameraProcess` !

5.a. The CameraProcess class
++++++++++++++++++++++++++++

Just like the other custom objects that you can instantiate in Crappy, there is
a template for the :class:`~crappy.blocks.camera_processes.CameraProcess` :

.. code-block:: python

   import crappy

   class MyCameraProcess(crappy.blocks.camera_processes.CameraProcess):

       def __init__():
           super().__init__()

       def init(self):
           ...

       def loop(self):
           ...

       def finish(self):
           ...

Let's review one by one the methods that you can define :

- In :meth:`~crappy.blocks.camera_processes.CameraProcess.__init__` you should
  only handle the arguments that your CameraProcess accepts, nothing more ! The
  reason for that is that this method runs in a separate "context" than the
  following ones, so as little as possible should be performed there.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.init` is where you can
  instantiate and initialize the various objects that you will use for the
  image processing. It is fine to leave this method undefined.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.loop` is called
  repeatedly, and is the equivalent of the :meth:`~crappy.blocks.Block.loop`
  method of the Block. It should handle the received images, process them, and
  send the result to downstream Blocks. The methods and objects to use for that
  are detailed below.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.finish` is the
  equivalent of the :meth:`~crappy.blocks.Block.finish` method of the Block. It
  is called at the very end when Crappy finishes, and should de-initialize the
  objects used for the image processing. It is fine to leave this method
  undefined.

The Base CameraProcess class handles the calls to these methods, as well as the
exceptions that might be raised. All the user has to do is to define them. In
addition to the methods that the user has to define, there are three other
methods that can be called and provide extra functionalities :

- :meth:`~crappy.blocks.camera_processes.CameraProcess.send` is the equivalent
  of the :meth:`~crappy.blocks.Block.send` method of the Block, of which it is
  almost an exact copy. It allows to send data to downstream Block, and takes
  one argument either as a :obj:`dict` or as an
  :obj:`~collections.abc.Iterable` if the :py:`self._labels` attribute is
  defined (and not :py:`self.labels` like in the Block). Refer to the method of
  Block for more information.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` allows to
  send :class:`~crappy.tool.camera_config.config_tools.Overlay` objects for the
  displayer to show as an overlay on top of the displayed images. It is
  discussed in more details in a :ref:`next subsection
  <5.c. Sending an overlay to the Displayer>`.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.log` is the equivalent
  of the :meth:`~crappy.blocks.Block.log` method of the Block, and allows
  handling log messages without resorting to the :obj:`print` function.

On top of that, two very useful attributes are defined by the CameraProcess
class :

- :py:`self.img` contains the latest image captured by the Camera Block, as a
  :mod:`numpy` array. It is updated automatically, so users just have to use it
  as is. Also note that the
  :meth:`~crappy.blocks.camera_processes.CameraProcess.loop` method is only
  called again if a new image was received since the last call, so
  :py:`self.img` should be a different image at every call !
- :py:`self.metadata` contains the metadata associated with the image stored in
  :py:`self.img`. The metadata is in the format described in :ref:`the
  dedicated section <4.c. Manage the metadata of the images>`. It is especially
  useful for retrieving the timestamp and the frame index of the processed
  image.

Now that you have a general overview of the methods and attribute that the
CameraProcess exposes, it is time to demonstrate how to use them in a demo
CameraProcess :

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-6, 31-53, 60-61

In the example code, the defined class uses OpenCV to detect eyes on the
received images. It returns the timestamp of the image, and an object
containing the coordinates of the detected eyes. Here, the
:meth:`~crappy.blocks.camera_processes.CameraProcess.finish` method is missing,
because there is nothing to de-initialize. As described above, the
:meth:`~crappy.blocks.camera_processes.CameraProcess.__init__` method only
handles the given arguments,
:meth:`~crappy.blocks.camera_processes.CameraProcess.init` makes the class
ready for looping, and
:meth:`~crappy.blocks.camera_processes.CameraProcess.loop` performs the main
detection task. The :py:`self.img` attribute is used as an argument to the eye
detection function, and :py:`self.metadata` is used for returning the timestamp
of the current image to downstream Blocks. This class alone is not enough for
running the eye detection with Crappy, a corresponding custom
:class:`~crappy.blocks.Camera` Block now has to be defined in the next
subsection !

.. Note::
   By default, the :meth:`~crappy.blocks.camera_processes.CameraProcess.loop`
   method is called every time a new image is grabbed by the CameraProcess. It
   is possible to tune this behavior by overriding the
   :meth:`~crappy.blocks.camera_processes.CameraProcess._get_data` method. See
   the :class:`~crappy.blocks.camera_processes.ImageSaver` Process for an
   example.

.. Note::
   By default, a counter accessible via the :py:`self.fps_count` attribute is
   incremented every time a new image is grabbed by the CameraProcess. It is
   only used in case the :py:`display_freq` argument of the
   :class:`~crappy.blocks.Camera` Block is set to :obj:`True`, to keep track of
   the framerate achieved by the CameraProcess. If you have specific situations
   to handle (e.g. a call to
   :meth:`~crappy.blocks.camera_processes.CameraProcess.loop` that does not
   actually process the new image), you can access the :py:`self.fps_count`
   attribute and decrement or modify it yourself.

5.b. Writing the custom Camera Block
++++++++++++++++++++++++++++++++++++

To be able to use your freshly defined custom
:class:`~crappy.blocks.camera_processes.CameraProcess`, you now have to create
a custom :class:`~crappy.blocks.Camera` Block that makes use of the
CameraProcess. Since most of the complexity is handled in the base parent
class, the template for a child of the Camera Block is pretty basic :

.. code-block:: python

   import crappy

   class MyCameraBlock(crappy.blocks.Camera):

       def __init__(self,
                    camera,
                    transform=None,
                    config=True,
                    display_images=False,
                    displayer_backend=None,
                    displayer_framerate=5,
                    software_trig_label=None,
                    display_freq=False,
                    freq=200,
                    debug=False,
                    save_images=False,
                    img_extension="tiff",
                    save_folder=None,
                    save_period=1,
                    save_backend=None,
                    image_generator=None,
                    img_shape=None,
                    img_dtype=None,
                    **kwargs):

           super().__init__(camera=camera,
                            transform=transform,
                            config=config,
                            display_images=display_images,
                            displayer_backend=displayer_backend,
                            displayer_framerate=displayer_framerate,
                            software_trig_label=software_trig_label,
                            display_freq=display_freq,
                            freq=freq,
                            debug=debug,
                            save_images=save_images,
                            img_extension=img_extension,
                            save_folder=save_folder,
                            save_period=save_period,
                            save_backend=save_backend,
                            image_generator=image_generator,
                            img_shape=img_shape,
                            img_dtype=img_dtype,
                            **kwargs)

       def prepare(self):
           self.process_proc = CustomCameraProcess()

Notice that since your new :meth:`~crappy.blocks.Camera.__init__` method
overrides the one from the parent class, you have to handle all the parameters
of the parent class in addition to the ones that you might add ! As usual,
:meth:`~crappy.blocks.Camera.__init__` should instantiate all the objects that
will be used in your class and handle the arguments. In simple cases,
:meth:`~crappy.blocks.Camera.prepare` is very basic and is only used for
setting the CameraProcess to use. Except for that, there is nothing more to
do on the Camera Block side !

.. Note::
   If you use the :class:`~crappy.blocks.VideoExtenso` Block for example, you
   have to select spots to track in the configuration window. To achieve such
   a behavior, you'll need to override the
   :meth:`~crappy.blocks.Camera._configure` method in your child Camera Block,
   and to define your own version of
   :class:`~crappy.tool.camera_config.CameraConfig`. This possibility is very
   specific, so it is not described in the tutorials.

5.c. Sending an overlay to the Displayer
++++++++++++++++++++++++++++++++++++++++

Because the :class:`~crappy.blocks.camera_processes.CameraProcess` deals with
images, it can be interesting to have a real-time display of how the processing
is performing. To do so, the base CameraProcess class provides the
:meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` method that
allows to send objects to the
:class:`~crappy.blocks.camera_processes.Displayer` Process to draw overlays on
top of the displayed images. Of course, it will only work if the
:py:`display_images` argument of the Camera Block is set to :obj:`True`.

The objects indicating what to draw should be children of the
:class:`~crappy.tool.camera_config.config_tools.Overlay` class. They only need
to define the :meth:`~crappy.tool.camera_config.config_tools.Overlay.draw`
method, that takes the image to display as an argument and draws the overlay on
top of it. Here is what it looks like for displaying a black ellipse :

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-29

To transmit the overlay to the Displayer Process, the
:meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` should send
a collection of instances of Overlays. It is as simple as that ! Crappy only
comes with one predefined Overlay object, the
:class:`~crappy.tool.camera_config.config_tools.Box`, but it is easy enough to
define your own ones. Here is what the custom CameraProcess defined in the
previous sub-section looks like after integrating the code for sending
overlays :

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-6, 31-61

5.d. Final runnable example
+++++++++++++++++++++++++++

It is now time to put together all the custom classes that were defined in the
previous sub-sections. There is first the custom
:class:`~crappy.tool.camera_config.config_tools.Overlay` class for drawing an
ellipse overlay on top of the displayed images. It is used by the custom
:class:`~crappy.blocks.camera_processes.CameraProcess` that performs eye
detection on the acquired images. This custom CameraProcess is itself
instantiated by a custom child of the :class:`~crappy.blocks.Camera` Block,
that is the final object called by the user in its script. Based on these
development, here is a final runnable code performing eye detection and adding
the detected eyes on the displayed images :

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python

.. Note::
   To run this example, you'll need to have the *opencv-python* and *Pillow*
   Python modules installed.

This custom Camera Block script is based on an example that you can find in the
`examples folder on GitHub  <https://github.com/LaboratoireMecaniqueLille/
crappy/examples/custom_objects>`__. You can :download:`download it
</downloads/complex_custom_objects/custom_camera_block.py>` to run it locally
on your machine. Note that the :py:`'Webcam'` camera is used here, so this
example will require a camera readable by OpenCV to be plugged to the computer.
The instantiation of custom image processing in Crappy is definitely one of the
most advanced things you can perform, but it is totally worth it if you want to
have your processing parallelized with the acquisition and the display and/or
recording of the images. There will likely be changes and improvements on these
aspects in future releases.

6. Sharing custom objects and Blocks
------------------------------------

You have been through all the tutorials of Crappy and have now become a master
at creating and using your own objects, and you now **want to share your**
**works with other user** ? No problem ! There are several options for that,
some very simple and some other much more demanding. Let's review them all in
this last section of the tutorials !

The first and **most simple option for sharing your custom objects is to put**
**them in separate files**, along with their necessary imports, and to share
these files. Other people will be able to use them by importing your custom
objects in their script, e.g. with :py:`from file_name import CustomObject`. It
is the way to go in most situations, as it is very quicly done and only
requires to send one or a few files. The persons who receive the file can also
easily modify it and share it themselves. You got it, the main advantage of
processing this way is that it is very flexible. The only drawback is that the
versions of Crappy for the sender and the receiver might not be the same, in
which case the code might not run on the receiver's side. Also, for the
receiver, two steps are involved : installing Crappy and copying the sent
files.

Some users might want to distribute their work in a more rigid way, for example
an engineer distributing the same immutable code to several users of a machine.
It is possible to **create and share installation files**, a.k.a *wheels*,
**that contain a modified version of Crappy** that can be installed using
:mod:`pip`. To do so, one has to clone Crappy, i.e. make a local copy of its
source files, modify it to include the new custom objects, and build the wheel
to share. This way, everyone runs the same code, and also cannot have an
incompatible version since the version is fixed by the creator of the wheel.
How to properly modify a Python module to include new files is not described
here, neither is how to build and install a new wheel. This paragraph simply
indicates the possibility to do so. You should however find plenty of help on
internet if you want to give it a try !

There is one last possible way to share your work, it is to **integrate it to**
**the collection of classes and Blocks distributed with the official version**
**of Crappy** ! To do so, you should first *fork* Crappy, i.e. create a copy of
it on your own GitHub account. After modifying this copy to include your own
files, you can submit a *pull request* to the maintainers to request
integration of your work on the official repository of Crappy. Again, this
paragraph is not a *git* or GitHub tutorial, and we're not going to give more
details about this whole process. If you wish to contribute to Crappy, you
should anyway get in touch with the developers on GitHub at some point ! For
contributors, the :ref:`Developers information` page of the documentation
provides a few guidelines, as well as more insights on the content of the
module than the tutorials. If there's a feature you would like to see in
Crappy, but that you don't feel capable of implementing yourself, you can also
request improvements directly on GitHub.

That concludes the tutorials of Crappy ! We hope they have been helpful for
getting started with the module, and that you were able to find an answer to
all your questions here. If not, do not hesitate to request halp on our GitHub
page ! Also, **if you publish any academic work conducted with the help of**
**Crappy, please do not forget to** :ref:`cite us <Citing Crappy>` !
