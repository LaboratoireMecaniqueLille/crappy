===================================
More about custom objects in Crappy
===================================

.. role:: py(code)
  :language: python
  :class: highlight

This page covers advanced ways to customize Crappy: Generator Paths, zeroing
InOuts, position-controlled Actuators, Camera settings, VisionBlocks,
all-in-one Camera Blocks, and distribution of custom objects.

The :doc:`../concepts/choosing_custom_object_type` guide explains which custom
object type matches a device or task. Review :doc:`../concepts/image_pipelines`
before choosing between a custom VisionBlock and the all-in-one Camera
customization path.

1. Custom Generator Paths
-------------------------

Since version 2.0.0, users can create custom
:ref:`Generator Paths <crappy_docs/blocks:generator paths>`. A custom Path
defines how the Generator produces commands and when it advances to the next
Path.

Just like for the other custom objects, there is a template for creating
custom Paths and the Paths have to be children of
:class:`crappy.blocks.generator_path.meta_path.Path`:

.. code-block:: python

   import crappy

   class MyPath(crappy.blocks.generator_path.meta_path.Path):

       def __init__():
           super().__init__()

       def get_cmd(self, data):
           ...

The template defines two methods. As with the other
custom objects, :meth:`~crappy.blocks.generator_path.meta_path.Path.__init__`
should initialize the parent class. It can also accept arguments, that will
correspond to the keys and values given in the :obj:`dict` passed to the
:ref:`Generator <crappy_docs/blocks:generator>` Block. Note that in addition to
these arguments, the value of the last command sent by the Generator and the
moment when it was sent are accessible through the :py:`self.t0` and
:py:`self.last_cmd` attributes.

The :meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method is for
generating the next command for the Generator to send. It must return the next
command as a :obj:`float` (:obj:`None` is also acceptable if there is no new
command to send). It accepts one argument, which is the :obj:`dict` returned by
the :meth:`~crappy.blocks.Block.recv_all_data` method of the Generator, and
that contains all data recently received over incoming Links. This supports
Generator Paths with stop conditions based on the value of
a label, described in the :ref:`conditional Generator tutorial
<tutorial-generator-conditions>`.

To signal that a stop condition is met, the
:meth:`~crappy.blocks.generator_path.meta_path.Path.get_cmd` method raises a
:exc:`StopIteration` exception. It can do so at any time, based on
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

The following example demonstrates a custom Generator Path and its stop
condition. It generates a square wave whose duty cycle can be
either fixed or controlled by the value of an input label:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/complex_custom_objects/custom_path.py
      :language: python
      :emphasize-lines: 35, 40-41, 49, 52-53

|

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
varies according to the target duty cycle, as expected. In the `custom objects
examples on GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/tree/
master/examples/custom_objects>`_, you'll find another example of a custom
Generator Path.

.. Note::
   If you want to have debug information displayed in the terminal from your
   Path, do not use the :func:`print` function. Instead, use the
   :meth:`~crappy.blocks.generator_path.meta_path.Path.log` method provided by
   the parent :class:`~crappy.blocks.generator_path.meta_path.Path` class. This
   way, the log messages are included in the log file and handled by
   Crappy's centralized logging.

There's one more very specific point that we'd like to outline about the use of
Generator Paths in Crappy. Earlier, it was mentioned that the
:meth:`~crappy.blocks.generator_path.meta_path.Path.parse_condition` method of
the base Path object accepts :obj:`~collections.abc.Callable`. More precisely,
it accepts Callables that take as only argument a :obj:`dict` whose keys are
:obj:`str` and values are :obj:`list`, and that return a :obj:`bool` value.
This means that it is actually possible to pass a Callable as the value for
the :py:`condition` argument, not just a :obj:`str` or :obj:`None`. This is
useful with an existing Path and an unusual stop condition, such as one that
depends on two labels.

2. More about custom InOuts
---------------------------

In addition to what was described in the tutorial about :ref:`how to create
custom InOut objects <tutorial-custom-inout>`, there
is one more minor feature that the
:ref:`In / Out <crappy_docs/inouts:in / out>` objects provide: an InOut can
acquire data before a test starts and
use this data to offset its channels to zero. To do so, the script must match
two conditions. First, the
:py:`make_zero_delay` argument of the
:ref:`IOBlock <crappy_docs/blocks:ioblock>` must be set to a positive value.
And second, the used InOut must have its :meth:`~crappy.inout.InOut.get_data`
method defined (it cannot be a pure stream class). If both of these conditions
are met, then the InOut will acquire data using
:meth:`~crappy.inout.InOut.get_data` during
:meth:`~crappy.blocks.IOBlock.prepare` for the specified delay, and create
offsets so that for each acquired channel its value starts from zero at the
beginning of the test. It also works for streams, provided that the number of
channels acquired in *streamer* mode is the same as the number of channels
acquired by :meth:`~crappy.inout.InOut.get_data`.

When the hardware supports per-channel offsets, you can set the zeroing
offsets directly on the device rather than relying on Crappy. To achieve that,
the :meth:`~crappy.inout.InOut.make_zero` method of the base
:class:`~crappy.inout.InOut` has to be overridden in the child InOut class, and
the way it is performed depends on the capabilities of the hardware. What is
usually done is that the :meth:`~crappy.inout.InOut.make_zero` method of the
base class calculates the offset values, and the one of the child class sets
these values on the hardware and resets the offsets on Crappy's side. This
kind of implementation can be found in the
:ref:`Labjack T7 <crappy_docs/inouts:labjack t7>` or the
:ref:`Comedi <crappy_docs/inouts:comedi>` InOuts. There is also an offsetting example in the `examples
on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
custom_objects>`_ where the method is overridden and the offsets are simply
doubled.

There is no need for a specific example in this sub-section, it is mostly
included to signal the existence of the zeroing feature and the possibility for
users to override it.

3. More about custom Actuators
------------------------------

In the tutorial section about :ref:`how to create custom Actuator objects
<tutorial-custom-actuator>`, the entire speed management
aspect in :py:`position` mode was left out. This section explains target-speed
inputs in :py:`position` mode and the corresponding
:meth:`~crappy.actuator.Actuator.set_position` implementation.

In the :obj:`dict` containing information about the
:class:`~crappy.actuator.Actuator` to drive, there are two optional keys that
allow tuning the target speed in :py:`position` mode. They can both be set, or
only one, or none. These keys are:

- :py:`'speed'`, that sets a target speed value from the beginning of the test.
  This value might be overridden if :py:`'speed_cmd_label'` is given. If it is
  not overridden, it persists for the test duration.
- :py:`'speed_cmd_label'`, that provides the name of a label carrying the
  target speed values. As soon as a value is received over this label, the
  previous target value is overridden and the new one is set.

If no target speed value is set, i.e. if none of the two possible keys is
provided or if :py:`'speed'` is not set and no target speed has been received
over the :py:`'speed_cmd_label'` so far, the target speed is set to
:obj:`None`.

For a custom Actuator, target speed only affects the
:meth:`~crappy.actuator.Actuator.set_position` method, all the other ones are
unaffected. The target speed value is always passed to the Actuator as the
second argument of the :meth:`~crappy.actuator.Actuator.set_position` method.
It is passed even when its value is :obj:`None`. Handle both a numerical speed
and :obj:`None`.
For hardware that doesn't support speed adjustment when operated in position
mode, this argument can always be ignored. You can have a look at the
`Actuators distributed with Crappy <https://github.com/
LaboratoireMecaniqueLille/crappy/tree/master/src/crappy/actuator>`_ to see how
the various :meth:`~crappy.actuator.Actuator.set_position` methods implement
the speed management in position mode. Also, an example of a
:ref:`Machine <crappy_docs/blocks:machine>` Block with a variable target speed
can be found in the `blocks examples folder on GitHub
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/blocks>`_.

4. More about custom Cameras
----------------------------

Because image acquisition is such a complex topic, the
:class:`~crappy.camera.Camera` object exposes settings, image metadata, and
region-of-interest helpers in addition to its hardware lifecycle. Not all features
could be presented in the previous tutorial sections. The following sections
cover the remaining specialized features.

4.a. Pre-defined settings
+++++++++++++++++++++++++

4.a.1. Trigger setting
""""""""""""""""""""""

:ref:`In the custom Camera tutorial <tutorial-custom-camera>`, the
three methods allowing to instantiate a
:class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` were
presented. While these methods cover a wide range of situations, we found that
they were not always well-suited to manage the trigger setting that some
cameras possess. Indeed, when a camera is switched to external trigger mode, it
will only acquire images when receiving an external signal. But if this signal
is itself issued by a device controlled from Crappy, then the camera cannot
acquire images for display in the
:class:`~crappy.tool.camera_config.CameraConfig` window, as the
:class:`~crappy.inout.InOut` used for generating the signal will only do so
once the configuration window closes. Use
:meth:`~crappy.camera.Camera.add_trigger_setting` for this case.

When calling this method, a new
:class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting` is
instantiated with the name :py:`'trigger'`. Its possible choices are
:py:`'Free run'`, :py:`'Hdw after config'` and :py:`'Hardware'`, and its
default is :py:`'Free run'`. The only arguments left for the user to set are
thus the getter and the setter methods. This trigger setting appears in the
configuration window just like any other setting, and can be accessed and
modified in the code as well. It has a predefined name and set of choices.

When set to :py:`'Free run'` mode, the camera should acquire images without
needing an external trigger. When set to :py:`'Hardware'`, the camera should
only acquire images when receiving a hardware trigger. What is more interesting
The :py:`'Hdw after config'` mode keeps the camera in
free run mode as long as the configuration window is opened, but switches to
hardware trigger mode as soon as the window closes. This permits interactive
configuration followed by hardware-triggered acquisition during the test.

As mentioned above, the user still has to define the getter and setter methods.
For the setter, both the :py:`'Free run'` and :py:`'Hdw after config'` settings
should set the camera to free run mode, and the :py:`'Hardware'` setting should
set the camera to hardware trigger mode. For the getter now, it should return
:py:`'Hardware'` is the camera is in hardware trigger mode, and either
:py:`'Free run'` or :py:`'Hdw after config'` otherwise, depending on the last
value set by the setter. See the
:ref:`Xi API <crappy_docs/cameras:xi api>` Camera that implements it already.

4.a.2. Software ROI setting
"""""""""""""""""""""""""""

In addition to the trigger setting, another improvement was brought to make
camera integration easier: the :meth:`~crappy.camera.Camera.add_software_roi`
method. It crops acquired images to the desired dimensions so that
they take less space when recorded, or can be processed faster. The remaining
region of interest (ROI) should contain the area relevant to your
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
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
blocks>`_.

4.b. Reload slider and choice settings
++++++++++++++++++++++++++++++++++++++

The software ROI limits depend on the acquired image size. When another setting
changes the image format, the limits of the
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

In practice, each setting except a Boolean setting has a
:meth:`~crappy.camera.meta_camera.camera_setting.CameraSetting.reload` method,
that reloads it. The required arguments depend on the type of
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
   :ref:`Troubleshooting <troubleshooting:troubleshooting>` page).

4.c. Manage the metadata of the images
++++++++++++++++++++++++++++++++++++++

The :meth:`~crappy.camera.Camera.get_image` method can return an image metadata
dictionary instead of a bare timestamp. This option is useful when the
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
the :py:`'DateTimeOriginal'` and :py:`'SubsecTimeOriginal'` tags. For now, only
a fraction of the Camera objects implemented in Crappy return metadata as a
:obj:`dict`.

5. Custom VisionBlocks
----------------------

The :class:`~crappy.blocks.vision.VisionBlock` class is the recommended base
for a custom Block that produces, consumes, or transforms images. Typical
reasons for writing one include integrating an image source that is not a
:ref:`Camera <crappy_docs/cameras:camera>`, implementing a new analysis
algorithm, or adapting images for another VisionBlock. Since it is an ordinary
Block, a custom VisionBlock can be combined with the built-in display,
recording, and processing Blocks in any useful arrangement.

Creating a new class is not necessary just to rearrange an image workflow. If
the built-in VisionBlocks already perform the required tasks, instantiate and
connect them directly. Likewise, use a regular :class:`~crappy.blocks.Block`
when a custom operation only handles labeled numerical data. The
:class:`~crappy.blocks.vision.VisionBlock` base is specifically useful when
images must enter or leave the custom Block through an
:class:`~crappy.links.ImageLink`.

5.a. Choose the role of the Block
+++++++++++++++++++++++++++++++++

Before writing code, decide what the new Block accepts and produces. Here are
the typical categories, but exotic objects might behave differently:

- An **image source** has no image input and one or more image outputs. It
  might wrap a third-party acquisition library, read a generated image stream,
  synthesize images, etc.
- An **image analyzer** has one or more image inputs and normally no image
  output. It publishes measurements or display overlays through regular Links.
- An **image filter** has both image inputs and image outputs. It receives an
  image, transforms it, and publishes the result for other VisionBlocks.

These are design roles rather than restrictions imposed by the base class. A
custom VisionBlock decides which combinations it supports and checks them in
``prepare``. The ``img_inputs`` and ``img_outputs`` lists contain the
ImageLinks connected by the script and should only be inspected, not modified.
The separate ``inputs`` and ``outputs`` lists contain regular Links.

The :doc:`../concepts/regular_links_and_image_links` page defines ImageLink
transport and graph constraints. One source can feed any number of different
consumers. Regular Links can be added in either direction for results,
commands, triggers, and feedback.

5.b. The rules to follow
++++++++++++++++++++++++

A custom VisionBlock only needs to follow a small set of rules:

1. Call the parent constructor. If the Block has image outputs, provide their
   ``img_shape`` and ``img_dtype``. A Block with no image output can omit them.
2. In ``prepare``, validate the supported Link arrangement and initialize any
   library or resource used by the Block. Call the parent ``prepare`` last.
3. Use :meth:`~crappy.blocks.vision.VisionBlock.send_img` to publish images and
   :meth:`~crappy.blocks.vision.VisionBlock.receive_imgs` to receive them.
   Continue to use :meth:`~crappy.blocks.Block.send` and the usual Block
   receive methods for labeled data.
4. Every published metadata dictionary must contain ``'t(s)'`` and
   ``'ImageUniqueID'``. The image shape and dtype must match the values
   declared for the output.
5. If ``finish`` is overridden, release the custom resources and then call the
   parent ``finish``. If no custom cleanup is needed, do not override it.

The parent calls are the only image-management boilerplate required from a
custom class. Similarly, if ``begin`` is overridden, its parent implementation
should be called.

5.c. Example: write an image source
+++++++++++++++++++++++++++++++++++

The following source generates a bright column moving across a black image. It
declares its output format in ``__init__``, checks that the script connected it
as a source, and publishes one image from every call to ``loop``:

.. code-block:: python

   from time import time

   import numpy as np
   import crappy

   class MovingSource(crappy.VisionBlock):

       def __init__(self):
           super().__init__(img_shape=(120, 160),
                            img_dtype='uint8',
                            freq=10)
           self._index = 0

       def prepare(self):
           if self.img_inputs or not self.img_outputs:
               raise IOError("MovingSource requires image outputs only")
           super().prepare()

       def loop(self):
           image = np.zeros((120, 160), dtype=np.uint8)
           image[:, self._index % 160] = 255
           metadata = {'t(s)': time() - self.t0,
                       'ImageUniqueID': self._index}

           self.send_img(metadata, image)
           self._index += 1

           if self._index >= 30:
               self.stop()

The shape and dtype passed to ``send_img`` must stay ``(120, 160)`` and
``uint8`` throughout the test. Extra application-specific metadata can be
added freely. Calling ``send_img`` once makes the same image and metadata
available to every downstream ImageLink.

5.d. Example: write an image analyzer
+++++++++++++++++++++++++++++++++++++

This second Block accepts exactly one image stream and sends the mean pixel
value as ordinary labeled data:

.. code-block:: python

   import numpy as np
   import crappy

   class ImageMean(crappy.VisionBlock):

       def __init__(self):
           super().__init__(freq=50)

       def prepare(self):
           if len(self.img_inputs) != 1 or self.img_outputs:
               raise IOError("ImageMean requires exactly one image input")
           super().prepare()

       def loop(self):
           updated = self.receive_imgs()
           if not updated:
               return

           received = self.last_received[updated[0]]
           if received.metadata is None:
               raise RuntimeError("Image metadata is missing")

           self.send({'t(s)': received.metadata['t(s)'],
                      'mean': float(np.mean(received.img))})

``receive_imgs`` returns the names of the ImageLinks on which a new image was
found. The matching image and metadata are available under that name in
``last_received``. With several accepted inputs, the returned names make it
possible to handle only the sources that were updated.

The result from ``ImageMean`` is a small dictionary, so it belongs on a regular
Link. The two custom Blocks and a standard :class:`~crappy.blocks.LinkReader`
can now be assembled as follows:

.. code-block:: python

   source = MovingSource()
   mean = ImageMean()
   reader = crappy.blocks.LinkReader()

   crappy.img_link(source, mean, name='source-images')
   crappy.link(mean, reader)

   crappy.start()

This script stops after the source publishes 30 images. The `complete custom
VisionBlock example <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
master/examples/vision_blocks/custom_vision_blocks.py>`_ adds more topology
checks, image statistics, logging, and comments while retaining the same
source-and-analyzer structure.

5.e. Limits and advanced cases
++++++++++++++++++++++++++++++

There are a few important limits to consider when designing a custom
VisionBlock:

- Design consumers for the latest-frame behavior documented in
  :doc:`../concepts/regular_links_and_image_links`. A consumer can skip frames
  and must use the metadata it actually receives.
- A Block has one output image format for the duration of a test. All images
  sent by it must have that shape and dtype, and all its outgoing ImageLinks
  expose the same published image. Use separate VisionBlocks when a workflow
  needs distinct image variants.
- A custom filter combines the two examples above: it declares its output
  format, accepts both input and output ImageLinks, calls ``receive_imgs``, and
  publishes the transformed array with ``send_img``.
- Most custom Blocks should receive their settings through constructor
  arguments or regular Links. Source-side interactive configuration through
  :meth:`~crappy.blocks.vision.VisionBlock.request_config` is an advanced case:
  refer to the built-in :class:`~crappy.blocks.vision.DICVEProcessor`,
  :class:`~crappy.blocks.vision.DISCorrelProcessor`, and
  :class:`~crappy.blocks.vision.VideoExtensoProcessor` implementations when
  that behavior is required.

6. Custom Camera Blocks (all-in-one architecture)
-------------------------------------------------

This section describes how to subclass the all-in-one
:class:`crappy.blocks.Camera` Block and add an internal processing stage.

The all-in-one Camera Block provides three functions. First, it acquires
images by driving a :ref:`Camera <crappy_docs/cameras:camera>` object. Then, it
can optionally display the acquired images in a dedicated window. And third, it
can optionally record the acquired images. It performs these operations in
separate processes, but embeds them in one Block rather than exposing them as
independent graph nodes. More details about the implementation of
the Camera Block can be found in the
:ref:`contributor architecture <architecture-all-in-one-camera>`.

VisionBlocks are recommended for new image pipelines. The all-in-one Camera
Blocks remain supported and are not planned for deprecation. For a new
processing stage, a custom
:ref:`VisionBlock <tutorials/complex_custom_objects:5. custom visionblocks>` is
has fewer responsibilities because it does not also own the Camera object,
displayer, and recorder. The all-in-one approach described below can
still be convenient when those components should deliberately be exposed as a
single Block.

The Camera Block can perform a fourth operation in a separate process: image
processing. Camera Block subclasses select a processing implementation. For
instance, the :ref:`Video Extenso <crappy_docs/blocks:video extenso>` and the
:ref:`DIC VE <crappy_docs/blocks:dic ve>` Blocks are children of Camera that
implement real-time video-extensometry on the acquired images.

The base Camera Block handles image acquisition, display, and recording. A
custom extension defines how to process images and which results to send to
downstream Blocks. Processing runs in another process through a
:class:`~crappy.blocks.camera_processes.CameraProcess`. Using this all-in-one
architecture for custom image processing requires two new classes:
one child of :class:`~crappy.blocks.Camera`, and one child of
:class:`~crappy.blocks.camera_processes.CameraProcess`.

6.a. The CameraProcess class
++++++++++++++++++++++++++++

Just like the other custom objects that you can instantiate in Crappy, there is
a template for the :class:`~crappy.blocks.camera_processes.CameraProcess`:

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

The class can define these methods:

- In :meth:`~crappy.blocks.camera_processes.CameraProcess.__init__` you should
  only handle the arguments that your CameraProcess accepts. The
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
  is called at the very end when Crappy finishes, and should deinitialize the
  objects used for the image processing. It is fine to leave this method
  undefined.

The Base CameraProcess class handles the calls to these methods, as well as the
exceptions that might be raised. All the user has to do is to define them. In
addition to the methods that the user has to define, there are four other
methods that can be called and provide extra functionalities:

- :meth:`~crappy.blocks.camera_processes.CameraProcess.set_config` is an
  optional hook for processing state chosen in a custom CameraConfig window.
  The base Camera Block calls it before the CameraProcess starts, unpacking the
  tuple returned by
  :meth:`~crappy.tool.camera_config.CameraConfig.get_config`. Its arguments
  should be stored for use by
  :meth:`~crappy.blocks.camera_processes.CameraProcess.init`.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.send` is the equivalent
  of the :meth:`~crappy.blocks.Block.send` method of the Block, of which it is
  almost an exact copy. It sends data to downstream Blocks and takes
  one argument either as a :obj:`dict` or as an
  :obj:`~collections.abc.Iterable` if the :py:`self._labels` attribute is
  defined (and not :py:`self.labels` like in the Block). Refer to the method of
  Block for more information.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` sends
  :class:`~crappy.tool.camera_config.config_tools.Overlay` objects for the
  displayer to show as an overlay on top of the displayed images. It is
  discussed in more details in a :ref:`next subsection
  <tutorials/complex_custom_objects:6.c. sending an overlay to the displayer>`.
- :meth:`~crappy.blocks.camera_processes.CameraProcess.log` is the equivalent
  of the :meth:`~crappy.blocks.Block.log` method of the Block, and allows
  handling log messages without resorting to the :obj:`print` function.

On top of that, two very useful attributes are defined by the CameraProcess
class:

- :py:`self.img` contains the latest image captured by the Camera Block, as a
  :mod:`numpy` array. It is updated automatically, so users just have to use it
  as is. Also note that the
  :meth:`~crappy.blocks.camera_processes.CameraProcess.loop` method is only
  called again after a new image is received, so :py:`self.img` corresponds to
  the newly received frame.
- :py:`self.metadata` contains the metadata associated with the image stored in
  :py:`self.img`. The metadata is in the format described in :ref:`the
  dedicated section <tutorials/complex_custom_objects:4.c. manage the metadata
  of the images>`. It is especially useful for retrieving the timestamp and the
  frame index of the processed image.

Now that you have a general overview of the methods and attribute that the
CameraProcess exposes, it is time to demonstrate how to use them in a demo
CameraProcess:

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-6, 31-53, 60-61

In the example code, the defined class uses OpenCV to detect eyes on the
received images. It returns the timestamp of the image, and an object
containing the coordinates of the detected eyes. Here, the
:meth:`~crappy.blocks.camera_processes.CameraProcess.finish` method is missing,
because there is nothing to deinitialize. As described above, the
:meth:`~crappy.blocks.camera_processes.CameraProcess.__init__` method only
handles the given arguments,
:meth:`~crappy.blocks.camera_processes.CameraProcess.init` makes the class
ready for looping, and
:meth:`~crappy.blocks.camera_processes.CameraProcess.loop` performs the main
detection task. The :py:`self.img` attribute is used as an argument to the eye
detection function, and :py:`self.metadata` is used for returning the timestamp
of the current image to downstream Blocks. This class alone is not enough for
running the eye detection with Crappy. The next subsection defines the
corresponding custom :class:`~crappy.blocks.Camera` Block.

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

6.b. Writing the custom Camera Block
++++++++++++++++++++++++++++++++++++

To use the custom
:class:`~crappy.blocks.camera_processes.CameraProcess`, create
a custom :class:`~crappy.blocks.Camera` Block that makes use of the
CameraProcess. Since most of the complexity is handled in the base parent
class, the template for a child of the Camera Block is pretty basic:

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
overrides the one from the parent class, handle all parent parameters in
addition to any new ones. The
:meth:`~crappy.blocks.Camera.__init__` should instantiate all the objects that
will be used in your class and handle the arguments. When no extra
configuration is required, :meth:`~crappy.blocks.Camera.prepare` only selects
the CameraProcess.

.. Note::
   If you use the :class:`~crappy.blocks.VideoExtenso` Block for example, you
   have to select spots to track in the configuration window. To achieve such
   a behavior, you'll need to override the
   :meth:`~crappy.blocks.Camera._configure` method in your child Camera Block,
   and to define your own version of
   :class:`~crappy.tool.camera_config.CameraConfig`. Its
   :meth:`~crappy.tool.camera_config.CameraConfig.get_config` method must return
   a tuple matching the arguments of the custom CameraProcess's
   :meth:`~crappy.blocks.camera_processes.CameraProcess.set_config` method.
   The base Camera Block performs this handoff after the window closes and
   before starting the CameraProcess.

6.c. Sending an overlay to the Displayer
++++++++++++++++++++++++++++++++++++++++

Because the :class:`~crappy.blocks.camera_processes.CameraProcess` deals with
images, it can be interesting to have a real-time display of how the processing
is performing. To do so, the base CameraProcess class provides the
:meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw` method that
sends objects to the
:class:`~crappy.blocks.camera_processes.Displayer` Process to draw overlays on
top of the displayed images. This requires the
:py:`display_images` argument of the Camera Block is set to :obj:`True`.

The objects indicating what to draw should be children of the
:class:`~crappy.tool.camera_config.config_tools.Overlay` class. They only need
to define the :meth:`~crappy.tool.camera_config.config_tools.Overlay.draw`
method, that takes the image to display as an argument and draws the overlay on
top of it. Here is what it looks like for displaying a black ellipse:

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-29

To transmit the overlay to the Displayer Process, pass a collection of Overlay
instances to
:meth:`~crappy.blocks.camera_processes.CameraProcess.send_to_draw`. Crappy
provides the predefined
:class:`~crappy.tool.camera_config.config_tools.Box` Overlay. You can define
another Overlay by implementing its ``draw`` method. Here is the custom
CameraProcess from the
previous sub-section looks like after integrating the code for sending
overlays:

.. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
   :language: python
   :lines: 1-6, 31-61

6.d. Final runnable example
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
the detected eyes on the displayed images:

.. collapse:: (Expand to see the full code)

   .. literalinclude:: /downloads/complex_custom_objects/custom_camera_block.py
      :language: python

|

.. Note::
   To run this example, you'll need to have the *opencv-python* and *Pillow*
   Python modules installed.

This custom Camera Block script is based on an example that you can find in the
`custom objects examples folder on GitHub <https://github.com/
LaboratoireMecaniqueLille/crappy/tree/master/examples/custom_objects>`__. You
can :download:`download it
</downloads/complex_custom_objects/custom_camera_block.py>` to run it locally
on your machine. Note that the :py:`'Webcam'` camera is used here, so this
example will require a camera readable by OpenCV to be plugged to the computer.
This all-in-one extension path is one of the most advanced customization tasks
in Crappy, but remains useful when the processing should be tightly packaged
with acquisition, display, and recording. For new image pipelines whose stages
should be reusable or combined independently, prefer the custom VisionBlock
approach from the previous section.

7. Sharing custom objects and Blocks
------------------------------------

Custom objects can be shared as source files, packaged in a private
distribution, or contributed to Crappy.

For direct reuse, put custom objects in separate files with their required
imports and share
these files. Other people will be able to use them by importing your custom
objects in their script, e.g. with :py:`from file_name import CustomObject`. It
requires only one or a few files. Recipients can modify and redistribute those
files. The drawback is that the
versions of Crappy for the sender and the receiver might not be the same, in
which case the code might not run on the receiver's side. Also, for the
receiver, two steps are involved : installing Crappy and copying the sent
files.

Some users might want to distribute their work in a more rigid way, for example
an engineer distributing the same immutable code to several users of a machine.
It is possible to create and share installation files, or *wheels*, that
contain a modified version of Crappy and can be installed using
:mod:`pip`. To do so, one has to clone Crappy, i.e. make a local copy of its
source files, modify it to include the new custom objects, and build the wheel
to share. This way, everyone runs the same code, and also cannot have an
incompatible version since the version is fixed by the creator of the wheel.
How to properly modify a Python module to include new files is not described
here, neither is how to build and install a new wheel. This paragraph notes the
option. See the Python Packaging User Guide for build and
installation instructions.

To contribute an integration to Crappy, first *fork* the repository to your
GitHub account. After modifying this copy to include your own
files, you can submit a *pull request* to the maintainers to request
integration of your work on the official repository of Crappy. Again, this
paragraph is not a *git* or GitHub tutorial, and we're not going to give more
details about this process. The
:ref:`Developers information <developers:developers information>` page of the
documentation provides a few guidelines, as well as more insights on the
content of the module than the tutorials. If there's a feature you would like
to see in Crappy, but that you don't feel capable of implementing yourself, you
can also request improvements directly on GitHub.

For questions not covered by the tutorials, use the support channels listed on
the :ref:`Troubleshooting <troubleshooting:troubleshooting>` page. If you use
Crappy in an academic publication, follow the :ref:`citation guidance
<citing:citing crappy>`.
