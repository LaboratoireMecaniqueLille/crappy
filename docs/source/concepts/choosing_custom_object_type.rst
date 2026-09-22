.. _concepts-choosing-extension-point:

=============================
Choosing a custom object type
=============================

Use this page when Crappy's built-in objects do not cover your hardware or the
behavior required by your test. It helps you choose which Crappy base class to
subclass, or when a plain callable is sufficient, to add a custom device
driver, data transformation, command profile, or processing task.

This guide is specifically about integrating custom objects into Crappy for a
particular setup. It is not about configuring an existing object. Before
writing a new class, check the :doc:`../features` page and examples for an
existing object that already performs the task.

Decision guide
--------------

Follow the first matching branch to choose the type of custom object to create:

1. **Are you integrating a physical device?**

   - For a camera that acquires images, write a
     :class:`~crappy.camera.meta_camera.camera.Camera` object.
   - For a sensor, data-acquisition board, or general input/output device,
     write an :class:`~crappy.inout.meta_inout.inout.InOut` object.
   - For a motor, positioner, or other actuator that receives motion commands,
     write an :class:`~crappy.actuator.meta_actuator.actuator.Actuator` object.

2. **Are you adding an image-acquisition or image-processing stage?**

   - For a new composable stage, write a
     :class:`~crappy.blocks.vision.block.VisionBlock`.
   - Only when customizing the supported all-in-one Camera Block architecture,
     write a :class:`~crappy.blocks.camera_processes.CameraProcess`. This is an
     advanced customization path.

3. **Are you defining one command segment for a Generator?**

   Write a :class:`~crappy.blocks.generator_path.meta_path.path.Path`. A Path
   calculates command values and decides when its segment is complete. It does
   not replace the Generator Block that runs the sequence.

4. **Are you making a short transformation to each dictionary on one Link?**

   Write a :class:`~crappy.modifier.meta_modifier.modifier.Modifier`, or use a
   plain callable for a small script-specific transformation. A Modifier can
   rename, scale, filter, combine, or discard values without adding another
   Block.

5. **Does the task need independent repeated work or its own setup and
   cleanup?**

   Write a :class:`~crappy.blocks.meta_block.block.Block`. This is the general
   base class for a new acquisition, control, communication, calculation,
   display, or recording task that does not fit a more specific type above.

If a single proposed class appears to match several branches, separate its
responsibilities where practical. For example, put a hardware protocol in an
InOut and place the experiment-specific calculation in a Block or Modifier.
The hardware integration can then be reused without copying the calculation.

Custom object types compared
----------------------------

.. list-table:: Crappy custom object types
   :header-rows: 1
   :widths: 19 43 38

   * - Custom object
     - Use it for
     - Used by or connected through
   * - Actuator
     - A hardware driver that accepts speed, position, or other motion
       commands.
     - A :class:`~crappy.blocks.Machine` Block.
   * - InOut
     - A hardware driver that reads measurements, writes outputs, or does both.
     - An :class:`~crappy.blocks.IOBlock`.
   * - Camera object
     - A camera driver that exposes settings and returns acquired images.
     - CameraSource or an all-in-one Camera Block.
   * - Modifier
     - A quick transformation applied to each dictionary sent through one
       regular Link.
     - The ``modifier`` argument of :func:`crappy.link`.
   * - Generator Path
     - One reusable command profile or segment with a stop condition.
     - A :class:`~crappy.blocks.Generator` Block.
   * - Block
     - A complete non-image task with its own lifecycle and repeated work.
     - Regular Links to other Blocks.
   * - VisionBlock
     - A complete image source, processor, display, or recording stage.
     - ImageLinks for images and regular Links for labeled values.
   * - CameraProcess
     - Advanced processing inside a custom all-in-one Camera Block.
     - Managed internally by the owning Camera Block.

Hardware integrations and Blocks are different layers
------------------------------------------------------

Actuator, InOut, and Camera objects describe how to communicate with hardware.
They are owned by Blocks that place the hardware into a complete test:

- Machine owns one or more Actuator objects.
- IOBlock owns one InOut object.
- CameraSource or an all-in-one Camera Block owns one Camera object.

Keep the hardware class focused on operations that are meaningful for the
device. Link handling, experiment sequencing, displays, and file recording
normally belong to Blocks around it. This separation makes the same driver
usable in different tests.

Modifier or Block
-----------------

Choose a Modifier when the operation is naturally expressed as “for each
dictionary sent through this Link, transform or discard it.” A Modifier runs
as part of sending the data, so lengthy work also delays the source Block. It
does not have independent setup, timing, Links, or cleanup.

Choose a Block when the operation needs to run independently, combine several
inputs, publish to several consumers, open resources, control its own update
rate, or clean up when the test ends. The :doc:`lifecycle_shutdown` page
describes the hooks available to a custom Block.

Block or VisionBlock
--------------------

Choose a regular Block for labeled dictionaries. Choose a VisionBlock when the
stage receives or publishes image arrays through ImageLinks. A VisionBlock is
also a Block, so it has the same lifecycle and can use regular Links alongside
its ImageLinks.

VisionBlocks are recommended for new image pipelines. The all-in-one Camera
Blocks remain supported and are not planned for deprecation. Use the
:doc:`image_pipelines` comparison before choosing the advanced CameraProcess
customization path.

API and customization guides
----------------------------

- :doc:`../tutorials/custom_objects` links every custom-object guide.
- :doc:`../tutorials/custom_generator_path` covers Generator Paths.
- :doc:`../tutorials/custom_vision_block` covers reusable image stages.
- :doc:`../tutorials/custom_all_in_one_camera` covers advanced processing
  inside the supported all-in-one Camera Block.
- :doc:`../api` lists the public API for each custom object type.
