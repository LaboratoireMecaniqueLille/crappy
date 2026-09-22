.. _tutorial-custom-objects:

=====================
Create custom objects
=====================

Use these guides when Crappy's built-in objects do not support a device or
task required by an experiment. Before writing a new object, check
:doc:`../features` for an existing integration.

If you are unsure which guide applies, start with
:doc:`../concepts/choosing_custom_object_type`.

Integrate hardware
------------------

:doc:`custom_actuator`
  Drive a motor, positioner, or other device that receives motion commands.

:doc:`custom_inout`
  Acquire measurements, write outputs, or perform both operations with an
  unsupported instrument.

:doc:`custom_camera`
  Acquire images and expose adjustable settings for an unsupported camera.

Add experiment-specific behavior
--------------------------------

:doc:`custom_modifier`
  Apply a quick calculation or transformation to data on one Link.

:doc:`custom_block`
  Add a complete task with its own update rate, inputs, outputs, setup, or
  cleanup.

:doc:`custom_generator_path`
  Define one reusable command profile for a Generator.

Each guide provides a complete hardware-free example followed by the changes
needed for a real experiment. The API reference linked from each guide defines
the full supported interface.

Advanced customization
----------------------

:doc:`custom_streaming_inout`
  Retrieve high-rate measurements from a device in chunks.

:doc:`custom_vision_block`
  Add a reusable image source, analysis, or transformation stage.

:doc:`custom_all_in_one_camera`
  Add image processing inside the supported all-in-one Camera Block.

Read :doc:`../concepts/image_pipelines` before choosing an image architecture.

Reuse or contribute a custom object
-----------------------------------

Keep a reusable class in its own Python file. Another experiment can import it
with a statement such as ``from laboratory_devices import LoadCell`` before
creating the associated Block. For reuse across several computers, package the
classes as a normal Python distribution and declare the supported Crappy
version.

To propose an integration for Crappy itself, follow the contribution guidance
in :doc:`../developers`. Keep device-specific dependencies optional and include
tests that do not require access to the physical device.

.. toctree::
   :hidden:

   custom_modifier
   custom_actuator
   custom_inout
   custom_camera
   custom_block
   custom_generator_path
   custom_streaming_inout
   custom_vision_block
   custom_all_in_one_camera
