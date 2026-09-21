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

The :doc:`complex_custom_objects` guide currently covers streaming InOuts,
specialized Camera settings, VisionBlocks, and the supported all-in-one Camera
customization path. Read :doc:`../concepts/image_pipelines` before choosing an
image architecture.

.. toctree::
   :hidden:

   custom_modifier
   custom_actuator
   custom_inout
   custom_camera
   custom_block
   custom_generator_path
