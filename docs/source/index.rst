======
Crappy
======

Crappy is an open-source Python framework for command and data acquisition on
experimental setups. A Crappy test is a Python script assembled from
**Blocks** and **Links**. Each Block performs a task, such as acquiring a
measurement, driving an actuator, processing data, or saving results. A Link
carries labeled data from one Block to another.

If you are deciding whether Crappy suits your setup, start with
:doc:`what_is_crappy`. Otherwise, choose the path that matches what you want to
do.

.. note::

   This documentation describes Crappy |release|. See the `release notes on
   GitHub <https://github.com/LaboratoireMecaniqueLille/crappy/releases>`_ for
   changes in published versions, or visit :doc:`troubleshooting` for usage
   questions and bug reports.

Run a first test
----------------

:doc:`Install Crappy <installation>`, then follow the
:doc:`first-test tutorial <tutorials/getting_started>`. Its first example uses
simulated hardware and shows how to connect Blocks, display measurements, and
record them from one script.

Control hardware
----------------

Use an :ref:`IOBlock <tutorials/getting_started:2.e. the ioblock block>` to
acquire measurements or send outputs through an InOut. Use a
:ref:`Machine Block <tutorials/getting_started:2.f. the machine block>` to
drive one or more Actuators. The :doc:`features` page lists the integrations
distributed with Crappy.

Build an image pipeline
-----------------------

Read :doc:`concepts/image_pipelines` to choose an architecture, then follow the
:ref:`Camera acquisition and VisionBlocks tutorial
<tutorials/getting_started:2.b. camera acquisition and visionblocks>`. Separate
VisionBlocks are recommended for new image pipelines. The all-in-one Camera
Blocks remain supported and are not planned for deprecation.

Adapt Crappy to a specific need
-------------------------------

Use :doc:`concepts/choosing_custom_object_type` to identify the right kind of
custom object for new hardware, data handling, commands, or image processing.
The :doc:`custom-object tutorials <tutorials/custom_objects>` then explain how
to implement and use it.

Find explanations and reference material
----------------------------------------

- :doc:`concepts` explains behavior shared by different tasks.
- :doc:`api` documents public classes, functions, and arguments.
- :doc:`troubleshooting` lists support channels and the information to include
  when reporting a problem.
- The `examples directory
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples>`_
  contains runnable scripts for acquisition, control, and image processing.

.. toctree::
  :maxdepth: 2

  Is Crappy right for you? <what_is_crappy>
  Install Crappy <installation>
  Tutorials by task <tutorials>
  Core concepts <concepts>
  Features and integrations <features>
  API reference <api>
  Runtime architecture <architecture>
  Contributing <developers>
  Citing Crappy <citing>
  Troubleshooting and support <troubleshooting>
