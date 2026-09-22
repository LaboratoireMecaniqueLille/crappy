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
   changes in published versions, or visit :doc:`support` for usage questions
   and bug reports.

Run a first test
----------------

:doc:`Install Crappy <installation>`, then follow the
:doc:`first-test tutorial <tutorials/quickstart>`. Its example uses simulated
hardware, prints measurements in the terminal, and stops automatically after a
few seconds.

Control hardware
----------------

Follow :doc:`the data-acquisition tutorial <tutorials/data_acquisition>` to
read measurements through an IOBlock. The
:doc:`data-recording tutorial <tutorials/data_recording>` saves selected
measurements in a CSV file. Follow
:doc:`the actuator tutorial <tutorials/actuator_control>` to drive an Actuator
through a Machine Block. The :doc:`features` page lists the integrations
distributed with Crappy.

Build an image pipeline
-----------------------

Read :doc:`concepts/image_pipelines` to choose an architecture, then follow the
:doc:`first image-pipeline tutorial <tutorials/image_pipeline>`. VisionBlocks
are recommended for new image pipelines. The all-in-one Camera Blocks remain
supported and are not planned for deprecation.

Adapt Crappy to a specific need
-------------------------------

Use :doc:`concepts/choosing_custom_object_type` to identify the right kind of
custom object for new hardware, data handling, commands, or image processing.
The :doc:`custom-object tutorials <tutorials/custom_objects>` then explain how
to implement and use it.

Find explanations and reference material
----------------------------------------

- :doc:`concepts` explains behavior shared by different tasks.
- :doc:`examples` indexes every distributed example by task.
- :doc:`api` documents public classes, functions, and arguments.
- :doc:`troubleshooting` explains common failures and recovery steps.
- :doc:`support` lists support channels and the information to include when
  reporting a problem.
- The `examples directory
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples>`_
  contains a collection of ready-to-use examples for Crappy.

.. toctree::
  :maxdepth: 2

  Is Crappy right for you? <what_is_crappy>
  Install Crappy <installation>
  Tutorials by task <tutorials>
  Examples by task <examples>
  Core concepts <concepts>
  Features and integrations <features>
  Hardware compatibility <hardware>
  API reference <api>
  Runtime architecture <architecture>
  Contributing <developers>
  Citing Crappy <citing>
  Troubleshooting <troubleshooting>
  Support and reporting <support>
