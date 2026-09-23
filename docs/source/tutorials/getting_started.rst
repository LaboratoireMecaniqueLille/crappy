.. _tutorial-getting-started:

===============
Getting started
===============

Use this learning path to run a first test, then continue with the task that
matches what you want to do. Every example starts with simulated hardware and
states what must change before using real hardware.

Start here
----------

:doc:`quickstart`
  Run a complete three-second test without physical hardware, a graphical
  interface, or optional Python packages. This tutorial introduces Blocks,
  Links, and the labels carried by data.

Choose your next task
---------------------

:doc:`data_acquisition`
  Read simulated measurements with an IOBlock and display them in the terminal.

:doc:`data_recording`
  Save selected simulated measurements in a CSV file.

:doc:`signal_display`
  Plot a simulated measurement live with a Grapher.

:doc:`command_generation`
  Build a finite sequence of commands with a Generator.

:doc:`actuator_control`
  Send commands to a simulated Actuator through a Machine Block.

:doc:`image_pipeline`
  Acquire and display simulated images with a first VisionBlock pipeline.

Each tutorial provides a complete downloadable script and stops automatically.
Read its prerequisites and side-effects statement before running it.

Continue learning
-----------------

- See :doc:`../concepts/blocks_links_labels` for the shared ideas behind all
  Crappy tests.
- See :doc:`../concepts/lifecycle_shutdown` for the supported ways to stop a
  test and how cleanup works.
- Continue with :doc:`more_complexity` for feedback loops, Modifiers, streaming
  acquisition, and advanced Generator conditions.

.. toctree::
   :hidden:

   quickstart
   data_acquisition
   data_recording
   signal_display
   command_generation
   actuator_control
   image_pipeline
