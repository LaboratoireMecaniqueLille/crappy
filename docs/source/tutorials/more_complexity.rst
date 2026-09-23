.. _tutorial-advanced:

==================
Advanced tutorials
==================

These tutorials build on :doc:`getting_started`. Choose the task that matches
what you want to add to an experiment.

Choose a task
-------------

:doc:`feedback_loops`
  Make a measured value follow a target with a PID controller.

:doc:`modifiers`
  Transform measurements while they travel between Blocks.

:doc:`generator_conditions`
  Change a command when a measurement crosses a threshold.

:doc:`streaming_acquisition`
  Acquire and record measurements in chunks.

:doc:`organize_scripts`
  Use functions and loops to keep a larger test script readable.

:doc:`test_hardware_object`
  Check a Camera, InOut, or Actuator directly before using it in a test.

Each tutorial provides a complete downloadable example that works without
physical hardware, followed by the changes needed for real equipment.

Runtime details
---------------

Most test scripts only need :ref:`crappy.start()
<crappy_docs/aliases:crappy.start()>`. For its options and the way a test
starts and stops, see :doc:`../concepts/lifecycle_shutdown`. The
:doc:`../architecture` page describes the separate preparation, priority, and
launch stages for developers who need control between them. Exact signatures
are available in the :ref:`crappy_docs/aliases:aliases` API reference.

.. toctree::
   :hidden:

   feedback_loops
   modifiers
   generator_conditions
   streaming_acquisition
   organize_scripts
   test_hardware_object
