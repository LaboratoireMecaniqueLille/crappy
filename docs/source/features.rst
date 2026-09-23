=========================
Features and integrations
=========================

Crappy coordinates acquisition, control, processing, display, and recording
in Python test scripts. This page summarizes those capabilities and points to
the right level of documentation:

- Follow the :doc:`tutorials` to learn a workflow step by step.
- Browse :doc:`examples` to find a complete script for a particular task.
- Check :doc:`hardware` before choosing a Camera, InOut, or Actuator driver.
- Use the :doc:`api` for complete classes, arguments, methods, and return
  values.

Compose an experiment
---------------------

A test is assembled from **Blocks**, which perform tasks, and **Links**, which
carry labeled data between them. This makes it possible to combine simulated
objects, physical hardware, processing, and user interfaces in the same
script.

- :doc:`concepts/blocks_links_labels` introduces the shared vocabulary and
  shows how data moves through a test.
- :doc:`concepts/regular_links_and_image_links` explains when to use regular
  labeled data or image data.
- :doc:`concepts/lifecycle_shutdown` describes startup, execution, cleanup,
  and the supported ways to stop a test.
- The :doc:`quickstart tutorial <tutorials/quickstart>` builds and runs a
  complete hardware-free test.

Acquire data and control hardware
---------------------------------

- :ref:`IOBlock <crappy_docs/blocks:ioblock>` reads sensors and acquisition
  boards, sends commands to outputs, and supports both individual samples and
  streamed arrays. Start with the :doc:`data-acquisition tutorial
  <tutorials/data_acquisition>` or the :ref:`acquisition examples
  <examples:acquisition and recording>`.
- :ref:`Machine <crappy_docs/blocks:machine>` drives one or more Actuators in
  speed or position and can return their measured state. See the
  :doc:`actuator-control tutorial <tutorials/actuator_control>` and
  :ref:`control examples <examples:control and command generation>`.
- :ref:`Camera Source <crappy_docs/blocks:camera source>` acquires images from
  a Camera object for a composable image pipeline. The :doc:`image-pipeline
  tutorial <tutorials/image_pipeline>` provides a hardware-free introduction.
- :ref:`UController <crappy_docs/blocks:ucontroller>` exchanges commands and
  measurements with a microcontroller over a serial connection.

The :doc:`hardware compatibility matrix <hardware>` lists every distributed
Camera, InOut, and Actuator driver with its platform, connection, optional
dependencies, maintenance status, and verification status.

Generate commands and close feedback loops
------------------------------------------

- :ref:`Generator <crappy_docs/blocks:generator>` builds commands from
  reusable :doc:`Generator Paths <crappy_docs/generator_paths>`, including
  constants, ramps, cyclic paths, sine waves, and conditions based on measured
  values.
- :ref:`Button <crappy_docs/blocks:button>` lets an operator emit a value at a
  chosen moment.
- :ref:`PID <crappy_docs/blocks:pid>` calculates a command from a target and a
  measured value.
- :ref:`Auto Drive <crappy_docs/blocks:auto drive>` is specialized for moving
  a camera during video-extensometry tests.

See the :doc:`command-generation tutorial <tutorials/command_generation>`,
the :doc:`feedback-loop tutorial <tutorials/feedback_loops>`, and the
:ref:`command and control examples <examples:control and command generation>`.

Transform and coordinate measurements
-------------------------------------

Small transformations can be attached directly to a Link with a
:doc:`Modifier <crappy_docs/modifiers>`. Distributed Modifiers can demultiplex,
differentiate, integrate, downsample, offset, smooth, summarize, or
conditionally forward data. They can be chained, and custom callables are also
accepted.

For transformations that need their own state or timing:

- :ref:`Mean Block <crappy_docs/blocks:mean block>` averages each label over a
  chosen period.
- :ref:`Multiplexer <crappy_docs/blocks:multiplexer>` resamples labels onto a
  common time axis.
- :ref:`Synchronizer <crappy_docs/blocks:synchronizer>` preserves one label's
  samples while interpolating other labels onto its timestamps.

The :doc:`Modifier tutorial <tutorials/modifiers>` introduces transformations
in transit. Runnable scripts are indexed under :ref:`hardware-free examples
<examples:hardware-free examples>`.

Display and record results
--------------------------

- :ref:`Link Reader <crappy_docs/blocks:link reader>` prints received data for
  quick inspection.
- :ref:`Dashboard <crappy_docs/blocks:dashboard>` displays the latest values,
  while :ref:`Grapher <crappy_docs/blocks:grapher>` plots their history.
- :ref:`Canvas <crappy_docs/blocks:canvas>` places live values on a
  user-defined drawing.
- :ref:`Recorder <crappy_docs/blocks:recorder>` writes regular labeled data to
  a text file. :ref:`HDF Recorder <crappy_docs/blocks:hdf recorder>` stores
  arrays produced by streaming acquisition.
- :ref:`Image Displayer <crappy_docs/blocks:image displayer>` and
  :ref:`Image Recorder <crappy_docs/blocks:image recorder>` display or save
  images independently from acquisition and processing.

Use the :doc:`signal-display tutorial <tutorials/signal_display>`, the
:doc:`data-recording tutorial <tutorials/data_recording>`, or the
:ref:`hardware-free examples <examples:hardware-free examples>` as starting
points.

Build image pipelines
---------------------

VisionBlocks are recommended for new image pipelines. Acquisition, processing,
display, and recording are independent stages connected by ImageLinks, so one
image source can feed several consumers:

- :ref:`Camera Source <crappy_docs/blocks:camera source>` acquires images and
  metadata.
- :ref:`DIC VE Processor <crappy_docs/blocks:dic ve processor>` tracks
  textured patches and calculates displacement and strain.
- :ref:`DIS Correl Processor <crappy_docs/blocks:dis correl processor>`
  calculates full-field displacements on a selected image region.
- :ref:`Video Extenso Processor <crappy_docs/blocks:video extenso processor>`
  tracks contrasted spots and calculates strain.
- :ref:`Image Displayer <crappy_docs/blocks:image displayer>` and
  :ref:`Image Recorder <crappy_docs/blocks:image recorder>` consume images
  without being coupled to the acquisition rate.

The all-in-one :ref:`Camera Block <crappy_docs/blocks:camera block>` and its
DIC VE, DIS Correl, GPU, and Video Extenso variants remain supported and are
not planned for deprecation. They combine acquisition with processing,
display, or recording inside one Block.

:doc:`concepts/image_pipelines` compares both designs. The
:ref:`image examples <examples:image acquisition and processing>` list the
composable VisionBlock pipelines first and the all-in-one alternatives second.

Manage and connect tests
------------------------

- :ref:`Stop Block <crappy_docs/blocks:stop block>` stops a test when received
  data meets a condition, and :ref:`Stop Button <crappy_docs/blocks:stop
  button>` provides a manual graphical control.
- :ref:`Pause Block <crappy_docs/blocks:pause block>` pauses selected Blocks
  until a condition is met. Its API documentation describes the limitations
  to consider before using it.
- :ref:`Client Server <crappy_docs/blocks:client server>` exchanges data with
  other applications or computers through MQTT.
- :ref:`Fake Machine <crappy_docs/blocks:fake machine>` simulates a tensile
  test for development, while :ref:`Sink <crappy_docs/blocks:sink>` consumes
  otherwise unused data during prototyping.

See :doc:`concepts/lifecycle_shutdown` for test-wide behavior and
:ref:`complete setup examples <examples:complete setups>` for larger graphs.

Adapt Crappy to a specific setup
--------------------------------

Crappy can be customized when the distributed objects do not match a setup:

- Write a Modifier for a small transformation on Link data.
- Write an Actuator, InOut, or Camera object for a hardware protocol.
- Write a Block for an independent task with its own timing or state.
- Write a VisionBlock for reusable image acquisition or processing.
- Write a Generator Path for reusable command-generation logic.

:doc:`concepts/choosing_custom_object_type` helps select the appropriate
object. Continue with the :doc:`custom-object tutorials
<tutorials/custom_objects>` and :ref:`custom-integration examples
<examples:custom integrations>`.

The :doc:`driver collection <crappy_docs/collection>` retains additional
community-contributed hardware integrations for compatibility. Collection
drivers are separate from the maintained core API and are clearly identified
in the :doc:`hardware matrix <hardware>`.

Find complete object lists
--------------------------

The API reference is the canonical inventory and contains the detailed
configuration for every object:

- :doc:`Ordinary Blocks <crappy_docs/ordinary_blocks>`
- :doc:`VisionBlocks <crappy_docs/vision_blocks>`
- :doc:`Generator Paths <crappy_docs/generator_paths>`
- :doc:`Modifiers <crappy_docs/modifiers>`
- :doc:`Camera drivers <crappy_docs/cameras>`
- :doc:`InOut drivers <crappy_docs/inouts>`
- :doc:`Actuator drivers <crappy_docs/actuators>`
- :doc:`LaMcube-specific integrations <crappy_docs/lamcube>`
