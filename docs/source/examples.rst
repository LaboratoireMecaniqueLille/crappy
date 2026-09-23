================
Examples by task
================

This page indexes the Python scripts distributed in the `examples directory
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples>`_.
Each link opens the source on GitHub. Read the module description at the top of
a script before running it: it states whether hardware or optional Python
packages are required.

If you are new to Crappy, follow the :doc:`tutorials` first. The examples below
are smaller demonstrations and complete setup scripts rather than guided
lessons.

Hardware-free examples
----------------------

These examples are convenient for exploring general Blocks and Modifiers.
Some require an optional plotting, storage, or networking package, but none
requires measurement hardware. Many image-processing and custom-integration
examples in later sections are hardware-free as well.

General Blocks
~~~~~~~~~~~~~~

- :example:`Button <blocks/button.py>` - emit a value from an on-screen button.
- :example:`Canvas <blocks/canvas.py>` - display values on a configurable
  drawing.
- :example:`Client <blocks/client_server/client.py>` and
  :example:`server <blocks/client_server/server.py>` - exchange data through
  an MQTT broker.
- :example:`Dashboard <blocks/dashboard.py>` - display the latest received
  values.
- :example:`Grapher <blocks/grapher.py>` - plot values while a test runs.
- :example:`HDF5 Recorder <blocks/hdf5_recorder.py>` - record streamed arrays
  in HDF5 format.
- :example:`Link Reader <blocks/link_reader.py>` - print all data received on
  a Link.
- :example:`Mean Block <blocks/mean.py>` - average values over time.
- :example:`Multiplexer <blocks/multiplexer.py>` - resample labels onto one
  time axis.
- :example:`Pause Block <blocks/pause_block.py>` - pause selected Blocks until
  a condition is met.
- :example:`Recorder <blocks/recorder.py>` - save labeled data in a text file.
- :example:`Sink <blocks/sink.py>` - consume data without producing output.
- :example:`Stop Block <blocks/stop_block.py>` - stop a test from a received
  value.
- :example:`Stop Button <blocks/stop_button.py>` - stop a test from a small
  window.
- :example:`Synchronizer <blocks/synchronizer.py>` - align data from two
  sources.

Modifiers
~~~~~~~~~

- :example:`Demux <modifiers/demux.py>` - points to the streaming acquisition
  example that demonstrates Demux.
- :example:`Differentiate <modifiers/differentiate.py>` - differentiate a
  signal.
- :example:`Downsampler <modifiers/downsampler.py>` - reduce a signal's sample
  rate.
- :example:`Integrate <modifiers/integrate.py>` - integrate a signal.
- :example:`Mean <modifiers/mean.py>` - average multiple values in each
  message.
- :example:`Median <modifiers/median.py>` - take the median of multiple values
  in each message.
- :example:`Moving average <modifiers/moving_avg.py>` - smooth data with a
  rolling average.
- :example:`Moving median <modifiers/moving_med.py>` - smooth data with a
  rolling median.
- :example:`Offset <modifiers/offset.py>` - subtract an initial value.
- :example:`Trigger on change <modifiers/trig_on_change.py>` - forward data
  when a label changes.
- :example:`Trigger on value <modifiers/trig_on_value.py>` - forward data when
  a label matches a target.

Acquisition and recording
-------------------------

Start with the simulated InOut examples, then adapt the same IOBlock patterns
to a driver from the :doc:`hardware` page or to your own driver.

- :example:`Basic IOBlock <blocks/ioblock/ioblock_basic.py>` - acquire values
  from a simulated InOut.
- :example:`Zero an input <blocks/ioblock/ioblock_make_zero.py>` - compensate
  an initial offset.
- :example:`Streaming acquisition <blocks/ioblock/ioblock_streamer.py>` -
  receive chunks of samples and demultiplex them.
- :example:`Triggered acquisition <blocks/ioblock/ioblock_trigger.py>` -
  control acquisitions with another signal.
- :example:`UController <blocks/ucontroller/ucontroller.py>` - communicate
  with a MicroPython board over USB.
- :example:`MicroPython companion <blocks/ucontroller/microcontroller.py>` -
  code to install on the board used by the UController example.
- :example:`Labjack T7 <other_examples/labjack_t7.py>` - read and drive T7
  digital channels.
- :example:`Labjack T7 streaming <other_examples/labjack_t7_stream.py>` -
  acquire and record T7 data at high rates.

Control and command generation
------------------------------

Command paths
~~~~~~~~~~~~~

- :example:`Basic Generator <blocks/generator/generator_basic.py>` - generate
  a simple command path.
- :example:`Complex path <blocks/generator/generator_complex_path.py>` -
  combine many path types.
- :example:`Custom condition
  <blocks/generator/generator_custom_condition.py>` - switch paths with a
  Python callable.
- :example:`Cycles <blocks/generator/generator_cycles.py>` - repeat groups of
  paths.
- :example:`Feedback condition
  <blocks/generator/generator_feedback_loop.py>` - change paths using a
  measured label.
- :example:`Safe start <blocks/generator/generator_safe_start.py>` - wait for
  the first feedback value before generating commands.

Actuator control and feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Basic Machine <blocks/machine/machine_basic.py>` - drive a
  simulated DC motor in speed.
- :example:`Multiple Actuators
  <blocks/machine/machine_multiple_actuators.py>` - drive two simulated
  actuators together.
- :example:`Position with a speed limit
  <blocks/machine/machine_speed.py>` - send position and speed commands to a
  simulated stepper motor.
- :example:`Fake Machine <blocks/fake_machine.py>` - simulate a tensile-test
  machine.
- :example:`PID <blocks/pid.py>` - close a feedback loop around a simulated
  motor.

Image acquisition and processing
--------------------------------

VisionBlock pipelines are the recommended starting point for new image work.
The all-in-one Camera Blocks remain supported for scripts that benefit from
their integrated design.

Composable VisionBlock pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Acquire and display <vision_blocks/camera_basic_display.py>` - the
  smallest CameraSource and ImageDisplayer pipeline.
- :example:`Acquire and record <vision_blocks/camera_basic_record.py>` - write
  images with an independent ImageRecorder.
- :example:`Software trigger <vision_blocks/camera_software_trigger.py>` -
  trigger CameraSource from regular Link data.
- :example:`Custom VisionBlocks <vision_blocks/custom_vision_blocks.py>` -
  implement a reusable image source and consumer.
- :example:`DIC video extensometry <vision_blocks/dic_ve.py>` - measure strain
  with DICVEProcessor.
- :example:`DIC VE and DIS correlation
  <vision_blocks/dic_ve_dis_correl.py>` - run two analyses on the same images.
- :example:`DIS correlation <vision_blocks/dis_correl.py>` - calculate custom
  displacement fields.
- :example:`Video extensometry <vision_blocks/video_extenso.py>` - track spots
  with VideoExtensoProcessor.

All-in-one Camera Blocks
~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Display images <blocks/camera/camera_basic_display.py>` - acquire
  and display simulated images.
- :example:`Record images <blocks/camera/camera_basic_record.py>` - acquire
  and save simulated images.
- :example:`Use a webcam <blocks/camera/camera_basic_webcam.py>` - display
  images from an OpenCV-compatible camera.
- :example:`Disable configuration <blocks/camera/camera_no_config.py>` - run
  acquisition without the configuration window.
- :example:`Report saved images
  <blocks/camera/camera_record_send_msg.py>` - send a regular message after
  recording an image.
- :example:`Software trigger <blocks/camera/camera_software_trigger.py>` -
  trigger integrated acquisition from regular Link data.
- :example:`DIC video extensometry <blocks/dic_ve/dic_ve_basic.py>` - run the
  DICVE Block interactively.
- :example:`DIC VE without configuration
  <blocks/dic_ve/dic_ve_no_config.py>` - provide patches directly.
- :example:`DIS correlation <blocks/dis_correl/dis_correl_basic.py>` - run the
  DISCorrel Block interactively.
- :example:`Custom DIS field
  <blocks/dis_correl/dis_correl_custom_field.py>` - provide displacement
  fields.
- :example:`DIS without configuration
  <blocks/dis_correl/dis_correl_no_config.py>` - provide the image patch
  directly.
- :example:`Video extensometry <blocks/video_extenso.py>` - track spots with
  the VideoExtenso Block.
- :example:`Automatic video-extenso drive
  <blocks/auto_drive_video_extenso.py>` - center tracked spots by driving a
  simulated motor.

Custom integrations
-------------------

Use these examples with :doc:`concepts/choosing_custom_object_type` and the
:doc:`custom-object tutorials <tutorials/custom_objects>`. For custom image
processing, start with the :example:`custom VisionBlocks example
<vision_blocks/custom_vision_blocks.py>` before considering an all-in-one
Camera Block subclass.

General and hardware-facing objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Custom Actuator <custom_objects/custom_actuator.py>` - implement a
  position-controlled Actuator.
- :example:`Custom Block <custom_objects/custom_block.py>` - implement a Block
  and its lifecycle methods.
- :example:`Custom Generator Path
  <custom_objects/custom_generator_path.py>` - implement reusable command
  logic.
- :example:`Input-only InOut
  <custom_objects/custom_inout/custom_inout_basic_in.py>` - implement regular
  acquisition.
- :example:`Input/output InOut
  <custom_objects/custom_inout/custom_inout_basic_inout.py>` - acquire values
  and receive commands.
- :example:`InOut zeroing
  <custom_objects/custom_inout/custom_inout_make_zero.py>` - customize offset
  compensation.
- :example:`Streaming InOut
  <custom_objects/custom_inout/custom_inout_streamer.py>` - implement chunked
  acquisition.
- :example:`Custom Modifier <custom_objects/custom_modifier.py>` - transform
  Link data with a callable object.

Camera objects and the all-in-one alternative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Basic Camera object
  <custom_objects/custom_camera/custom_camera_basic.py>` - implement image
  acquisition.
- :example:`Camera settings
  <custom_objects/custom_camera/custom_camera_settings.py>` - expose settings
  in the configuration window.
- :example:`Camera metadata
  <custom_objects/custom_camera/custom_camera_metadata.py>` - attach metadata
  to acquired images.
- :example:`Camera software ROI
  <custom_objects/custom_camera/custom_camera_software_roi.py>` - provide a
  configurable region of interest.
- :example:`Camera hardware trigger
  <custom_objects/custom_camera/custom_camera_hardware_trigger.py>` - expose a
  trigger setting.
- :example:`Custom all-in-one Camera Block
  <custom_objects/custom_camera_block.py>` - implement a Camera Block and its
  internal CameraProcess.

Complete setups
---------------

The fake tests demonstrate complete test graphs without laboratory hardware.
The real-setup scripts document specific machines and are starting points for
adaptation, not generic configurations.

Hardware-free test graphs
~~~~~~~~~~~~~~~~~~~~~~~~~

- :example:`Simulated tensile test <fake_tests/fake_test.py>` - coordinate
  command generation, simulation, display, and recording.
- :example:`Tensile test with DIC VE <fake_tests/dic_ve.py>` - add
  DIC-based video extensometry.
- :example:`Tensile test with DIS correlation <fake_tests/dis_correl.py>` -
  add full-field image correlation.
- :example:`Strain-controlled DIS test
  <fake_tests/dis_correl_strain_controlled.py>` - use image measurements in a
  feedback loop.
- :example:`Tensile test with video extensometry
  <fake_tests/video_extenso.py>` - add spot-tracking measurements.

Laboratory setups
~~~~~~~~~~~~~~~~~

- :example:`Biotens <real_setups_scripts/biotens.py>` - a starting script for
  the Biotens machine.
- :example:`Furnace <real_setups_scripts/furnace.py>` - control a
  solidification furnace.
- :example:`Instron tensile test
  <real_setups_scripts/tensile_instron.py>` - run a video-extensometry-driven
  tensile test.
