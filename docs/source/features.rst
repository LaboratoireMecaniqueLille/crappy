=======================
Current functionalities
=======================

On this page are listed all the objects currently distributed with Crappy and
exposed to the users. Information on how to use them can be found in the
:ref:`Tutorials <tutorials:tutorials>`, as well as guidelines for creating your own objects. For most
Blocks, one or several directly runnable example scripts are available
in the `examples folder <https://github.com/LaboratoireMecaniqueLille/crappy/
tree/master/examples>`_ of the GitHub repository. For each object listed on
this page, you can click on its name to open the complete documentation given
in the :ref:`API <api:api>`.

Functionalities (Blocks)
------------------------

The Blocks are the base bricks of Crappy, that fulfill various functions. In
the tutorials, you can learn more about :ref:`how to use Blocks
<tutorials/getting_started:1. understanding crappy's syntax>` and :ref:`how to create new Blocks
<tutorials/custom_objects:5. custom blocks>`.

Image-handling Blocks are available in two complementary styles. The
:class:`~crappy.blocks.vision.VisionBlock` family separates image acquisition,
processing, display, and recording into independent Blocks connected by
:ref:`Image Links <crappy_docs/links:image link>`. This explicit architecture requires a few
more Blocks and connections to reproduce a simple all-in-one Camera workflow,
but it allows each image stream to be combined and fanned out much more freely.
Each VisionBlock also has fewer responsibilities and arguments, which makes
individual Blocks easier to use and custom image processing easier to add.
Users are encouraged to choose VisionBlocks for new scripts.

The existing :ref:`Camera Block <crappy_docs/blocks:camera block>` and its processing subclasses remain
available and are not planned for deprecation. They continue to provide a
convenient all-in-one interface when their fixed acquisition, processing,
display, and recording architecture matches the intended workflow.

Data display
++++++++++++

- :ref:`Canvas <crappy_docs/blocks:canvas>`

  Displays the data it receives on top of a static image, e.g. for having a
  real-time temperature map.

  The examples folder on GitHub contains `one example of the Canvas Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/canvas.py>`_.

- :ref:`Dashboard <crappy_docs/blocks:dashboard>`

  Prints the values it receives in a popup window with a nicer formatting than
  :ref:`Link Reader <crappy_docs/blocks:link reader>`. Unlike the :ref:`Grapher <crappy_docs/blocks:grapher>` Block, only the latest value is
  displayed for each label.

  The examples folder on GitHub contains `one example of the Dashboard Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/dashboard.py>`_.

- :ref:`Grapher <crappy_docs/blocks:grapher>`

  Plots real-time 2D graphs. It is possible to plot several datasets on a same
  graph. The *x* axis can be the time information, or any other label that the
  Grapher receives. Unlike the :ref:`Dashboard <crappy_docs/blocks:dashboard>` Block, the displayed data is
  persistent and displays the history of a label.

  The examples folder on GitHub contains `one example of the Grapher Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/dashboard.py>`_ specifically, but it is also used in most of the other
  examples.

  :ref:`A tutorial section <tutorials/getting_started:2.c. the grapher block>` is also dedicated to the
  Grapher Block.

- :ref:`Image Displayer <crappy_docs/blocks:image displayer>`

  Displays the newest image received from one VisionBlock through an
  :ref:`Image Link <crappy_docs/links:image link>`, with OpenCV or Matplotlib. It runs independently from
  acquisition and image processing, and can combine the image with overlays
  received through regular Links. A slow display can skip intermediate frames
  without slowing down the image source.

  The examples folder on GitHub contains a `basic CameraSource and
  ImageDisplayer pipeline <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/examples/vision_blocks/camera_basic_display.py>`_. The
  `combined DIC VE and DIS Correl example <https://github.com/
  LaboratoireMecaniqueLille/crappy/blob/master/examples/vision_blocks/
  dic_ve_dis_correl.py>`_ shows one displayer drawing overlays from multiple
  processors.

- :ref:`Link Reader <crappy_docs/blocks:link reader>`

  Prints the values it receives in the terminal. Mostly useful for debugging.

  The examples folder on GitHub contains `one example of the Link Reader Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/link_reader.py>`_.

Data recording
++++++++++++++

- :ref:`HDF Recorder <crappy_docs/blocks:hdf recorder>`

  Writes the data it receives to a *.hdf5* file. Only compatible with the
  :ref:`IOBlock <crappy_docs/blocks:ioblock>` in *streamer* mode. The :ref:`Recorder <crappy_docs/blocks:recorder>` should be used for
  recording any other type of data.

  The examples folder on GitHub contains `one example of the HDF Recorder Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/hdf5_recorder.py>`_.

- :ref:`Image Recorder <crappy_docs/blocks:image recorder>`

  Saves the newest images and their metadata received through one ImageLink.
  Recording runs independently from acquisition and can deliberately keep only
  one image out of a chosen number. SimpleITK, Pillow, OpenCV, and raw NumPy
  files are supported, and a regular Link can notify downstream Blocks after
  an image is actually saved.

  The examples folder on GitHub contains a `basic image-recording pipeline
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  vision_blocks/camera_basic_record.py>`_. The `combined processing,
  displaying and recording example <https://github.com/
  LaboratoireMecaniqueLille/crappy/blob/master/examples/vision_blocks/
  dic_ve_dis_correl.py>`_ demonstrates that these tasks can consume the same
  source independently.

- :ref:`Recorder <crappy_docs/blocks:recorder>`

  Writes the data it receives to a *.csv* file, for recording it. It is
  compatible with data from any Block, except data coming from an
  :ref:`IOBlock <crappy_docs/blocks:ioblock>` in *streamer* mode. The :ref:`HDF Recorder <crappy_docs/blocks:hdf recorder>` should be used
  instead in this situation.

  The examples folder on GitHub contains `one example of the Recorder Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/recorder.py>`_.

  :ref:`A tutorial section <tutorials/getting_started:2.d. the recorder block>` is also dedicated to the
  Recorder Block.

Data processing
+++++++++++++++

- :ref:`Mean <crappy_docs/blocks:mean block>`

  Calculates the average of the received labels over a given period, and sends
  it to downstream Blocks. One average value is given for each label, it is not
  meant to average several labels together. Can be used as a less
  computationally-intensive :ref:`Multiplexer <crappy_docs/blocks:multiplexer>`.

  The examples folder on GitHub contains `one example of the Mean Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/mean.py>`_.

- :ref:`Multiplexer <crappy_docs/blocks:multiplexer>`

  Synchronizes labels emitted at different frequencies onto one time base.
  Useful for plotting curves out of two labels from different Blocks with a
  :ref:`Grapher <crappy_docs/blocks:grapher>`, as the timestamps of the data points would otherwise never
  match. Also used before saving data with a :ref:`Recorder <crappy_docs/blocks:recorder>` to simplify the
  post-processing.

  The examples folder on GitHub contains `one example of the Multiplexer Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/multiplexer.py>`_.

- :ref:`Synchronizer <crappy_docs/blocks:synchronizer>`

  Allows putting labels emitted at different frequencies on the same time base
  as a reference label. Very similar to the :ref:`Multiplexer <crappy_docs/blocks:multiplexer>` Block, except
  the :ref:`Multiplexer <crappy_docs/blocks:multiplexer>` takes an independent time base for interpolation. Used
  when the original values of a label need to be preserved while the other
  labels can be interpolated, for example when the reference is the output of a
  low-frequency image-processing.

Real-time image correlation
+++++++++++++++++++++++++++

- :ref:`DIS Correl Processor <crappy_docs/blocks:dis correl processor>`

  Receives images from a VisionBlock and performs real-time Dense Inverse
  Search (DIS) image correlation on one selected patch. It projects the
  displacement field onto predefined or custom fields and sends the results
  through regular Links. Acquisition, display, and recording can be connected
  independently according to the needs of the script.

  The examples folder on GitHub contains a `non-interactive DIS Correl
  Processor example <https://github.com/LaboratoireMecaniqueLille/crappy/blob/
  master/examples/vision_blocks/dis_correl.py>`_ and an example `combining it
  with DIC VE processing, display, and recording <https://github.com/
  LaboratoireMecaniqueLille/crappy/blob/master/examples/vision_blocks/
  dic_ve_dis_correl.py>`_.

- :ref:`DIS Correl <crappy_docs/blocks:dis correl>`

  Child of the :ref:`Camera <crappy_docs/cameras:camera>` Block that can acquire, record and display images.
  In addition, it performs real-time Dense Inverse Search (DIS) image
  correlation on the acquired images using :mod:`cv2`'s `DISOpticalFlow`, and
  projects the displacement field on a predefined basis. The result is then
  sent to downstream Blocks.

  The examples folder on GitHub contains `several examples of the DIS Correl
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
  examples/blocks/dis_correl>`_.

  This all-in-one Block remains supported. For new scripts, the :ref:`DIS
  Correl Processor <crappy_docs/blocks:dis correl processor>` offers the same kind of processing in the more flexible
  VisionBlock architecture.

- :ref:`GPU Correl <crappy_docs/blocks:gpu correl>`

  Same as :ref:`DIS Correl <crappy_docs/blocks:dis correl>`, except the computation is performed on a
  Cuda-compatible GPU.

  There is currently no example featuring this Block distributed in the
  examples folder on GitHub.

  .. Important::
     This Block has not been maintained or tested recently. Its current
     behavior is not verified. A maintained implementation should replace it.

  .. Warning::
     This Block cannot run with CUDA versions greater than 11.3. This is due
     to a deprecation in pycuda, and is unlikely to be fixed anytime soon in
     Crappy or pycuda.

Video-extensometry
++++++++++++++++++

- :ref:`Auto Drive <crappy_docs/blocks:auto drive>`

  This Block drives an :ref:`Actuator <crappy_docs/actuators:actuator>`, just like the :ref:`Machine <crappy_docs/blocks:machine>` Block.
  However, it does it in a very specific context. It allows moving a
  :ref:`Camera <crappy_docs/cameras:camera>` performing video-extensometry and mounted on an
  :ref:`Actuator <crappy_docs/actuators:actuator>`, so that the barycenter of the tracked dots remains in the
  center of the image. To do so, it takes the output of a :ref:`Video Extenso <crappy_docs/blocks:video extenso>`
  Block as its input.

  The examples folder on GitHub contains `one example of the Auto Drive Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/auto_drive_video_extenso.py>`_.

- :ref:`DIC VE <crappy_docs/blocks:dic ve>`

  Child of the :ref:`Camera <crappy_docs/cameras:camera>` Block that can acquire, record and display images.
  In addition, it performs image correlation on four patches on the acquired
  images. From the correlation, it deduces the *x* and *y* displacement of each
  patch, and can then calculate the global strain on the filmed sample. The
  displacements and the strain values are sent to downstream Block. Can be used
  to replace :ref:`Video Extenso <crappy_docs/blocks:video extenso>` on samples with a speckle drawn on them, each
  patch playing the same role as a dot.

  The examples folder on GitHub contains `several examples of the DIC VE Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/dic_ve>`_.

  This all-in-one Block remains supported. For new scripts, the :ref:`DIC VE
  Processor <crappy_docs/blocks:dic ve processor>` separates this processing from acquisition, display, and
  recording.

- :ref:`DIC VE Processor <crappy_docs/blocks:dic ve processor>`

  Tracks between one and four textured patches in images received from a
  VisionBlock, and calculates their displacement and the sample strain. Patch
  selection can be requested from the upstream image source or provided
  directly. Measurements and patch overlays are published through regular
  Links, leaving the user free to connect display and recording independently.

  The examples folder on GitHub contains a `DIC VE Processor pipeline
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  vision_blocks/dic_ve.py>`_ and a `combined DIC VE and DIS Correl pipeline
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  vision_blocks/dic_ve_dis_correl.py>`_.

- :ref:`GPU VE <crappy_docs/blocks:gpu ve>`

  Same as :ref:`DIC VE <crappy_docs/blocks:dic ve>`, except the computation is done on a Cuda-compatible
  GPU.

  There is currently no example featuring this Block distributed in the
  examples folder on GitHub.

  .. Important::
     This Block has not been maintained or tested recently. Its current
     behavior is not verified. A maintained implementation should replace it.

  .. Warning::
     This Block cannot run with CUDA versions greater than 11.3. This is due
     to a deprecation in pycuda, and is unlikely to be fixed anytime soon in
     Crappy or pycuda.

- :ref:`Video Extenso <crappy_docs/blocks:video extenso>`

  Child of the :ref:`Camera <crappy_docs/cameras:camera>` Block that can acquire, record and display images.
  In addition, it performs real-time video-extensometry on the acquired images.
  It can track from two to four spots drawn on the filmed sample, and tracks
  the position of each spot to get their displacement. Based on the
  displacements, a global *x* and *y* strain values are computed. The strain
  and the displacement values are sent to downstream Blocks. The :ref:`DIC VE <crappy_docs/blocks:dic ve>`
  Block performs a similar task but uses image correlation for tracking the
  areas.

  The examples folder on GitHub contains `one example of the Video Extenso
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/
  examples/blocks/video_extenso.py>`_.

  This all-in-one Block remains supported. For new scripts, the :ref:`Video
  Extenso Processor <crappy_docs/blocks:video extenso processor>` provides the tracking stage as an independent
  VisionBlock.

- :ref:`Video Extenso Processor <crappy_docs/blocks:video extenso processor>`

  Tracks up to four contrasted spots in images received through an ImageLink,
  and sends their positions and the calculated strain through regular Links.
  Spot selection is performed in a specialized configuration window requested
  from the upstream image source. Acquisition and optional display or
  recording remain independent Blocks.

  The examples folder on GitHub contains a `complete Video Extenso Processor
  pipeline <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/
  examples/vision_blocks/video_extenso.py>`_.

Signal generation
+++++++++++++++++

- :ref:`Button <crappy_docs/blocks:button>`

  Creates a small graphical window with a button in it, and generates a signal
  when the user clicks on the button. This signal is sent to downstream Blocks.
  Useful for triggering a behavior at a user-chosen moment during a test.

  The examples folder on GitHub contains `one example of the Button Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/button.py>`_.

- :ref:`Generator <crappy_docs/blocks:generator>`

  Generates a signal following a pattern given by the user (like sine waves,
  triangles, squares, etc.), and sends this signal to downstream Blocks. It
  can only output a combination of :ref:`Generator Paths <crappy_docs/blocks:generator paths>`.

  The examples folder on GitHub contains `several examples of the Generator
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
  examples/blocks/generator>`_ specifically, but it is also used in many of the
  other examples.

  :ref:`A tutorial section <tutorials/getting_started:2.a. the generator block and its paths>` is also
  dedicated to the Generator Block, and :ref:`another one
  <tutorials/complex_custom_objects:1. custom generator paths>` is dedicated to the creation of custom Generator
  Paths.

- :ref:`PID <crappy_docs/blocks:pid>`

  Takes a setpoint target as an input, as well as an actual measured value.
  Then, calculates a command value following a PID controller logic, and sends
  it to downstream Blocks (usually to the actuator that drives the system on
  which the measured value is acquired). Useful for driving a system whose
  exact characteristics are unknown or can vary.

  The examples folder on GitHub contains `one example of the PID Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/pid.py>`_.

Hardware control
++++++++++++++++

- :ref:`Camera Source <crappy_docs/blocks:camera source>`

  Drives one :ref:`Camera <crappy_docs/cameras:camera>` object and publishes the acquired images and their
  metadata through one or more ImageLinks. It is intentionally limited to
  acquisition, so independent processors, displayers, and recorders can all
  consume the same image stream at their own frequencies. It can also serve
  specialized configuration requests from downstream processing Blocks.

  The examples folder on GitHub contains a `basic display pipeline
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  vision_blocks/camera_basic_display.py>`_, a `recording pipeline
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  vision_blocks/camera_basic_record.py>`_, and a `software-triggered
  acquisition example <https://github.com/LaboratoireMecaniqueLille/crappy/
  blob/master/examples/vision_blocks/camera_software_trigger.py>`_.

- :ref:`Camera <crappy_docs/blocks:camera block>`

  Acquires images from a :ref:`Camera <crappy_docs/cameras:camera>` object, and then displays and/or records
  the acquired images. It is the base class for other Blocks that can also
  perform image processing, in addition to the recording and display. This
  Block usually doesn't have input nor output Links, but can in some specific
  situations.

  The examples folder on GitHub contains `several examples of the Camera Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/camera>`_.

  This Block remains supported and is not planned for deprecation. New scripts
  are encouraged to use a :ref:`Camera Source <crappy_docs/blocks:camera source>` connected to the desired
  VisionBlocks when a more flexible image architecture is useful.

  :ref:`A tutorial section <tutorials/getting_started:2.b. camera acquisition and visionblocks>` is also
  dedicated to camera acquisition, and :ref:`another one <tutorials/custom_objects:4. custom cameras>`
  is dedicated to the creation of custom Camera objects.

- :ref:`IOBlock <crappy_docs/blocks:ioblock>`

  Controls one :ref:`InOut <crappy_docs/inouts:in / out>` object, allowing to read data from
  sensors and/or to give it commands to set on hardware. It is originally
  intended for interfacing with DAQ boards, but can also be used to drive a
  variety of other devices. It has output Links when acquiring data, and input
  Links when setting commands.

  The examples folder on GitHub contains `several examples of the IOBlock Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/ioblock>`_.

  :ref:`A tutorial section <tutorials/getting_started:2.e. the ioblock block>` is also dedicated to the
  IOBlock Block, and :ref:`another one <tutorials/custom_objects:3. custom inouts>` is dedicated to the
  creation of custom InOut objects.

- :ref:`Machine <crappy_docs/blocks:machine>`

  Drives one or several :ref:`Actuator <crappy_docs/actuators:actuator>` in speed or in position, based on the
  received command labels. Can also acquire the current speed and/or position
  from the driven Actuators, and return it to the downstream Blocks. This Block
  is intended for driving motors and similar devices.

  The examples folder on GitHub contains `several examples of the Machine Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/machine>`_.

  :ref:`A tutorial section <tutorials/getting_started:2.f. the machine block>` is also dedicated to the
  Machine Block, and :ref:`another one <tutorials/custom_objects:2. custom actuators>` is dedicated to
  the creation of custom Actuator objects.

- :ref:`UController <crappy_docs/blocks:ucontroller>`

  Controls a microcontroller over serial. :ref:`A MicroPython and an Arduino
  template <crappy_docs/tools:microcontroller templates>` to use along with this Block are
  provided with Crappy. This Block can start or stop the script on the
  microcontroller, send commands, and receive data.

  The examples folder on GitHub contains `one example of the UController Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/ucontroller>`_.

Test management
+++++++++++++++

- :ref:`Pause Block <crappy_docs/blocks:pause block>`

  Pauses the current Crappy script if the received data meets one of the given
  criteria. Useful when human intervention on hardware is needed during a test,
  but has some strong limitations. Refer to the documentation of this Block for
  more details.

  The examples folder on GitHub contains `one example of the Pause Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/pause_block.py>`_.

- :ref:`Stop Block <crappy_docs/blocks:stop block>`

  Stops the current Crappy script if the received data meets one of the given
  criteria. One of the clean ways to stop a script in Crappy.

  The examples folder on GitHub contains `one example of the Stop Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/stop_block.py>`_.

  Refer to the :ref:`dedicated tutorial section
  <tutorials/getting_started:3. properly stopping a script>` to learn more about how to properly stop a
  script in Crappy.

- :ref:`Stop Button <crappy_docs/blocks:stop button>`

  Stops the current Crappy script when the user clicks on a button in a GUI.
  One of the clean ways to stop a script in Crappy.

  The examples folder on GitHub contains `one example of the Stop Button Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/stop_button.py>`_.

  Refer to the :ref:`dedicated tutorial section
  <tutorials/getting_started:3. properly stopping a script>` to learn more about how to properly stop a
  script in Crappy.

Others
++++++

- :ref:`Client Server <crappy_docs/blocks:client server>`

  Sends and/or receives data over a local network via an MQTT server. Can also
  start a `Mosquitto <https://mosquitto.org/>`_ MQTT broker. Used for
  communicating with distant devices over a network, e.g. for remotely
  controlling a test.

  The examples folder on GitHub contains `one example of the ClientServer Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/client_server>`_.

- :ref:`Fake Machine <crappy_docs/blocks:fake machine>`

  Emulates the behavior of a tensile test machine, taking a position command as
  input and outputting the force and the displacement. Mainly used in the
  examples because it doesn't require any hardware, but may as well be used for
  debugging or prototyping.

  The examples folder on GitHub contains `one example of the Fake Machine Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/fake_machine.py>`_.

- :ref:`Sink <crappy_docs/blocks:sink>`

  Discards any received data. Used for prototyping and debugging only.

  The examples folder on GitHub contains `one example of the Sink Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/sink.py>`_.

Supported hardware (Cameras, InOuts, Actuators)
-----------------------------------------------

Each hardware category below separates the drivers maintained with Crappy from
the community-driven :ref:`Driver collection <crappy_docs/collection:driver collection>`.

Supported Cameras
+++++++++++++++++

- :ref:`Basler Ironman Camera Link <crappy_docs/cameras:basler ironman camera link>`

  Allows reading images from a camera communicating over Camera Link plugged to
  a `microEnable 5 ironman AD8-PoCL <https://www.baslerweb.com/en/
  acquisition-cards/frame-grabbers/>`_ PCIexpress board. May as well work with
  similar boards.

  .. Important::
     This Camera object relies on C++ libraries, which are not distributed with
     :mod:`pip`. They can only be used after a local install, after cloning the
     repo.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified, and it is a candidate for removal.

- :ref:`Camera GStreamer <crappy_docs/cameras:camera gstreamer>`

  This Camera object opens video streams using the Python binding of `Gstreamer
  <https://gstreamer.freedesktop.org/>`_. It can open a camera by path (in
  Linux) or number (in Windows and Mac), in which case the GStreamer pipeline
  is generated automatically. Alternatively, it can also open a stream
  following a custom pipeline given by the user.

  Compared to the :ref:`Camera OpenCV <crappy_docs/cameras:camera opencv>` camera, the GStreamer one is less
  CPU-intensive and is compatible with more devices. Its dependencies are
  however harder to install (especially on Windows) and it is harder to make it
  work properly.

  .. Important::
     Full functionality requires Linux and the *v4l-utils* package.

- :ref:`Camera OpenCV <crappy_docs/cameras:camera opencv>`

  This Camera object opens video streams using OpenCV. It allows tuning the
  device number, as well as the image format and the number of channels. It is
  compatible with many USB cameras.

  .. Important::
     Full functionality requires Linux and the *v4l-utils* package.

- :ref:`Fake Camera <crappy_docs/cameras:fake camera>`

  Simply displays an animated image of a chosen size and at a given frequency.
  Doesn't require any hardware, used mainly for debugging and prototyping.

- :ref:`File Reader <crappy_docs/cameras:file reader>`

  Successively reads images already saved in a folder, and returns them as if
  they just had been acquired by a real camera. No real image acquisition is
  performed though, and no hardware is required.

- :ref:`JAI GO-5000C-PMCL <crappy_docs/cameras:jai go-5000c-pmcl>`

  Allows reading images from a `Jai GO-5000M-PMCL <https://www.jai.com/
  products/go-5000c-pmcl>`_ camera. It relies on the :ref:`Basler Ironman
  Camera Link <crappy_docs/cameras:basler ironman camera link>` object.

  .. Important::
     This Camera object relies on C++ libraries, which are not distributed with
     :mod:`pip`. They can only be used after a local install, after cloning the
     repo.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified, and it is a candidate for removal.

- :ref:`Raspberry Pi Camera 2 <crappy_docs/cameras:raspberry pi camera 2>`

  Allows reading images from an official Raspberry Pi Camera, with Crappy
  running on a Raspberry Pi. It has been tested on Raspberry Pi 4 and 5, with
  Raspberry Pi camera HQ and V3 models. Probably works with other official and
  unofficial Raspberry Pi cameras.

  .. Note::
     This Camera is an updated version of :ref:`Raspberry Pi Camera <crappy_docs/cameras:raspberry pi camera>`, which is
     now deprecated and should only be used for compatibility with old
     Raspberry Pi OS versions.

- :ref:`Webcam <crappy_docs/cameras:webcam>`

  Reads images from a video device recognized by OpenCV. Usually webcams fall
  into this category, but some other cameras as well. This class is really
  basic and is intended for demonstration, see :ref:`Camera OpenCV <crappy_docs/cameras:camera opencv>` and
  :ref:`Camera GStreamer <crappy_docs/cameras:camera gstreamer>` for classes providing a finer controls over the
  devices.

- :ref:`Xi API <crappy_docs/cameras:xi api>`

  Allows reading images from any `Ximea <https://www.ximea.com/>`_ camera. The
  backend is the official Ximea API.

Collection Camera drivers
"""""""""""""""""""""""""

The following drivers are not actively maintained and require an explicit
``import crappy.collection`` before they can be selected by name.

- :ref:`Camera gPhoto2 <crappy_docs/cameras:camera gphoto2>`

  Reads images over USB from a camera supported by gPhoto2, including most of
  the Canon and Nikon models. It can either acquire images continuously, or
  wait for an acquisition to be triggered via a remote controller button.

  .. Important::
     This class was only tested on Linux. The installation of its dependencies
     is expected to be troublesome on macOS and Windows.

  .. Important::
     This class relies on the `gphoto2 <https://pypi.org/project/gphoto2/>`_
     Python module, that must be installed before using it.

- :ref:`Raspberry Pi Camera <crappy_docs/cameras:raspberry pi camera>`

  Allows reading images from a Raspberry Pi Camera, with Crappy running on a
  Raspberry Pi. It has been tested on Raspberry Pi 3 and 4, with a variety of
  official Raspberry Pi cameras.

  .. Warning::
     This Camera object is deprecated, and :ref:`Raspberry Pi Camera 2 <crappy_docs/cameras:raspberry pi camera 2>` should
     be used instead. It is only kept for compatibility with older OS versions.

  .. Important::
     This object requires a Raspberry Pi and is currently compatible only
     with the *Buster* version of Raspberry Pi OS, or with *Bullseye* in legacy
     camera mode.

- :ref:`Seek Thermal Pro <crappy_docs/cameras:seek thermal pro>`

  Allows reading images from a Seek Thermal `Compact Pro <https://
  www.thermal.com/compact-series-cameras.html>`_ infrared camera.

Supported Actuators
+++++++++++++++++++


- :ref:`Fake DC Motor <crappy_docs/actuators:fake dc motor>`

  Emulates the dynamic behavior of a DC motor, but doesn't drive any hardware.
  Used in the examples, may also be used for prototyping or debugging.

- :ref:`Fake Stepper Motor <crappy_docs/actuators:fake stepper motor>`

  Emulates the dynamic behavior of a stepper motor used as a linear actuator,
  but does not drive any actual hardware. It is used in examples, and can also
  be used for debugging. Unlike the :ref:`Fake DC Motor <crappy_docs/actuators:fake dc motor>`, it can drive the
  motor in position.

- :ref:`JVL Mac140 <crappy_docs/actuators:jvl mac140>`

  Drives JVL's `MAC140 <https://www.jvl.dk/276/integrated-servo-motors-mac050
  -141>`_ integrated servomotor in speed or in position. Probably works with
  other integrated servomotors from JVL, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Kollmorgen ServoStar 300 <crappy_docs/actuators:kollmorgen servostar 300>`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in position or sets it to the
  analog driving mode. This is the same conditioner as for the :ref:`Biaxe <crappy_docs/lamcube:biaxe>`
  Actuator, but this object was designed for another application.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Phidget Stepper4A <crappy_docs/actuators:phidget stepper4a>`

  Drives 4A bipolar stepper motors using Phidget's `Stepper4A <https://
  www.phidgets.com/?prodid=1278>`_ in speed or in position, by using several
  Phidget libraries.

  .. Important::
     This Actuator must be connected to Phidget's VINT Hub to work. See the
     following link `<https://www.phidgets.com/?prodid=1278#Tab_User_Guide>`_
     to connect properly to the Hub.

- :ref:`Pololu Tic <crappy_docs/actuators:pololu tic>`

  Drives Pololu's `Tic <https://www.pololu.com/category/212/tic-stepper-motor-
  controllers>`_ stepper motor drivers in speed or in position. Designed for
  driving all the Tic drivers, but tested only on the 36v4 model.

Collection Actuator drivers
"""""""""""""""""""""""""""

The following drivers are not actively maintained and require an explicit
``import crappy.collection`` before they can be selected by name.

- :ref:`Adafruit DC Motor Hat <crappy_docs/actuators:adafruit dc motor hat>`

  Drives up to 4 DC motors using Adafruit's `DC & Stepper Motor HAT for
  Raspberry Pi <https://www.adafruit.com/product/2348>`_, using either
  Adafruit's Blinka library or :mod:`smbus2` if driven from a Raspberry Pi.
  Although this component can also drive stepper motors, this feature was not
  implemented.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Newport TRA6PPD <crappy_docs/actuators:newport tra6ppd>`

  Drives Newport's `TRA6PPD <https://www.newport.com/p/TRA6PPD>`_ miniature
  linear stepper motor actuator, in speed or in position.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Oriental ARD-K <crappy_docs/actuators:oriental ard-k>`

  Drives Oriental Motor's `ARD-K <https://catalog.orientalmotor.com/item/s-
  closed-loop-stepper-motor-drivers-dc-input/ard-closed-loop-stepper-driver-
  pulse-input-dc/ard-k>`_ stepper motor driver in speed or in position.
  Probably works with other stepper motor drivers in the same range of
  products, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Schneider MDrive 23 <crappy_docs/actuators:schneider mdrive 23>`

  Drives Schneider Electric's `MDrive 23 <https://www.novantaims.com/downloads
  /quickreference/mdi23plus_qr.pdf>`_ stepper motor in speed or in position.
  Probably works with other stepper motors in the same range of products,
  although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

Supported Sensors and outputs
+++++++++++++++++++++++++++++

Acquisition boards
""""""""""""""""""

- :ref:`Labjack T7 <crappy_docs/inouts:labjack t7>`

  Controls Labjack's `T7 <https://labjack.com/products/labjack-t7>`_
  acquisition board. It can acquire data from its ADCs, set the output of DACs,
  read and set the GPIOs, and also supports more advanced functions like
  reading thermocouples.

- :ref:`Labjack T7 Streamer <crappy_docs/inouts:labjack t7 streamer>`

  Controls Labjack's `T7 <https://labjack.com/products/labjack-t7>`_
  acquisition board in streaming mode. In this mode, it can only acquire data
  from the ADCs and does not support any other function.

Sensors
"""""""

- :ref:`ADS1115 <crappy_docs/inouts:ads1115>`

  Reads voltages from Adafruit's `ADS 1115 <https://www.adafruit.com/product/
  1085>`_ ADC. Communicates over I2C.

- :ref:`Fake Inout <crappy_docs/inouts:fake inout>`

  Can acquire the current RAM usage of the computer using the :mod:`psutil`
  module, and also instantiate useless objects to reach a target memory usage
  (if superior to the base memory usage). It supports the streamer mode for the
  data acquisition. Mainly intended for demonstration, and used in the
  distributed examples.

- :ref:`MPRLS <crappy_docs/inouts:mprls>`

  Reads pressures from Adafruit's `MPRLS <https://www.adafruit.com/product/
  3965>`_ pressure sensor. Communicates over I2C.

- :ref:`NAU7802 <crappy_docs/inouts:nau7802>`

  Reads voltages from SparkFun's `'Qwiic Scale' NAU7802 <https://www.sparkfun.
  com/products/15242>`_ load cell conditioner. Communicates over I2C.

- :ref:`Phidget Wheatstone Bridge <crappy_docs/inouts:phidget wheatstone bridge>`

  Reads voltages from Phidget's `Wheatstone Bridge <https://www.phidgets.com/
  ?prodid=957>`_ load cell conditioner, by using several Phidget libraries.

  .. Important::
     This InOut must be connected to Phidget's VINT Hub to work. See the
     following link `<https://www.phidgets.com/?prodid=957#Tab_User_Guide>`_ to
     connect properly to the Hub.

Multi-device drivers
""""""""""""""""""""

- :ref:`DAQmx <crappy_docs/inouts:daqmx>`

  Same as :ref:`NI DAQmx <crappy_docs/inouts:ni daqmx>`, except it relies on the :mod:`PyDAQmx` module. The
  differences between the two modules weren't further investigated.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`NI DAQmx <crappy_docs/inouts:ni daqmx>`

  Controls National Instrument's `USB 6008 <https://www.ni.com/en-us/support/
  model.usb-6008.html>`_ DAQ module using the :mod:`nidaqmx` module. The code
  was written to work as-is on other National Instruments acquisition modules,
  but this hasn't been tested. Communicates over USB.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

Outputs
"""""""

- :ref:`GPIO PWM <crappy_docs/inouts:gpio pwm>`

  Controls a PWM output on a single GPIO of a Raspberry Pi.

  .. Important:: This object requires a Raspberry Pi. It was tested on Raspberry Pi 3 and 4,
     with the *Buster* and *Bullseye* Raspberry Pi Os for the latter.

- :ref:`GPIO Switch <crappy_docs/inouts:gpio switch>`

  Drives a single GPIO on a Raspberry Pi, or any other board supporting Blinka.

  .. Important:: This object requires a Raspberry Pi. It was tested on Raspberry Pi 3 and 4,
     with the *Buster* and *Bullseye* Raspberry Pi Os for the latter.

Collection InOut drivers
""""""""""""""""""""""""

The following drivers are not actively maintained and require an explicit
``import crappy.collection`` before they can be selected by name.

Collection acquisition boards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`Labjack UE9 <crappy_docs/inouts:labjack ue9>`

  Controls Labjack's `UE9 <https://labjack.com/products/
  calibration-service-with-cert>`_ acquisition board. It can only read the
  input analog channels of the board.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Waveshare AD/DA <crappy_docs/inouts:waveshare ad/da>`

  Controls Waveshare's `AD/DA <https://www.waveshare.com/product/raspberry-pi/
  hats/ad-da-audio-sensors/high-precision-ad-da-board.htm>`_ Raspberry Pi
  acquisition hat. May be used from any device with a proper wiring, but more
  convenient to use from a Raspberry Pi. Communicates over SPI.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Waveshare High Precision <crappy_docs/inouts:waveshare high precision>`

  Controls Waveshare's `High Precision HAT
  <https://www.waveshare.com/18983.htm>`_ Raspberry Pi acquisition hat. It
  features a 10-channels 32 bits ADC. It may be used from any device able to
  communicate over SPI, but is originally meant for interfacing with a
  Raspberry Pi.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

Collection sensors
^^^^^^^^^^^^^^^^^^

- :ref:`Agilent 34420A <crappy_docs/inouts:agilent 34420a>`

  Reads voltages or resistances from Agilent's `34420A <https://www.keysight.
  com/us/en/product/34420A/micro-ohm-meter.html?&cc=FR&lc=fre>`_ precision
  multimeter. Communicates over serial.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`Eurotherm EPC3008 <crappy_docs/inouts:eurotherm epc3008>`

  Controls an `Eurotherm EPC3008 <https://www.eurotherm.com/us/products/
  temperature-controllers-us/single-loop-temperature-controllers-us/
  epc3000-programmable-controllers/>`_ temperature controller. Allows setting
  the temperature setpoint and reading the current process value. Typically
  used for managing the temperature of a furnace or industrial process over a
  serial Modbus RTU connection.

  .. Note::
     This object was developed for furnace control but could be adapted to
     similar Eurotherm models supporting Modbus RTU.

- :ref:`Flow Controller Alicat <crappy_docs/inouts:flow controller alicat>`

  Reads and controls an `Alicat <https://www.alicat.com/products/
  mass-flow-meters-and-controllers/mass-flow-controllers/>`_ mass flow
  controller over Modbus RTU. Acquires several process variables such as
  pressure, temperature, mass flow and volumetric flow, and also sets the mass
  flow setpoint. Communicates over a serial RS485 connection.

  .. Note::
     This object was developed for Alicat MFCs supporting the Modbus RTU
     protocol. Other communication protocols (e.g. ASCII) are not supported in
     this implementation.

- :ref:`MCP9600 <crappy_docs/inouts:mcp9600>`

  Reads temperatures from Adafruit's `MCP9600 <https://www.adafruit.com/product
  /4101>`_ thermocouple amplifier. Communicates over I2C.

- :ref:`OpSens HandySens <crappy_docs/inouts:opsens handysens>`

  Reads data from OpSens' `single channel signal conditioner <https://opsens-
  solutions.com/products/signal-conditioners-oem-boards/handysens-w/>`_ for
  fiber-optics temperature, strain, pressure or position measurement.
  Communicates over serial.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

- :ref:`PiJuice <crappy_docs/inouts:pijuice>`

  Reads the charging status and battery level of Kubii's `PiJuice <https://
  www.kubii.com/fr/14-chargeurs-alimentations-raspberry/2019-pijuice-hat-kubii
  -3272496008793.html>`_ Raspberry Pi power supply.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Sager SG-GS1700 <crappy_docs/inouts:sager sg-gs1700>`

  Controls a `Sager SG-GS1700 <https://sagerindustrial.en.alibaba.com/
  productgrouplist-805331243/
  Four_tubulaire.html?spm=a2700.shop_index.88.23.432c2d34arZxF6/>`_ furnace
  controller over a serial link using the AIBUS protocol. Allows reading the
  process temperature (PV) and the current setpoint (SV), and writing a new
  temperature setpoint.

  .. Note::
     This object was developed for a specific furnace controller. The AIBUS
     frame format and checksum (ECC) follow the implementation provided with
     the device.

- :ref:`Spectrum M2I 4711 <crappy_docs/inouts:spectrum m2i 4711>`

  Reads voltages from Spectrum's `M2i 4711 EXP <https://spectrum-
  instrumentation.com/products/details/M2i4711.php>`_ high-speed ADC
  communicating over PCIexpress.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

Collection multi-device drivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`Comedi <crappy_docs/inouts:comedi>`

  Reads voltages from an `USB-DUX Sigma <https://github.com/glasgowneuro/usbdux/
  tree/main/usbdux-sigma>`_ ADC (not manufactured anymore) using the `Comedi
  <https://www.comedi.org/>`_ driver. The code was written to work as-is on
  other acquisition boards supporting the Comedi driver, but this hasn't been
  tested. Communicates over serial.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

Collection outputs
^^^^^^^^^^^^^^^^^^

- :ref:`Sim868 <crappy_docs/inouts:sim868>`

  Uses Waveshare's `GSM/GPRS/GNSS/Bluetooth hat <https://www.waveshare.com/
  gsm-gprs-gnss-hat.htm>`_ for sending SMS. The other functionalities are not
  implemented. Usable from any device with a proper wiring, but more convenient
  to use with a Raspberry Pi. Communicates over serial.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

Collection enhanced actuators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`Kollmorgen AKD PDMM <crappy_docs/inouts:kollmorgen akd pdmm>`

  Drives Kollmorgen's `AKD PDMM <https://www.kollmorgen.com/en-us/products/
  drives/servo/akd-pdmm/akd-pdmm-programmable-drive-multi-axis-master/>`_
  servomotor controller. As this device supports many settings, it was decided
  to consider it as an InOut to fully take advantage of its versatility.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object has not been maintained or tested recently. Its current
     behavior is not verified.

LaMcube-specific hardware
+++++++++++++++++++++++++

- :ref:`Bi Spectral <crappy_docs/lamcube:bi spectral>`

  An infrared camera acquiring on two wavelengths at the same time.

  .. Important::
     Only intended for an internal use in our laboratory as it is not
     commercially available.

- :ref:`Biaxe <crappy_docs/lamcube:biaxe>`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in speed. May as well work on
  other conditioners from the same brand, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Biotens <crappy_docs/lamcube:biotens>`

  A simple wrapper around the :ref:`JVL Mac140 <crappy_docs/actuators:jvl mac140>` Actuator, to keep the legacy
  name of this object.

On-the-fly data modification (Modifiers)
----------------------------------------

- :ref:`Demux <crappy_docs/modifiers:demux>`

  Takes the signal returned by a streaming :ref:`IOBlock <crappy_docs/blocks:ioblock>` and transforms it
  into a regular signal usable by most Blocks. This Modifier is mandatory for
  plotting data from a streaming device.

- :ref:`Differentiate <crappy_docs/modifiers:differentiate>`

  Calculates the time derivative of a given label.

- :ref:`DownSampler <crappy_docs/modifiers:downsampler>`

  Transmits the values to downstream Blocks only once every given number of 
  points. The values that are not sent are discarded. The values are directly 
  sent without being altered.

- :ref:`Integrate <crappy_docs/modifiers:integrate>`

  Integrates a given label over time.

- :ref:`Mean <crappy_docs/modifiers:mean>`

  Returns the mean value of a label over a given number of points. Only returns
  a value once every number of points.

- :ref:`Median <crappy_docs/modifiers:median>`

  Returns the median value of a label over a given number of points. Only
  returns a value once every number of points.

- :ref:`Offset <crappy_docs/modifiers:offset>`

  Offsets the given labels by a constant value calculated so that the first
  received value is offset to a given target.

- :ref:`Moving Average <crappy_docs/modifiers:moving average>`

  Returns the moving average of a label over a given number of points. Returns
  a value at the same frequency as the label.

- :ref:`Moving Median <crappy_docs/modifiers:moving median>`

  Returns the moving median of a label over a given number of points. Returns
  a value at the same frequency as the label.

- :ref:`Trig on change <crappy_docs/modifiers:trig on change>`

  Returns the received label only if the new value differs from the previous
  one.

- :ref:`Trig on value <crappy_docs/modifiers:trig on value>`

  Returns the received label only if the value is in a predefined list of
  accepted values.
