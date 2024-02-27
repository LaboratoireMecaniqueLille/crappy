=======================
Current functionalities
=======================

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

On this page are listed all the objects currently distributed with Crappy and
exposed to the users. Information on how to use them can be found in the
:ref:`Tutorials`, as well as guidelines for creating your own objects. For most
Blocks, one or several directly runnable example scripts are available
in the `examples folder <https://github.com/LaboratoireMecaniqueLille/crappy/
tree/master/examples>`_ of the GitHub repository. For each object listed on
this page, you can click on its name to open the complete documentation given
in the :ref:`API`.

Functionalities (Blocks)
------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

The Blocks are the base bricks of Crappy, that fulfill various functions. In
the tutorials, you can learn more about :ref:`how to use Blocks
<1. Understanding Crappy's syntax>` and :ref:`how to create new Blocks
<5. Custom Blocks>`.

Data display
++++++++++++

- :ref:`Canvas`

  Displays the data it receives on top of a static image, e.g. for having a
  real-time temperature map.

  The examples folder on GitHub contains `one example of the Canvas Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/canvas.py>`_.

- :ref:`Dashboard`

  Prints the values it receives in a popup window with a nicer formatting than
  :ref:`Link Reader`. Unlike the :ref:`Grapher` Block, only the latest value is
  displayed for each label.

  The examples folder on GitHub contains `one example of the Dashboard Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/dashboard.py>`_.

- :ref:`Grapher`

  Plots real-time 2D graphs. It is possible to plot several datasets on a same
  graph. The *x* axis can be the time information, or any other label that the
  Grapher receives. Unlike the :ref:`Dashboard` Block, the displayed data is
  persistent and allows to visualize the history of a label.

  The examples folder on GitHub contains `one example of the Grapher Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/dashboard.py>`_ specifically, but it is also used in most of the other
  examples.

  :ref:`A tutorial section <2.c. The Grapher Block>` is also dedicated to the
  Grapher Block.

- :ref:`Link Reader`

  Prints the values it receives in the terminal. Mostly useful for debugging.

  The examples folder on GitHub contains `one example of the Link Reader Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/link_reader.py>`_.

Data recording
++++++++++++++

- :ref:`HDF Recorder`

  Writes the data it receives to a *.hdf5* file. Only compatible with the
  :ref:`IOBlock` in *streamer* mode. The :ref:`Recorder` should be used for
  recording any other type of data.

  The examples folder on GitHub contains `one example of the HDF Recorder Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/hdf5_recorder.py>`_.

- :ref:`Recorder`

  Writes the data it receives to a *.csv* file, for recording it. It is
  compatible with data from any Block, except data coming from an
  :ref:`IOBlock` in *streamer* mode. The :ref:`HDF Recorder` should be used
  instead in this situation.

  The examples folder on GitHub contains `one example of the Recorder Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/recorder.py>`_.

  :ref:`A tutorial section <2.d. The Recorder Block>` is also dedicated to the
  Recorder Block.

Data processing
+++++++++++++++

- :ref:`Mean <Mean Block>`

  Calculates the average of the received labels over a given period, and sends
  it to downstream Blocks. One average value is given for each label, it is not
  meant to average several labels together. Can be used as a less
  computationally-intensive :ref:`Multiplexer`.

  The examples folder on GitHub contains `one example of the Mean Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/mean.py>`_.

- :ref:`Multiplexer`

  Allows putting labels emitted at different frequencies on a same time basis.
  Useful for plotting curves out of two labels from different Blocks with a
  :ref:`Grapher`, as the timestamps of the data points would otherwise never
  match. Also used before saving data with a :ref:`Recorder` to simplify the
  post-processing.

  The examples folder on GitHub contains `one example of the Multiplexer Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/multiplexer.py>`_.

Real-time image correlation
+++++++++++++++++++++++++++

- :ref:`DIS Correl`

  Child of the :ref:`Camera` Block that can acquire, record and display images.
  In addition, it performs real-time Dense Inverse Search (DIS) image
  correlation on the acquired images using :mod:`cv2`'s `DISOpticalFlow`, and
  projects the displacement field on a predefined basis. The result is then
  sent to downstream Blocks.

  The examples folder on GitHub contains `several examples of the DIS Correl
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
  examples/blocks/dis_correl>`_.

- :ref:`GPU Correl`

  Same as :ref:`DIS Correl`, except the computation is performed on a
  Cuda-compatible GPU.

  There is currently no example featuring this Block distributed in the
  examples folder on GitHub.

  .. Important::
     This Block hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected ! On the long-term, it should be replaced
     by another Block.

Video-extensometry
++++++++++++++++++

- :ref:`Auto Drive`

  This Block drives an :ref:`Actuator`, just like the :ref:`Machine` Block.
  However, it does it in a very specific context. It allows moving a
  :ref:`Camera` performing video-extensometry and mounted on an
  :ref:`Actuator`, so that the barycenter of the tracked dots remains in the
  center of the image. To do so, it takes the output of a :ref:`Video Extenso`
  Block as its input.

  The examples folder on GitHub contains `one example of the Auto Drive Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/auto_drive_video_extenso.py>`_.

- :ref:`DIC VE`

  Child of the :ref:`Camera` Block that can acquire, record and display images.
  In addition, it performs image correlation on four patches on the acquired
  images. From the correlation, it deduces the *x* and *y* displacement of each
  patch, and can then calculate the global strain on the filmed sample. The
  displacements and the strain values are sent to downstream Block. Can be used
  to replace :ref:`Video Extenso` on samples with a speckle drawn on them, each
  patch playing the same role as a dot.

  The examples folder on GitHub contains `several examples of the DIC VE Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/dic_ve>`_.

- :ref:`GPU VE`

  Same as :ref:`DIC VE`, except the computation is done on a Cuda-compatible
  GPU.

  There is currently no example featuring this Block distributed in the
  examples folder on GitHub.

  .. Important::
     This Block hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected ! On the long-term, it should be replaced
     by another Block.

- :ref:`Video Extenso`

  Child of the :ref:`Camera` Block that can acquire, record and display images.
  In addition, it performs real-time video-extensometry on the acquired images.
  It can track from two to four spots drawn on the filmed sample, and tracks
  the position of each spot to get their displacement. Based on the
  displacements, a global *x* and *y* strain values are computed. The strain
  and the displacement values are sent to downstream Blocks. The :ref:`DIC VE`
  Block performs a similar task but uses image correlation for tracking the
  areas.

  The examples folder on GitHub contains `one example of the Video Extenso
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/
  examples/blocks/video_extenso.py>`_.

Signal generation
+++++++++++++++++

- :ref:`Button`

  Creates a small graphical window with a button in it, and generates a signal
  when the user clicks on the button. This signal is sent to downstream Blocks.
  Useful for triggering a behavior at a user-chosen moment during a test.

  The examples folder on GitHub contains `one example of the Button Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/button.py>`_.

- :ref:`Generator`

  Generates a signal following a pattern given by the user (like sine waves,
  triangles, squares, etc.), and sends this signal to downstream Blocks. It
  can only output a combination of :ref:`Generator Paths`.

  The examples folder on GitHub contains `several examples of the Generator
  Block <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/
  examples/blocks/generator>`_ specifically, but it is also used in many of the
  other examples.

  :ref:`A tutorial section <2.a. The Generator Block and its Paths>` is also
  dedicated to the Generator Block, and :ref:`another one
  <1. Custom Generator Paths>` is dedicated to the creation of custom Generator
  Paths.

- :ref:`PID`

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

- :ref:`Camera <Camera Block>`

  Acquires images from a :ref:`Camera` object, and then displays and/or records
  the acquired images. It is the base class for other Blocks that can also
  perform image processing, in addition to the recording and display. This
  Block usually doesn't have input nor output Links, but can in some specific
  situations.

  The examples folder on GitHub contains `several examples of the Camera Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/camera>`_.

  :ref:`A tutorial section <2.b. The Camera Block>` is also dedicated to the
  Camera Block, and :ref:`another one <4. Custom Cameras>` is dedicated to the
  creation of custom Camera objects.

- :ref:`IOBlock`

  Controls one :ref:`InOut <In / Out>` object, allowing to read data from
  sensors and/or to give it commands to set on hardware. It is originally
  intended for interfacing with DAQ boards, but can also be used to drive a
  variety of other devices. It has output Links when acquiring data, and input
  Links when setting commands.

  The examples folder on GitHub contains `several examples of the IOBlock Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/ioblock>`_.

  :ref:`A tutorial section <2.e. The IOBlock Block>` is also dedicated to the
  IOBlock Block, and :ref:`another one <3. Custom InOuts>` is dedicated to the
  creation of custom InOut objects.

- :ref:`Machine`

  Drives one or several :ref:`Actuator` in speed or in position, based on the
  received command labels. Can also acquire the current speed and/or position
  from the driven Actuators, and return it to the downstream Blocks. This Block
  is intended for driving motors and similar devices.

  The examples folder on GitHub contains `several examples of the Machine Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/machine>`_.

  :ref:`A tutorial section <2.f. The Machine Block>` is also dedicated to the
  Machine Block, and :ref:`another one <2. Custom Actuators>` is dedicated to
  the creation of custom Actuator objects.

- :ref:`UController`

  Controls a microcontroller over serial. :ref:`A MicroPython and an Arduino
  template <Microcontroller templates>` to use along with this Block are
  provided with Crappy. This Block can start or stop the script on the
  microcontroller, send commands, and receive data.

  The examples folder on GitHub contains `on example of the UController Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/ucontroller>`_.

Others
++++++

- :ref:`Client Server`

  Sends and/or receives data over a local network via an MQTT server. Can also
  start a `Mosquitto <https://mosquitto.org/>`_ MQTT broker. Used for
  communicating with distant devices over a network, e.g. for remotely
  controlling a test.

  The examples folder on GitHub contains `on example of the Client Server Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples/
  blocks/client_server>`_.

- :ref:`Fake Machine`

  Emulates the behavior of a tensile test machine, taking a position command as
  input and outputting the force and the displacement. Mainly used in the
  examples because it doesn't require any hardware, but may as well be used for
  debugging or prototyping.

  The examples folder on GitHub contains `on example of the Fake Machine Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/fake_machine.py>`_.

- :ref:`Sink`

  Discards any received data. Used for prototyping and debugging only.

  The examples folder on GitHub contains `on example of the Sink Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/sink.py>`_.

- :ref:`Stop Block`

  Stops the current Crappy script if the received data meets one of the given
  criteria. One of the clean ways to stop a script in Crappy.

  The examples folder on GitHub contains `on example of the Stop Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/stop_block.py>`_.

  Refer to the :ref:`dedicated tutorial section
  <3. Properly stopping a script>` to learn more about how to properly stop a
  script in Crappy.

- :ref:`Stop Button`

  Stops the current Crappy script when the user clicks on a button in a GUI.
  One of the clean ways to stop a script in Crappy.

  The examples folder on GitHub contains `on example of the Stop Button Block
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/examples/
  blocks/stop_button.py>`_.

  Refer to the :ref:`dedicated tutorial section
  <3. Properly stopping a script>` to learn more about how to properly stop a
  script in Crappy.

Supported hardware (Cameras, InOuts, Actuators)
-----------------------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>

Supported Cameras
+++++++++++++++++

- :ref:`Basler Ironman Camera Link`

  Allows reading images from a camera communicating over Camera Link plugged to
  a `microEnable 5 ironman AD8-PoCL <https://www.baslerweb.com/en/
  acquisition-cards/frame-grabbers/>`_ PCIexpress board. May as well work with
  similar boards.

  .. Important::
     This Camera object relies on C++ libraries, which are not distributed with
     :mod:`pip`. They can only be used after a local install, after cloning the
     repo.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected ! On the long-term, it should be totally
     removed.

- :ref:`Camera GStreamer`

  This Camera object opens video streams using the Python binding of `Gstreamer
  <https://gstreamer.freedesktop.org/>`_. It can open a camera by path (in
  Linux) or number (in Windows and Mac), in which case the GStreamer pipeline
  is generated automatically. Alternatively, it can also open a stream
  following a custom pipeline given by the user.

  Compared to the :ref:`Camera OpenCV` camera, the GStreamer one is less
  CPU-intensive and is compatible with more devices. Its dependencies are
  however harder to install (especially on Windows) and it is harder to make it
  work properly.

  .. Important::
     This Camera object can only be used at its fullest on Linux, and only if
     the *v4l-utils* package is installed on the system !

- :ref:`Camera OpenCV`

  This Camera object opens video streams using OpenCV. It allows tuning the
  device number, as well as the image format and the number of channels. It is
  mostly compatible with USB cameras, and its dependencies are straightforward
  to install.

  .. Important::
     This Camera object can only be used at its fullest on Linux, and only if
     the *v4l-utils* package is installed on the system !

- :ref:`Fake Camera`

  Simply displays an animated image of a chosen size and at a given frequency.
  Doesn't require any hardware, used mainly for debugging and prototyping.

- :ref:`File Reader`

  Successively reads images already saved in a folder, and returns them as if
  they just had been acquired by a real camera. No real image acquisition is
  performed though, and no hardware is required.

- :ref:`JAI GO-5000C-PMCL`

  Allows reading images from a `Jai GO-5000M-PMCL <https://www.jai.com/
  products/go-5000c-pmcl>`_ camera. It relies on the :ref:`Basler Ironman
  Camera Link` object.

  .. Important::
     This Camera object relies on C++ libraries, which are not distributed with
     :mod:`pip`. They can only be used after a local install, after cloning the
     repo.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected ! On the long-term, it should be totally
     removed.

- :ref:`Raspberry Pi Camera`

  Allows reading images from a Raspberry Pi Camera, with Crappy running on a
  Raspberry Pi. It has been tested on Raspberry Pi 3 and 4, with a variety of
  official Raspberry Pi cameras.

  .. Important::
     Can only be run on a Raspberry Pi ! Also, it is for now only compatible
     with the *Buster* version of Raspberry Pi OS, or with *Bullseye* in legacy
     camera mode.

- :ref:`Seek Thermal Pro`

  Allows reading images from a Seek Thermal `Compact Pro <https://www.thermal.
  com/compact-series.html>`_ infrared camera.

- :ref:`Webcam`

  Reads images from a video device recognized by OpenCV. Usually webcams fall
  into this category, but some other cameras as well. This class is really
  basic and is intended for demonstration, see :ref:`Camera OpenCV` and
  :ref:`Camera GStreamer` for classes providing a finer controls over the
  devices.

- :ref:`Xi API`

  Allows reading images from any `Ximea <https://www.ximea.com/>`_ camera. The
  backend is the official Ximea API.

Supported Actuators
+++++++++++++++++++

- :ref:`Adafruit DC Motor Hat`

  Drives up to 4 DC motors using Adafruit's `DC & Stepper Motor HAT for
  Raspberry Pi <https://www.adafruit.com/product/2348>`_, using either
  Adafruit's Blinka library or :mod:`smbus2` if driven from a Raspberry Pi.
  Although this component can also drive stepper motors, this feature was not
  implemented.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Fake DC Motor`

  Emulates the dynamic behavior of a DC motor, but doesn't drive any hardware.
  Used in the examples, may also be used for prototyping or debugging.

- :ref:`Fake Stepper Motor`

  Emulates the dynamic behavior of a stepper motor used as a linear actuator,
  but does not drive any actual hardware. It is used in examples, and can also
  be used for debugging. Unlike the :ref:`Fake DC Motor`, it can drive the
  motor in position.

- :ref:`JVL Mac140`

  Drives JVL's `MAC140 <https://www.jvl.dk/276/integrated-servo-motors-mac050
  -141>`_ integrated servomotor in speed or in position. Probably works with
  other integrated servomotors from JVL, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Kollmorgen ServoStar 300`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in position or sets it to the
  analog driving mode. This is the same conditioner as for the :ref:`Biaxe`
  Actuator, but this object was designed for an other application.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Newport TRA6PPD`

  Drives Newport's `TRA6PPD <https://www.newport.com/p/TRA6PPD>`_ miniature
  linear stepper motor actuator, in speed or in position.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Oriental ARD-K`

  Drives Oriental Motor's `ARD-K <https://catalog.orientalmotor.com/item/s-
  closed-loop-stepper-motor-drivers-dc-input/ard-closed-loop-stepper-driver-
  pulse-input-dc/ard-k>`_ stepper motor driver in speed or in position.
  Probably works with other stepper motor drivers in the same range of
  products, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Phidget Stepper4A`

  Drives 4A bipolar stepper motors using Phidget's `Stepper4A <https://
  www.phidgets.com/?prodid=1278>`_ in speed or in position, by using several
  Phidget libraries.

  .. Important::
     This Actuator must be connected to Phidget's VINT Hub to work. See the
     following link `<https://www.phidgets.com/?prodid=1278#Tab_User_Guide>`_
     to connect properly to the Hub.

- :ref:`Pololu Tic`

  Drives Pololu's `Tic <https://www.pololu.com/category/212/tic-stepper-motor-
  controllers>`_ stepper motor drivers in speed or in position. Designed for
  driving all the Tic drivers, but tested only on the 36v4 model.

- :ref:`Schneider MDrive 23`

  Drives Schneider Electric's `MDrive 23 <https://www.novantaims.com/downloads
  /quickreference/mdi23plus_qr.pdf>`_ stepper motor in speed or in position.
  Probably works with other stepper motors in the same range of products,
  although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

Supported Sensors and outputs
+++++++++++++++++++++++++++++

Acquisition boards
""""""""""""""""""

- :ref:`Labjack T7`

  Controls Labjack's `T7 <https://labjack.com/products/labjack-t7>`_
  acquisition board. It can acquire data from its ADCs, set the output of DACs,
  read and set the GPIOs, and also supports more advanced functions like
  reading thermocouples.

- :ref:`Labjack T7 Streamer`

  Controls Labjack's `T7 <https://labjack.com/products/labjack-t7>`_
  acquisition board in streaming mode. In this mode, it can only acquire data
  from the ADCs and does not support any other function.

- :ref:`Labjack UE9`

  Controls Labjack's `UE9 <https://labjack.com/products/calibration-service-
  with-cert-u6-ue9-t7>`_ acquisition board. It can only read the input analog
  channels of the board.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Waveshare AD/DA`

  Controls Waveshare's `AD/DA <https://www.waveshare.com/product/raspberry-pi/
  hats/ad-da-audio-sensors/high-precision-ad-da-board.htm>`_ Raspberry Pi
  acquisition hat. May be used from any device with a proper wiring, but more
  convenient to use from a Raspberry Pi. Communicates over SPI.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Waveshare High Precision`

  Controls Waveshare's `High Precision HAT
  <https://www.waveshare.com/18983.htm>`_ Raspberry Pi acquisition hat. It
  features a 10-channels 32 bits ADC. It may be used from any device able to
  communicate over SPI, but is originally meant for interfacing with a
  Raspberry Pi.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

Sensors
"""""""

- :ref:`ADS1115`

  Reads voltages from Adafruit's `ADS 1115 <https://www.adafruit.com/product/
  1085>`_ ADC. Communicates over I2C.

- :ref:`Agilent 34420A`

  Reads voltages or resistances from Agilent's `34420A <https://www.keysight.
  com/us/en/product/34420A/micro-ohm-meter.html?&cc=FR&lc=fre>`_ precision
  multimeter. Communicates over serial.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Fake Inout`

  Can acquire the current RAM usage of the computer using the :mod:`psutil`
  module, and also instantiate useless objects to reach a target memory usage
  (if superior to the base memory usage). It supports the streamer mode for the
  data acquisition. Mainly intended for demonstration, and used in the
  distributed examples.

- :ref:`MCP9600`

  Reads temperatures from Adafruit's `MCP9600 <https://www.adafruit.com/product
  /4101>`_ thermocouple amplifier. Communicates over I2C.

- :ref:`MPRLS`

  Reads pressures from Adafruit's `MPRLS <https://www.adafruit.com/product/
  3965>`_ pressure sensor. Communicates over I2C.

- :ref:`NAU7802`

  Reads voltages from Sparfun's `'Qwiic Scale' NAU7802 <https://www.sparkfun.
  com/products/15242>`_ load cell conditioner. Communicates over I2C.

- :ref:`OpSens HandySens`

  Reads data from OpSens' `single channel signal conditioner <https://opsens-
  solutions.com/products/signal-conditioners-oem-boards/handysens-w/>`_ for
  fiber-optics temperature, strain, pressure or position measurement.
  Communicates over serial.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`Phidget Wheatstone Bridge`

  Reads volatges from Phidget's `Wheatstone Bridge <https://www.phidgets.com/
  ?prodid=957>`_ load cell conditioner, by using several Phidget libraries.

  .. Important::
     This InOut must be connected to Phidget's VINT Hub to work. See the
     following link `<https://www.phidgets.com/?prodid=957#Tab_User_Guide>`_ to
     connect properly to the Hub.

- :ref:`PiJuice`

  Reads the charging status and battery level of Kubii's `PiJuice <https://
  www.kubii.com/fr/14-chargeurs-alimentations-raspberry/2019-pijuice-hat-kubii
  -3272496008793.html>`_ Raspberry Pi power supply.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Spectrum M2I 4711`

  Reads voltages from Spectrum's `M2i 4711 EXP <https://spectrum-
  instrumentation.com/products/details/M2i4711.php>`_ high-speed ADC
  communicating over PCIexpress.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

Multi-device drivers
""""""""""""""""""""

- :ref:`Comedi`

  Reads voltages from an `USB-DUX Sigma <https://github.com/glasgowneuro/usbdux/
  tree/main/usbdux-sigma>`_ ADC (not manufactured anymore) using the `Comedi
  <https://www.comedi.org/>`_ driver. The code was written to work as-is on
  other acquisition boards supporting the Comedi driver, but this hasn't been
  tested. Communicates over serial.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`DAQmx`

  Same as :ref:`NI DAQmx`, except it relies on the :mod:`PyDAQmx` module. The
  differences between the two modules weren't further investigated.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

- :ref:`NI DAQmx`

  Controls National Instrument's `USB 6008 <https://www.ni.com/en-us/support/
  model.usb-6008.html>`_ DAQ module using the :mod:`nidaqmx` module. The code
  was written to work as-is on other National Instruments acquisition modules,
  but this hasn't been tested. Communicates over USB.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

Outputs
"""""""

- :ref:`GPIO PWM`

  Controls a PWM output on a single GPIO of a Raspberry Pi.

  .. Important:: Only works on a Raspberry Pi ! Tested on Raspberry Pi 3 and 4,
     with the *Buster* and *Bullseye* Raspberry Pi Os for the latter.

- :ref:`GPIO Switch`

  Drives a single GPIO on a Raspberry Pi, or any other board supporting Blinka.

  .. Important:: Only works on a Raspberry Pi ! Tested on Raspberry Pi 3 and 4,
     with the *Buster* and *Bullseye* Raspberry Pi Os for the latter.

- :ref:`Sim868`

  Uses Waveshare's `GSM/GPRS/GNSS/Bluetooth hat <https://www.waveshare.com/
  gsm-gprs-gnss-hat.htm>`_ for sending SMS. The other functionalities are not
  implemented. Usable from any device with a proper wiring, but more convenient
  to use with a Raspberry Pi. Communicates over serial.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

Enhanced Actuators
""""""""""""""""""

- :ref:`Kollmorgen AKD PDMM`

  Drives Kollmorgen's `AKD PDMM <https://www.kollmorgen.com/en-us/products/
  drives/servo/akd-pdmm/akd-pdmm-programmable-drive-multi-axis-master/>`_
  servomotor controller. As this device supports many settings, it was decided
  to consider it as an InOut to fully take advantage of its versatility.

  .. Important::
     This InOut was written for a specific application, so it may not be
     usable as-is in the general case.

  .. Important::
     This object hasn't been maintained nor tested for a while, it is not sure
     that it still works as expected !

LaMcube-specific hardware
+++++++++++++++++++++++++

- :ref:`Bi Spectral`

  An infrared camera acquiring on two wavelengths at the same time.

  .. Important::
     Only intended for an internal use in our laboratory as it is not
     commercially available.

- :ref:`Biaxe`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in speed. May as well work on
  other conditioners from the same brand, although it hasn't been tested.

  .. Important::
     This Actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Biotens`

  A simple wrapper around the :ref:`JVL Mac140` Actuator, to keep the legacy
  name of this object.

On-the-fly data modification (Modifiers)
----------------------------------------

.. sectionauthor:: Antoine Weisrock <antoine.weisrock@gmail.com>
.. sectionauthor:: Pierre Margotin <pierremargotin@gmail.com>

- :ref:`Demux`

  Takes the signal returned by a streaming :ref:`IOBlock` and transforms it
  into a regular signal usable by most Blocks. This Modifier is mandatory for
  plotting data from a streaming device.

- :ref:`Differentiate`

  Calculates the time derivative of a given label.

- :ref:`DownSampler`

  Transmits the values to downstream Blocks only once every given number of 
  points. The values that are not sent are discarded. The values are directly 
  sent without being altered.

- :ref:`Integrate`

  Integrates a given label over time.

- :ref:`Mean`

  Returns the mean value of a label over a given number of points. Only returns
  a value once every number of points.

- :ref:`Median`

  Returns the median value of a label over a given number of points. Only
  returns a value once every number of points.

- :ref:`Offset`

  Offsets the given labels by a constant value calculated so that the first
  received value is offset to a given target.

- :ref:`Moving Average`

  Returns the moving average of a label over a given number of points. Returns
  a value at the same frequency as the label.

- :ref:`Moving Median`

  Returns the moving median of a label over a given number of points. Returns
  a value at the same frequency as the label.

- :ref:`Trig on change`

  Returns the received label only if the new value differs from the previous
  one.

- :ref:`Trig on value`

  Returns the received label only if the value is in a predefined list of
  accepted values.
