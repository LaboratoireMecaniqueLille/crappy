=======================
Current functionalities
=======================

Here are listed all the blocks, hardware and other objects currently distributed
with Crappy. Information on how to use them can be found in :ref:`Tutorials`, as
well as guidelines for adding your own functionalities.

Functionalities (Blocks)
------------------------

Data plotting
+++++++++++++

- :ref:`Grapher`

  Plots real-time 2D graphs. Possible to plot several data sets on a same graph.

  Most of the `examples <https://github.com/LaboratoireMecaniqueLille/crappy/
  tree/master/Examples>`_ include a Grapher block, refer to them for a use case.

- :ref:`Canvas`

  Displays data on top of a static image, e.g. for having a real-time
  temperature map.

  Refer to the `drawing.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/drawing.py>`_ example for a use case.

- :ref:`Link Reader`

  Prints the received values in the terminal. Mostly useful for debugging.

  No example featuring a Link Reader block is currently distributed.

- :ref:`Dashboard`

  Prints received values in a popup window and with a nicer formatting than
  :ref:`Link Reader`.

  No example featuring a Dashboard block is currently distributed.

Data recording
++++++++++++++

- :ref:`Recorder`

  Writes received data to a file, usually in a `.csv`.

  Refer to the `tensile_1.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/tensile_1.py>`_ example (among others) for a use
  case.

- :ref:`HDF Recorder`

  Writes received data to an hdf5 file.

  Refer to the `spectrum.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/spectrum.py>`_ example for a use case.

Data processing
+++++++++++++++

- :ref:`Mean <Mean Block>`

  Calculates the average of the received labels over a given period. One
  average is given for every label, it is not meant to average several labels
  together. Can be used as a less computationally-intensive :ref:`Multiplexer`.

  Refer to the `mean.py <https://github.com/LaboratoireMecaniqueLille/crappy/
  blob/master/Examples/mean.py>`_ example for a use case.

- :ref:`Multiplexer`

  Allows putting labels emitted at different frequencies on a same time basis.
  Useful for plotting curves made of two labels from different blocks, as the
  timestamps of the data points would otherwise never match. Also used before
  saving data to ease post-processing.

  Refer to the `multiplexer.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/multiplexer.py>`_ example for a use case.

Real-time correlation
+++++++++++++++++++++

- :ref:`DIS Correl`

  Performs real-time Dense Inverse Search (DIS) image correlation using
  :mod:`cv2`'s `DISOpticalFlow`, and projects the displacement field on a
  predefined basis.

  Refer to the `discorrel_basic.py <https://github.com/LaboratoireMecanique
  Lille/crappy/blob/master/Examples/discorrel_basic.py>`_ example for a use
  case.

- :ref:`GPU Correl`

  Same as :ref:`DIS Correl` except the computation is done on a Cuda-compatible
  GPU.

  Refer to the `gpucorrel_fake_test.py <https://github.com/LaboratoireMecanique
  Lille/crappy/blob/master/Examples/gpucorrel_fake_test.py>`_ example for a use
  case.

Video-extensometry
++++++++++++++++++

- :ref:`Video Extenso`

  Performs real-time video-extensometry on two to four dots and returns the `x`
  and `y` strains.

  Refer to the `ve_fake_test.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/ve_fake_test.py>`_ example for a use case.

- :ref:`DIS VE`

  Performs DIS correlation just like :ref:`DIS Correl` but only on the areas
  selected by the user, and returns the `x` and `y` displacement for each area.
  Can be used to replace :ref:`Video Extenso` on speckled samples, each area
  playing the same role as a dot.

  No example featuring a Disve block is currently distributed.

- :ref:`GPU VE`

  Same as :ref:`DIS VE` except the computation is done on a Cuda-compatible GPU.

  No example featuring a GPUve block is currently distributed.

- :ref:`Auto Drive`

  Allows moving a camera performing video-extensometry and mounted on an
  actuator, so that the barycenter of the dots remains in the center of the
  image.

  No example featuring an Autodrive block is currently distributed.

Signal generation
+++++++++++++++++

- :ref:`Generator`

  Generates a signal following a predefined pattern. See :ref:`the tutorials
  <2. Adding signal generators>` for information on how to use it.

  Refer to the `generator.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/generator.py>`_ example for a use case.

- :ref:`Button`

  Generates a signal when the user clicks on a button in a GUI. Useful for
  triggering a behavior during an assay.

  No example featuring a Button block is currently distributed.

- :ref:`PID`

  Generates a signal based on the target and measured inputs according to a PID
  controller logic.

  Refer to the `pid.py <https://github.com/LaboratoireMecaniqueLille/crappy/
  blob/master/Examples/pid.py>`_ example for a use case.

Hardware control
++++++++++++++++

- :ref:`Machine`

  Controls one or several :ref:`actuators <Actuators>` according to the
  received command signal.

  Refer to the `custom_actuator.py <https://github.com/LaboratoireMecanique
  Lille/crappy/blob/master/Examples/custom_actuator.py>`_ example for a use
  case.

- :ref:`IOBlock`

  Controls one :ref:`inout <In / Out>` object, allowing to read data from it
  and/or to give it inputs.

  Refer to the `custom_in.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/custom_in.py>`_ and `custom_out.py <https://
  github.com/LaboratoireMecaniqueLille/crappy/blob/master/Examples/
  custom_in.py>`_ examples for use cases.

- :ref:`UController`

  Controls a microcontroller over serial. A template of a MicroPython script
  to run on the microcontroller is provided in Crappy. This block can start or
  stop the script on the microcontroller, send commands, and receive data.

  Refer to the `microcontroller.py <https://github.com/LaboratoireMecanique
  Lille/crappy/blob/master/Examples/microcontroller_example.py>`_ example for a
  use case.

- :ref:`Camera`

  Controls one :ref:`camera <Cameras>` and reads images from it.

  Refer to the `custom_camera.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/custom_camera.py>`_ example for a use case.

Others
++++++

- :ref:`Client Server`

  Sends and/or receives data from an MQTT server. Can also start a `Mosquitto
  <https://mosquitto.org/>`_ MQTT broker. Used for communicating with distant
  devices over a network, e.g. for remotely controlling an assay.

  No example featuring a Client Server block is currently distributed.

- :ref:`Fake Machine`

  Emulates the behavior of a tensile test machine, taking a position command as
  input and outputting the force and the displacement. Mainly used in the
  examples because it doesn't require any hardware, but may as well be used for
  debugging or prototyping.

  Refer to the `fake_test.py <https://github.com/LaboratoireMecaniqueLille/
  crappy/blob/master/Examples/fake_test.py>`_ example (among others) for a use
  case.

- :ref:`Sink`

  Discards any received data. Used for prototyping and debugging only.

  No example featuring a Sink block is currently distributed.

Supported hardware (cameras, inouts, actuators)
-----------------------------------------------

Supported cameras
+++++++++++++++++

- :ref:`Bi Spectral`

  An infrared camera acquiring on two wavelengths at the same time.

  .. Important::
     Only intended for an internal use in our laboratory as it is not
     commercially available.

- :ref:`Camera GStreamer`

  This camera object opens video streams using the Python binding of `Gstreamer
  <https://gstreamer.freedesktop.org/>`_. It can open a camera by path (in
  Linux) or number (in Windows and Mac), in which case the GStreamer pipeline
  is generated automatically. Alternatively, it can also open a stream
  following a custom pipeline given by the user.

  Compared to the :ref:`Camera OpenCV` camera, the GStreamer one is less
  CPU-intensive and is compatible with more devices. Its dependencies are
  however harder to install (especially on Windows) and it is harder to make it
  work properly.

- :ref:`Camera OpenCV`

  This camera object opens video streams using OpenCV. It allows tuning the
  device number, as well as the image format and the number of channels. It is
  mostly compatible with USB cameras, and its dependencies are straightforward
  to install.

- :ref:`Camera Link`

  Allows reading from a camera communicating over Camera Link plugged to a
  `microEnable 5 ironman AD8-PoCL <https://www.baslerweb.com/en/products/
  acquisition-cards/microenable-5-ironman/>`_ PCIexpress board. May as well
  work with similar boards.

  .. Important::
     This camera object relies on C++ libraries, which are not distributed with
     ``pip``. They are only available using a ``setup`` install, see
     :ref:`Installation` for details.

  .. Important::
     This camera object hasn't been tested on recent releases !

- :ref:`Fake Camera`

  Simply displays an animated image of a chosen size and at a given frequency.
  Doesn't require any hardware, used mainly for debugging and prototyping.

- :ref:`File Reader`

  Successively reads images already saved in a folder, and returns them. No
  image acquisition is performed and no hardware is required.

- :ref:`JAI`

  Allows reading from a `Jai GO-5000M-PMCL <https://www.jai.com/products/
  go-5000c-pmcl>`_ camera. It relies on the :ref:`Camera Link` object.

  .. Important::
     This camera object relies on C++ libraries, which are not distributed with
     ``pip``. They are only available using a ``setup`` install, see
     :ref:`Installation` for details.

  .. Important::
     This camera object hasn't been tested on recent releases !

- :ref:`PiCamera`

  Allows reading images from a PiCamera, using a Raspberry Pi.

  .. Important:: Can only be run on a Raspberry Pi !

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

- :ref:`Biaxe`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in speed. May as well work on
  other conditioners from the same brand, although it hasn't been tested.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Biotens`

  Drives JVL's `MAC140 <https://www.jvl.dk/276/integrated-servo-motors-mac050
  -141>`_ integrated servomotor in speed or in position. Probably works with
  other integrated servomotors from JVL, although it hasn't been tested.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`CM Drive`

  Drives Schneider Electric's `MDrive 23 <https://www.novantaims.com/downloads
  /quickreference/mdi23plus_qr.pdf>`_ stepper motor in speed or in position.
  Probably works with other stepper motors in the same range of products,
  although it hasn't been tested.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Fake Motor`

  Emulates the dynamic behavior of a DC motor, but doesn't drive any hardware.
  Used in the examples, may also be used for prototyping or debugging.

- :ref:`Motor kit pump`

  Drives Adafruit's `DC & Stepper Motor HAT for Raspberry Pi <https://www.
  adafruit.com/product/2348>`_ in Volts, using Adafruit's Blinka library.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Oriental`

  Drives Oriental Motor's `ARD-K <https://catalog.orientalmotor.com/item/s-
  closed-loop-stepper-motor-drivers-dc-input/ard-closed-loop-stepper-driver-
  pulse-input-dc/ard-k>`_ stepper motor driver in speed or in position. Probably
  works with other stepper motor drivers in the same range of products, although
  it hasn't been tested.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`Pololu Tic`

  Drives Pololu's `Tic <https://www.pololu.com/category/212/tic-stepper-motor-
  controllers>`_ stepper motor drivers in speed or in position. Designed for
  driving all the Tic drivers, but tested only on the 36v4 model.

- :ref:`Servostar`

  Drives Kollmorgen's `Servostar 300 <https://www.kollmorgen.com/en-us/products
  /drives/servo/s300/>`_ servomotor conditioner in position or sets it to the
  analog driving mode. This is the same conditioner as for the :ref:`Biaxe`
  actuator, but this object was designed for an other application.

  .. Important::
     This actuator was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`TRA6PPD`

  Drives Newport's `TRA6PPD <https://www.newport.com/p/TRA6PPD>`_ miniature
  linear stepper motor actuator, in speed or in position.

Supported Sensors and outputs
+++++++++++++++++++++++++++++

Acquisition boards
""""""""""""""""""

- :ref:`Labjack T7`

  Controls Labjack's `T7 <https://labjack.com/products/t7>`_ acquisition board.

- :ref:`Labjack UE9`

  Controls Labjack's `UE9 <https://labjack.com/products/calibration-service-
  with-cert-u6-ue9-t7>`_ acquisition board.

- :ref:`Labjack T7 Streamer`

  Controls Labjack's `T7 <https://labjack.com/products/t7>`_ acquisition board
  in streaming mode.

- :ref:`Waveshare AD/DA`

  Controls Waveshare's `AD/DA <https://www.waveshare.com/product/raspberry-pi/
  hats/ad-da-audio-sensors/high-precision-ad-da-board.htm>`_ Raspberry Pi
  acquisition hat. May be used from any device with a proper wiring, but more
  convenient to use from a Raspberry Pi. Communicates over SPI.

- :ref:`Waveshare AD/DA FT232H`

  Almost the same code as :ref:`Waveshare AD/DA`, except it's been optimized for
  use with a :ref:`FT232H` USB to SPI adapter. Communicates over SPI.

- :ref:`Waveshare High Precision`

  Controls Waveshare's `High Precision HAT
  <https://www.waveshare.com/18983.htm>`_ Raspberry Pi acquisition hat. It
  features a 10-channels 32 bits ADC. It may be used from any device able to
  communicate over SPI, but is originally meant for interfacing with a
  Raspberry Pi.

Sensors
"""""""

- :ref:`ADS1115`

  Reads voltages from Adafruit's `ADS 1115 <https://www.adafruit.com/product/
  1085>`_ ADC. Communicates over I2C.

- :ref:`Agilent 34420A`

  Reads voltages or resistances from Agilent's `34420A <https://www.keysight.
  com/us/en/product/34420A/micro-ohm-meter.html>`_ precision multimeter.
  Communicates over serial.

- :ref:`MCP9600`

  Reads temperatures from Adafruit's `MCP9600 <https://www.adafruit.com/product
  /4101>`_ thermocouple amplifier. Communicates over I2C.

- :ref:`MPRLS`

  Reads pressures from Adafruit's `MPRLS <https://www.adafruit.com/product/
  3965>`_ pressure sensor. Communicates over I2C.

- :ref:`NAU7802`

  Reads voltages from Sparfun's `'Qwiic Scale' NAU7802 <https://www.sparkfun.
  com/products/15242>`_ load cell conditioner. Communicates over I2C.

- :ref:`PiJuice`

  Reads the charging status and battery level of Kubii's `PiJuice <https://
  www.kubii.fr/14-chargeurs-alimentations-raspberry/2019-pijuice-hat-kubii-
  3272496008793.html>`_ Raspberry Pi power supply.

  .. Important::
     This inout was written for a specific application, so it may not be
     usable as-is in the general case.

- :ref:`OpSens`

  Reads data from OpSens' `single channel signal conditioner <https://opsens-
  solutions.com/products/signal-conditioners-oem-boards/picosens/>`_ for
  fiber-optics temperature, strain, pressure or position measurement.
  Communicates over serial.

- :ref:`Spectrum`

  Reads voltages from Spectrum's `M2i 4711 EXP <https://spectrum-
  instrumentation.com/products/details/M2i4711.php>`_ high-speed ADC
  communicating over PCIexpress.

Multi-device drivers
""""""""""""""""""""

- :ref:`Comedi`

  Reads voltages from an `USB-DUX Sigma <https://github.com/glasgowneuro/usbdux/
  tree/main/usbdux-sigma>`_ ADC (not manufactured anymore) using the `Comedi
  <https://www.comedi.org/>`_ driver. The code was written to work as-is on
  other acquisition boards supporting the Comedi driver, but this hasn't been
  tested. Communicates over serial.

- :ref:`NI DAQmx`

  Controls National Instrument's `USB 6008 <https://www.ni.com/en-us/support/
  model.usb-6008.html>`_ DAQ module using the :mod:`nidaqmx` module. The code
  was written to work as-is on other National Instruments acquisition modules,
  but this hasn't been tested. Communicates over USB.

- :ref:`DAQmx`

  Same as :ref:`NI DAQmx`, except it relies on the :mod:`PyDAQmx` module. The
  differences between the two modules weren't further investigated.

Outputs
"""""""

- :ref:`GPIO Switch`

  Drives a single GPIO on a Raspberry Pi.

  .. Important:: Only works on a Raspberry Pi !

- :ref:`GPIO PWM`

  Controls a PWM output on a single GPIO of a Raspberry Pi.

  .. Important:: Only works on a Raspberry Pi !

- :ref:`GSM`

  Uses Waveshare's `GSM/GPRS/GNSS/Bluetooth hat <https://www.waveshare.com/
  gsm-gprs-gnss-hat.htm>`_ for sending SMS. The other functionalities are not
  implemented. Usable from any device with a proper wiring, but more convenient
  to use with a Raspberry Pi. Communicates over serial.

  .. Important::
     This inout was written for a specific application, so it may not be
     usable as-is in the general case.

Enhanced actuators
""""""""""""""""""

- :ref:`Kollmorgen`

  Drives Kollmorgen's `AKD PDMM <https://www.kollmorgen.com/en-us/products/
  drives/servo/akd-pdmm/akd-pdmm-programmable-drive-multi-axis-master/>`_
  servomotor controller. As this device supports many settings, it was decided
  to consider it as an inout to fully take advantage of its versatility.

  .. Important::
     This inout was written for a specific application, so it may not be
     usable as-is in the general case.

Real-time data processing (Modifiers)
-------------------------------------

- :ref:`Demux`

  Takes the signal returned by a streaming device and transforms it into a
  signal similar to the one of a regular device. This modifier is mandatory for
  plotting data from a streaming device.

- :ref:`Differentiate`

  Calculates the time derivative of a given label.

- :ref:`Integrate`

  Integrates a given label over time.

- :ref:`Mean`

  Returns the mean value of a label over a given number of points. Only returns
  a value once every number of points.

- :ref:`Median`

  Returns the median value of a label over a given number of points. Only
  returns a value once every number of points.

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

  Returns the received label only if the value is in a predefined list.
