============================
Crappy for embedded hardware
============================

Concept and goals
-----------------

Accessibility to embedded devices
+++++++++++++++++++++++++++++++++

Over the last decades, huge progress has been made in the accessibility of the
general public to embedded electronic devices, pushed by companies like
`Adafruit <https://www.adafruit.com/>`_ or `SparkFun <https://www.sparkfun.
com/>`_. Anyone can now easily find and order sensors and actuators online,
often for a few dozen euros or even less. Additionally, many suppliers also
provide extensive tutorials, wiring charts, or even readily-usable Python
modules, usually for free !

These suppliers propose hardware that can be of interest for experimental
setups, like Digital to Analog Converters (DAC), Analog to Digital Converters
(ADC), thermocouple and load cell conditioners, DC and stepper motor drivers,
etc. Most often, they come integrated on Printed Circuit Boards (PCB) and
interface over I2C or SPI, two communication protocols commonly used between
integrated components. While regular PCs cannot communicate over SPI or I2C,
microcontrollers or Single Board Computers (SBC) like the `Raspberry Pi
<https://www.raspberrypi.org/>`_ usually does.

A quick comparison
++++++++++++++++++

Note that the equipment supplied by embedded electronics websites is precisely
what companies like `National Instruments <https://www.ni.com/en-us.html>`_
(NI), that are explicitly intended for researchers, also sell. But 10 to 100
times cheaper. This price difference can be explained, let's take the example of
an ADC. On the one hand NI sells an `8-channels 12-bits ADC <https://www.ni.com
/docs/en-US/bundle/ni-9201-specs/page/specifications.html>`_ with a nice
packaging, that can interface over USB with a PC, can acquire voltages from -10
to 10V, and is controlled from a user-friendly Graphical User Interface (GUI).
It can also acquire data at frequencies up to 500k samples per second. On the
other hand Adafruit sells a `4-channels 16-bits ADC
<https://www.adafruit.com/product/1085>`_ on a raw PCB, that can interface with
SBCs or microcontrollers, can only acquire voltages from 0 to 3.3V, and is
controlled using a Python script. It can read data at 'only' around 800 samples
per second. But NI's costs 600$ while Adafruit's costs 15$, 20 times cheaper.

While NI's ADC is definitely more user-friendly and can acquire at tremendous
sample rates, Adafruit's is actually theoretically more precise (16 bits instead
of 12) and still has 4 channels and a consistent sample rate. Some applications
may require the high frequency of NI's device, but Adafruit's is actually
well-suited in many situations. Only the limitation in the accessible voltage
range may though be hard to overcome.

Embedded devices in Crappy
++++++++++++++++++++++++++

We believe that is wouldn't be relevant to buy NI's device when much cheaper
ones could be suitable, especially when budget is a strong constraint. However,
we're also aware that working with the I2C or SPI protocols is far from being
straightforward for a majority of people, and that not being able to interface
with a PC is a strong drawback. To address these limitations, we chose to
**make low-cost embedded devices easily accessible using Crappy, either from**
**an SBC, a microcontroller or even a PC !**

To this end, Crappy contains several features opening many possibilities to
interface with embedded electronics:

- It is fully compatible with a use on SBCs like the Raspberry Pi
- The :ref:`FT232H` tool allows interfacing over SPi or I2C directly from a PC,
  using Adafruit's FT232H USB to I2C and SPI adapter. Examples of code are
  provided below.
- The :ref:`UController` block, along with the `microcontroller.py
  <https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/crappy/tool/
  microcontroller.py>`_ and `microcontroller.ino <https://github.com/
  LaboratoireMecaniqueLille/crappy/blob/master/crappy/tool/
  microcontroller.ino>`_ templates are provided for controlling the execution
  of a script on a microcontroller from a PC, sending commands and receiving
  data.
- The :ref:`Client Server` block allows receiving from and sending data to a
  remote device connected on a common network, e.g. a microcontroller acquiring
  data in an enclosed area.

Specific blocks and features
----------------------------

As stated above, a variety of features dedicated to the control of embedded
devices have been included to Crappy. In this section, they will be detailed
and examples of code will be provided. Note that this section is not a tutorial.
It should be seen as a complement to the existing :ref:`Tutorials` in the
specific situation when the user deals with embedded devices. It can also be
considered as an overview of Crappy's capabilities and interface in such
situation.

Devices interfacing over I2C or SPI
+++++++++++++++++++++++++++++++++++

As demonstrated in the first section, numerous sensors and actuators well-suited
for a use in research setups can be purchased from embedded electronics
suppliers. These devices usually communicate over SPI or I2C, and can thus not
directly interface with regular PCs. The following devices are currently part of
the framework:

- `Adafruit's ADS1115 <https://www.adafruit.com/product/1085>`_ ADC
- `Adafruit's DC motor HAT <https://www.adafruit.com/product/2348>`_
- `Waveshare's GSM HAT <https://www.waveshare.com/gsm-gprs-gnss-hat.htm>`_
- `Adafruit's MCP9600 <https://www.adafruit.com/product/4101>`_ thermocouple
  amplifier
- `Adafruit's MPRLS <https://www.adafruit.com/product/3965>`_ pressure sensor
- `SparkFun's NAU7802 <https://www.sparkfun.com/products/15242>`_ load cell
  conditioner
- `The PiJuice <https://uk.pi-supply.com/products/pijuice-standard>`_ Raspberry
  Pi power platform
- `Waveshare's AD/DA <https://www.waveshare.com/high-precision-ad-da-
  board.htm>`_ ADC and DAC HAT
- `The PiCamera <https://www.raspberrypi.com/products/camera-module-v2/>`_
- `Pololu's Tic <https://www.pololu.com/product/3140>`_ stepper motor
  controllers

These devices can be integrated in two different ways into setups driven with
Crappy. These two options are presented below.

Using the GPIOs of an SBC
"""""""""""""""""""""""""

.. |ge| unicode:: U+2265

The most straightforward way to interface with hardware over SPI or I2C is to
use an SBC. Unlike the regular PCs, these computers are specifically designed
to communicate with hardware over low-level protocols. On the Raspberry Pi, it
is for example possible to control GPIOs, generate PWM signals, and several
I2C, SPI and serial buses are available.

Crappy is fully able to run on the Raspberry Pi 3 and 4, and should as well be
compatible with any SBC that can run Python |ge| 3.6, although that wasn't
tested. On these SBCs, hardware pins are dedicated to low-level communication
and the wiring is usually extremely simple. The Python modules :mod:`spidev`
and :mod:`smbus2` are also available, making it quite easy to issue SPI or I2C
commands in Python.

Consequently, there is no particular difficulty nor specificity to consider when
using embedded sensors or actuators on an SBC. Wiring the devices to the GPIOs
may be new to beginners, but it is actually extremely simple and much
documentation about it can be found on internet. There's also nothing special to
consider when writing Crappy's script, except that the keyword argument
`backend` should be set to ``Pi4`` or ``Blinka``. Here's a basic example of
code for reading data from an NAU7802 load cell conditioner on a Raspberry
Pi 4, and displaying it on a graph.

.. code-block:: python

   import crappy

   if __name__ == "__main__":

       nau = crappy.blocks.IOBlock('NAU7802',
                                   labels=['t(s)', 'out(V)'],
                                   backend='Pi4')

       graph = crappy.blocks.Grapher(('t(s)', 'out(V)'))

       crappy.link(nau, graph)

       crappy.start()

Using the FT232H tool
"""""""""""""""""""""

While SBCs are great for interfacing with embedded devices, they usually display
a limited computing performance and are thus poorly suited for computationally
intensive tests. For instance, they may not be powerful enough to perform
complex real-time image analysis at a high frame rate, or to display many
graphers at the same time in Crappy. In such situations, the computing
capabilities of a regular PC would be required.

As we didn't want to have to choose between the high performance of a PC and the
flexibility of embedded devices and SBCs, we incorporated `Adafruit's FT232H
<https://www.adafruit.com/product/2264>`_ USB to GPIO, serial, I2C and SPI
converter into Crappy. Using this board, it is possible to interface over I2C
and SPI from any PC as long as a USB port is available ! It still requires a
proper wiring on the FT232H's GPIOs, just like on the SBCs.

Although the implementation of the FT232H in Crappy is quite complex, it is all
kept under the hood and doesn't change much from the user's perspective. If only
one FT232H is connected, then the code given in the last section would become :

.. code-block:: python
   :emphasize-lines: 7

   import crappy

   if __name__ == "__main__":

       nau = crappy.blocks.IOBlock('NAU7802',
                                   labels=['t(s)', 'out(V)'],
                                   backend='ft232h')

       graph = crappy.blocks.Grapher(('t(s)', 'out(V)'))

       crappy.link(nau, graph)

       crappy.start()

.. Note::
   In Linux, the udev-rules must first be set before being able to communicate
   with the FT232H. This can be done using `an utility program <https://github.
   com/LaboratoireMecaniqueLille/crappy/blob/master/util/udev_rule_setter.sh>`_
   we developed.

It gets trickier when several FT232H are connected to a same computer, as it is
then necessary to specify for each device the serial number of the FT232H on
which it is connected. Note that there's no limit to the number of FT232H that
can be simultaneously plugged to a PC, and several devices can share a same bus
on a given FT232H. The only restriction is that one given FT232H can only
operate over either SPI or I2C, not both simultaneously. Here's an example of
code for a setup featuring two FT232H :

.. code-block:: python
   :emphasize-lines: 8,10-16,19

   import crappy

   if __name__ == "__main__":

       nau = crappy.blocks.IOBlock('NAU7802',
                                   labels=['t(s)', 'out(V)'],
                                   backend='ft232h',
                                   serial_nr='54321')

       ads = crappy.blocks.IOBlock('ADS1115',
                                   labels=['t(s)', 'U(V)'],
                                   backend='ft232h',
                                   serial_nr='12345')

       graph1 = crappy.blocks.Grapher(('t(s)', 'out(V)'))
       graph2 = crappy.blocks.Grapher(('t(s)', 'U(V)'))

       crappy.link(nau, graph1)
       crappy.link(ads, graph2)

       crappy.start()

Now how to get the serial number of an FT232H ? Well they do not come with a
pre-defined number, it is up to the user to set it. Fortunately, we developed a
short program that does it, it can be found `here <https://github.com/
LaboratoireMecaniqueLille/crappy/blob/master/util/Set_ft232h_serial_nr.py>`_.
To get the serial number of an FT232H that was already given one, the command
``usb-devices`` can be run in Linux.

.. Note::
   Because of limitations in the underlying `libsub` module, it is not possible
   to simultaneously use an FT232H and to communicate with a Pololu Tic using
   the ``USB`` backend. It is however still possible to use the ``ticcmd``
   backend.

Interfacing with microcontrollers
+++++++++++++++++++++++++++++++++

Why using microcontrollers ?
""""""""""""""""""""""""""""

As detailed above, interfacing embedded devices with Crappy on SBCs or PCs is
a powerful way to create setups in a more versatile and cost-effective way.
However, a strong limitation remains. Because of the way the OS are designed,
computers have to handle numerous processes running at the same time. To do so,
all the processes are constantly being interrupted by each other, making all of
them actually run intermittently. Consequently, there's no guaranty that a given
process is awake at a given moment, and this also applies to Crappy's processes.
Depending on the OS, the machine running it, and other nerdy parameters,
processes might sleep up to a few milliseconds in a row !

For many applications this is not a big deal, but in specific cases it can
become extremely limiting. For example if a signal needs to be generated at
several hundred Hz, its shape would be strongly affected. Or if a trigger has
to be sent within a short delay after an event occurs, the required
responsiveness might not be achieved. To overcome this limitation,
microcontrollers are a nice option. As they can only run one process, it is
never interrupted and extremely high looping frequencies might be achieved.
Moreover, microcontrollers often include many features for interfacing with
hardware like GPIOs, I2C, SPI, serial, PWM, WiFi, etc., making their integration
into setups very straightforward. The most powerful microcontrollers can even
run MicroPython, a lighter version of Python, making it easy to handle even for
beginners. Otherwise C code has to be used, which requires far more advanced
programming skills.

Microcontrollers in Crappy
""""""""""""""""""""""""""

Usually, microcontrollers are meant to run a script as soon they're powered on,
independently from any external input. It means that they cannot be started or
stopped by Crappy, which may be problematic. To address this issue, we developed
a MicroPython template, an Arduino template, and the UController Crappy block
for the situations when a microcontroller is linked to a PC through a serial
connection (USB cable). They allow communication between the microcontroller
and the PC during a test, and they also manage the beginning and the end of the
test.

The Micropython and Arduino templates and the UController block actually work
in very similar ways. They regularly listen to the serial connection, and read
any data sent from the other side. A specific syntax allows sending labeled
inputs to the microcontroller, for example to modify the value of a parameter.
Reversely, this syntax also allows the microcontroller to send back data or
feedback. On startup, a blocking call prevents the microcontroller from doing
anything until the UController block pings it, during `crappy.prepare()`. At
the end of the test, the microcontroller is reset to stop the script currently
running.

.. Note::
   This documentation is not meant to explain how to flash MicroPython on a
   microcontroller, nor how to upload MicroPython or C code to it. The specific
   constraints entailed by coding for microcontrollers are also not covered
   here. For more information on each of these topics, refer to the dedicated
   and extensive documentation that can be found on internet.

The syntax for using the UController block is not any different from the syntax
for the other blocks. Let's take the example of a microcontroller running a PID
loop for controlling a DC motor, with a variable target speed that we'll call
`cmd_speed`. Let's also assume that the microcontroller should return once in a
while the current motor speed, `cur_speed`. The most difficult part is to write
the script that will run on the microcontroller. It won't be covered here, but
using MicroPython rather than C and starting from the microcontroller.py
template may make it easier to write. On the PC, the UController block simply
needs to send the command, and to return the current speed. A :ref:`Generator`
block is needed for generating the command, and a :ref:`Dashboard` can be used
for reading the output. An example of code is presented here :

.. code-block:: python

   import crappy

   if __name__ == "__main__":

       gen = crappy.blocks.Generator([{'type': 'constant',
                                       'speed': 2000,
                                       'condition': 'delay=10'},
                                      {'type': 'constant',
                                       'speed': 3000,
                                       'condition': 'delay=20'},
                                      {'type': 'constant',
                                       'speed': 4000,
                                       'condition': 'delay=10'},
                                      {'type': 'constant',
                                       'speed': 3000,
                                       'condition': 'delay=5'}],
                                     cmd_label='cmd_speed')

       micro = crappy.blocks.UController(labels=['cur_speed'],
                                         cmd_labels=['cmd_speed'],
                                         init_output={'cur_speed': 0},
                                         port='/dev/ttyUSB0')

       dash = crappy.blocks.Dashboard(labels=['t(s)', 'cur_speed'])

       crappy.link(gen, micro)
       crappy.link(micro, dash)

       crappy.start()

Interfacing with remote devices over MQTT
+++++++++++++++++++++++++++++++++++++++++

An interesting feature of microcontrollers is that many of them can connect to
a WiFi network, or even generate it. Rather than exchanging data over serial
with the UController block, it is then possible to do it remotely without any
cable linking the PC to the microcontroller. This can prove extremely
convenient, for example for acquiring data from a fully enclosed area, or on
rotating parts, or if the PC cannot be placed close enough to the sensor for any
reason. To this end, the :ref:`Client Server` block allowing to communicate
remotely over a network was developed. Note that this block can also be used to
communicate with devices other than microcontrollers, like PCs. For instance we
used this block for following a long-lasting test remotely from our personal
computers over a university network.

The Client Server block uses the MQTT protocol to send and receive messages. It
can subscribe to topics, and receive the associated messages, and also publish
messages in topics. The program that manages the messages from the different
devices is the MQTT broker, which runs on one machine only. Many brokers exist,
with each their strengths and weaknesses. The broker runs independently from
Crappy, although we added the possibility to start and stop the `Mosquitto
<https://mosquitto.org/>`_ broker from Crappy. The block itself is quite similar
to all the other Crappy blocks, except it sends the data to a broker rather than
to a device. An example of code is presented here, which sends data from the
`to_send` label to the topic of the same name, and retrieves data from the
`to_receive` topic to the label of the same name. It assumes that the broker
runs at the IP address `192.0.2.1` and listens to the port 1148. It also
assumes that a remote device publishes in the topic `to_receive`.

.. code-block:: python

   import crappy

   if __name__ == "__main__":

       gen = crappy.blocks.Generator([{'type': 'ramp',
                                       'speed': 1,
                                       'condition': None}],
                                     freq=50,
                                     cmd_label='to_send')

       mqtt = crappy.blocks.ClientServer(address='192.0.2.1',
                                          port=1148,
                                          topics=[('to_receive',)],
                                          init_output={'to_receive': 0},
                                          cmd_labels=[('to_send',)])

       graph = crappy.blocks.Grapher(('t(s)', 'to_receive'))

       crappy.link(gen, mqtt)
       crappy.link(mqtt, graph)

       crappy.start()

If now the values of the label `to_send` have to be sent along with their
timestamp `t(s)`, the code can be modified as follows to send the timestamp as
`t_here` to the broker. This way it won't be mistaken with another `t(s)` label
if it is received by another Crappy program. Here we also assume that a remote
device sends the timestamp `t_remote` along with `to_receive`.

.. code-block:: python
   :emphasize-lines: 12-19

   import crappy

   if __name__ == "__main__":

       gen = crappy.blocks.Generator([{'type': 'ramp',
                                       'speed': 1,
                                       'condition': None}],
                                     freq=50)

       mqtt = crappy.blocks.ClientServer(address='192.0.2.1',
                                          port=1148,
                                          topics=[('t_remote', 'to_receive')],
                                          init_output={'t_remote': 0,
                                                       'to_receive': 0},
                                          cmd_labels=[('t(s)', 'to_send')],
                                          labels_to_send=[('t_here',
                                                           'to_send')])

       graph = crappy.blocks.Grapher(('t_remote', 'to_receive'))

       crappy.link(gen, mqtt)
       crappy.link(mqtt, graph)

       crappy.start()

Adding embedded devices to Crappy
---------------------------------

Adding embedded devices to Crappy is in nothing different from adding any other
device. However for the devices interfacing over SPI or I2C, additional
information can be given compared to the general case. This section comes then
in complement to the :ref:`Tutorials`.

Based on an existing Python module
++++++++++++++++++++++++++++++++++

Additionally to the hardware they sell, some companies like Adafruit also
provide Python modules for driving it. In Adafruit's case this module is called
`Blinka <https://circuitpython.org/blinka>`_, and can be installed simply using
``pip``. A limited number of commands can then be used to fully control devices,
all the complexity being kept under the hood of Blinka. Using this little set of
commands, codes for driving components from Crappy can be kept extremely
short, making even beginners fully able to write them. Note that we focus here
on Blinka, but this is also true for any similar module.

For the sake of the example, let's create from scratch a minimal version of
the :ref:`MPRLS` code. It is a pressure sensor, so it belongs to the
:ref:`In / Out` category of Crappy. Let's start from the template for InOuts
provided :ref:`here <1.d. inouts>`. Here we only want to acquire data, so the
``set_cmd`` method should be removed.

.. code-block:: python

   import crappy
   import time

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           pass

       def get_data(self):
           return [time.time(), 0]

       def close(self):
           pass

Then according to `Adafruit's documentation <https://learn.adafruit.com/adafruit
-mprls-ported-pressure-sensor-breakout/python-circuitpython>`_, we have to
import the :mod:`board` and :mod:`adafruit_mprls` modules to be able to use the
MPRLS. The object representing the sensor then has to be initialized in the
``open`` method. It gives:

.. code-block:: python
   :emphasize-lines: 3-4, 12

   import crappy
   import time
   import adafruit_mprls
   import board

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           self._mpr = adafruit_mprls.MPRLS(board.I2C(), psi_min=0, psi_max=25)

       def get_data(self):
           return [time.time(), 0]

       def close(self):
           pass

The only action that should be performed is to simply return the pressure value.
Still according to the online documentation, this value can be acquired using
the ``pressure`` attribute. There's thus only one replacement to do :

.. code-block:: python
   :emphasize-lines: 15

   import crappy
   import time
   import adafruit_mprls
   import board

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           self._mpr = adafruit_mprls.MPRLS(board.I2C(), psi_min=0, psi_max=25)

       def get_data(self):
           return [time.time(), self._mpr.pressure]

       def close(self):
           pass

And that's it ! The sensor can now be read extremely easily using the following
code :

.. code-block:: python
   :emphasize-lines: 20-28

   import crappy
   import time
   import adafruit_mprls
   import board

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()

       def open(self):
           self._mpr = adafruit_mprls.MPRLS(board.I2C(), psi_min=0, psi_max=25)

       def get_data(self):
           return [time.time(), self._mpr.pressure]

       def close(self):
           pass

   if __name__ == "__main__":

       mprls = crappy.blocks.IOBlock('Mprls_mini', labels=['t(s)', 'pressure'])

       graph = crappy.blocks.Grapher(('t(s)', 'pressure'))

       crappy.link(mprls, graph)

       crappy.start()

Based on a datasheet
++++++++++++++++++++

Unfortunately, not every supplier provides a Python module for their products,
sometimes even no code at all. Often, code developed by individuals may still be
available somewhere on internet, for example on `Github <https://github.com/>`_.
If no code at all can be found, the only option left is to follow the guidelines
of the datasheet. This is pretty cumbersome, and requires a good knowledge of
the SPI or I2C protocols. It is thus not recommended to beginners, or maybe only
to the very motivated ones.

Let's come back to the example of the MPRLS pressure sensor, and suppose we want
our code to be independent from Adafruit's modules. We know that the sensor
interfaces over I2C, so a good option is to use the :mod:`smbus2` module. For
a device interfacing over SPI the :mod:`spidev` module can be used. The first
thing to do is to initialize the bus, and to close it at the end of the program.
It is done as follows supposing that the bus n°1 is used, which is the default
one on the Raspberry Pi.

.. code-block:: python
   :emphasize-lines: 3,9,18

   import crappy
   import time
   import smbus2

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()
           self._bus = smbus2.SMBus(1)

       def open(self):
           pass

       def get_data(self):
           return [time.time(), 0]

       def close(self):
           self._bus.close()

Now we need to have a look at the datasheet to know exactly how to communicate
with the sensor. The datasheet can be found at `this address <https://prod-
edam.honeywell.com/content/dam/honeywell-edam/sps/siot/en-us/products/sensors/
pressure-sensors/board-mount-pressure-sensors/micropressure-mpr-series/
documents/sps-siot-mpr-series-datasheet-32332628-ciid-172626.pdf>`_, the
section of interest starts on page 15. First, there's no need to initialize any
parameter during ``open``, it can then be left as is. According to the
datasheet, two steps are mandatory when reading the pressure from the sensor:
first the three bytes ``0xAA, 0x00, 0x00`` should be written to the device, and
when the data is ready it can be retrieved by reading 4 bytes from the sensor.
After the writing operation, reading only 1 byte from the device allows to know
if data is ready, or it is also possible to simply wait for 5ms and the data
will then be ready for sure.

The commands of the :mod:`smbus2` module won't be detailed here, but more
information can be found on its `PyPi page <https://pypi.org/project/smbus2/>`_
or on `ReadTheDocs <https://smbus2.readthedocs.io/en/latest/>`_. To start a
conversion and read the result, only a few lines are necessary :

.. code-block:: python
   :emphasize-lines: 15-24

   import crappy
   import time
   import smbus2

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()
           self._bus = smbus2.SMBus(1)

       def open(self):
           pass

       def get_data(self):
           # Starting conversion
           self._bus.i2c_rdwr(smbus2.i2c_msg.write(0x18, [0xAA, 0x00, 0x00]))
           # Waiting for conversion to complete
           time.sleep(0.005)
           # Reading conversion result
           read = smbus2.i2c_msg.read(0x18, 4)
           self._bus.i2c_rdwr(read)
           # Extracting conversion result as an integer
           out = list(read)[1:]
           ret = (out[0] << 16) | (out[1] << 8) | out[2]

           return [time.time(), 0]

       def close(self):
           self._bus.close()

The last step is to return the result in hPa. This is done following the formula
given in the datasheet. It gives :

.. code-block:: python
   :emphasize-lines: 25-28

   import crappy
   import time
   import smbus2

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()
           self._bus = smbus2.SMBus(1)

       def open(self):
           pass

       def get_data(self):
           # Starting conversion
           self._bus.i2c_rdwr(smbus2.i2c_msg.write(0x18, [0xAA, 0x00, 0x00]))
           # Waiting for conversion to complete
           time.sleep(0.005)
           # Reading conversion result
           read = smbus2.i2c_msg.read(0x18, 4)
           self._bus.i2c_rdwr(read)
           # Extracting conversion result as an integer
           out = list(read)[1:]
           ret = (out[0] << 16) | (out[1] << 8) | out[2]
           # Converting to hPa
           pres = 68.947572932 * (ret - 0x19999A) * 25 / (0xE66666 - 0x19999A)

           return [time.time(), pres]

       def close(self):
           self._bus.close()

Finally, the code can be improved by checking if the conversion is ready rather
than waiting 5ms. This way greater sample rates can be achieved.

.. code-block:: python
   :emphasize-lines: 18-22

   import crappy
   import time
   import smbus2

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()
           self._bus = smbus2.SMBus(1)

       def open(self):
           pass

       def get_data(self):
           # Starting conversion
           self._bus.i2c_rdwr(smbus2.i2c_msg.write(0x18, [0xAA, 0x00, 0x00]))
           # Waiting for conversion to complete
           while True:
               wait = smbus2.i2c_msg.read(0x18, 1)
               self._bus.i2c_rdwr(wait)
               if not list(wait)[0] & 0x20:
                   break
           # Reading conversion result
           read = smbus2.i2c_msg.read(0x18, 4)
           self._bus.i2c_rdwr(read)
           # Extracting conversion result as an integer
           out = list(read)[1:]
           ret = (out[0] << 16) | (out[1] << 8) | out[2]
           # Converting to hPa
           pres = 68.947572932 * (ret - 0x19999A) * 25 / (0xE66666 - 0x19999A)

           return [time.time(), pres]

       def close(self):
           self._bus.close()

As demonstrated here, writing the code using the datasheet is a bit complex and
necessitates a good knowledge of both the I2C protocol and the associated Python
library. It is nevertheless still accessible to anyone with a bit of patience
and motivation. Let's now read the sensor using the fully functional code !

.. code-block:: python
   :emphasize-lines: 37-45

   import crappy
   import time
   import smbus2

   class Mprls_mini(crappy.inout.InOut):

       def __init__(self):
           super().__init__()
           self._bus = smbus2.SMBus(1)

       def open(self):
           pass

       def get_data(self):
           # Starting conversion
           self._bus.i2c_rdwr(smbus2.i2c_msg.write(0x18, [0xAA, 0x00, 0x00]))
           # Waiting for conversion to complete
           while True:
               wait = smbus2.i2c_msg.read(0x18, 1)
               self._bus.i2c_rdwr(wait)
               if not list(wait)[0] & 0x20:
                   break
           # Reading conversion result
           read = smbus2.i2c_msg.read(0x18, 4)
           self._bus.i2c_rdwr(read)
           # Extracting conversion result as an integer
           out = list(read)[1:]
           ret = (out[0] << 16) | (out[1] << 8) | out[2]
           # Converting to hPa
           pres = 68.947572932 * (ret - 0x19999A) * 25 / (0xE66666 - 0x19999A)

           return [time.time(), pres]

       def close(self):
           self._bus.close()

   if __name__ == "__main__":

       mprls = crappy.blocks.IOBlock('Mprls_mini', labels=['t(s)', 'pressure'])

       graph = crappy.blocks.Grapher(('t(s)', 'pressure'))

       crappy.link(mprls, graph)

       crappy.start()
