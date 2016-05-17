Physical objects
================

As Crappy is originaly designed to control tests, the core of our framework
is the hardware we work with.

We chose to divide it in three different categories.

Sensors
-------
In Crappy, the sensors represent everything that can **acquire** a physical
signal. It can be an acquisition card, but also a camera, a thermocouple...

Actuators
---------
On the other hand, actuators represent all the objects that can **interact on 
the other part of the test**, and can be controled. The most common example are 
motors.

Technicals
----------
Some hardware is **both a sensor and an actuator** by our definitions. This is for 
example the case of a variable-frequency drive : they can set the speed of the
motor (the *actuator* part), but most of them can also read the position or the
speed of the motor the *sensor* part).