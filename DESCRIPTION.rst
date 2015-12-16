=======================
Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=======================

.. contents::


What is CRAPPY?
---------------

CRAPPY is an open-source software developped to provide easy-to-use tools 
for command and acquisition on complex experimental setups.
As every experimenter knows, designing complex setups and measuring physical
phenomenons with precision can be tricky. As we increase the number of sensors
to better understand what is really happening during a test, we need to have
simple tools to synchronise and quickly adapt a test sequence to new hardware.


How does it work?
-----------------

CRAPPY is a dataflow programming framework, allowing to write a new sequence 
easily by describing a sketch of your setup. Some classical part are already
implemented, as a signal generator, real-time graphs, and save functions.
CRAPPY provide a framework to add custom methods and ensure its compatibilty 
with the other parts.

CRAPPY keywords are :
  - interchangeable : allowing the user to switch between several hardware 
    without re-writing all the sequence.
  - independance : dissociate the different parts of the setup, especially the
    acquisition and the control.
  - synchronous : provide a common time-reference for all.
  - simultaneous : each part of the software is parallelized.


Structure
---------

CRAPPY is composed of 2 main parts : 

* A library part, containing:
  - Sensors : each sensors methods are available here.
  - Actuators : each actuators methods are available here.
  - Technicals : some hardware are both a sensor AND an actuator. Methods 
    common to both are available here

* A directly usable part, containing:
  - Blocks : blocks are independant parts. Each one of them run in a different
    process, and they use the methods available in the library part. They 
    communicate with each other through Links.
  - Links : links are connections between Blocks, as you graphically could 
    represent them as a line between 2 blocks. They send data from one to another,
    and can be customized with condition to modify the data or control when to
    send it.


Examples
--------

Examples of working sequences can be found in the Examples directory. Most of them
recquire specific hardware to work, so they may not all work.
