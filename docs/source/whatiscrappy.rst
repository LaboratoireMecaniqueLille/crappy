=================
What is Crappy ?
=================

|Downloads|
|Documentation status|
|PyPi version|

Overview
--------

CRAPPY is an acronym and stands for Command and Real-time Acquisition in
Parallelized PYthon.

Crappy is developed at the `LaMCube <https://lamcube.univ-lille.fr/>`_, a
mechanical research laboratory based in Lille, France to provide a powerful and
easy-to-use framework for materials testing.

In order to understand the mechanical behaviour of materials, we tend to perform
tests with more and more sensors and actuators from various suppliers. There's
thus an increasing need to drive these devices in a synchronized way while
managing the high complexity of the setups.

As we are one step ahead of industrials, the commercially available testing
solutions may also not be well-suited to our objectives. Custom software
solutions thus need to be developed in order to further improve our tests.

These are the original purposes of Crappy : providing a framework for
controlling tests and driving hardware in a synchronized and
supplier-independent software environment.

Key features of Crappy
----------------------

To this end, choices were made that are now keys to our framework:

- **open-source** :
  It is important for us that everyone can use our work, and bring its own code
  to the world.

- **modular** :
  We provide a software basis that can easily be extended to drive new hardware
  and perform custom operation on data.

- **simple** :
  `Python <https://www.python.org/>`_ has been chosen for its high level. We are
  not developers, and neither are our users, so we cannot afford to use a
  low-level programming language. We work with typical loop time of more than 1
  millisecond (10ms most of the time), and Python is enough for that. It is also
  pretty easy to add a small piece of C/C++ to the Python code if a speedup is
  needed.

- **performance** :
  A great deal of work is made to ensure the performance of the framework. Most
  tests require a good repeatability and stability, and may become hazardous in
  case of non-handled issue.

- **parallelization** :
  The key to a good test is the synchronisation between the different sensors.
  This is why we chose to massively parallelize our framework, ensuring
  every device can run simultaneously in a same time basis. This is also one of
  the major difficulties we have to deal with in Python.

Is Crappy for me ?
------------------

Although it was originally designed for driving mechanical tests, Crappy has
acquired the flexibility to adapt to other domains as well. Pretty much any
device communicating over USB, serial, SPI or I2C can be integrated within the
framework, making it suitable for many fields !

So Crappy is **the right solution** for you if :

- You want to drive sensors and actuators in a synchronized and parallelized
  way.

- You want a modular solution in which you can easily add new hardware,
  functions and write your own test protocols.

- You don't want to bother coding in a low-level language.

- You want to remain independent from commercial software environments.

As Crappy's scope is well-defined, there are also situations in which Crappy
won't be able to help you. So Crappy is **NOT for you** if :

- You need to acquire data or drive actuators at frequencies greater than 500Hz.

- Your devices cannot be driven without using proprietary software and such
  software cannot interface with other programs on the computer.

- You don't want to code in Python (you should really give it a try, it's
  great and easy to learn :) )

.. |Downloads| image:: https://static.pepy.tech/badge/crappy
   :target: https://static.pepy.tech/badge/crappy

.. |Documentation status| image:: https://readthedocs.org/projects/crappy/badge/?version=latest
   :target: https://crappy.readthedocs.io/en/latest/?badge=latest

.. |PyPi version| image:: https://badgen.net/pypi/v/crappy/
   :target: https://pypi.org/project/crappy