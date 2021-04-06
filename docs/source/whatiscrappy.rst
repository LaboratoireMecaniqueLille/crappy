=================
What is Crappy ?
=================

CRAPPY is an acronym and stands for Command and Real-time Acquisition in
Parallelized PYthon.

Crappy is developped at the "Laboratoire de Mécanique de Lille", a
mechanical research laboratory based in Lille, France to provide a powerful
and easy-to-use framework for materials testing.

In order to understand the mechanical behaviour of materials, we tend
to setup tests with more and more sensors, leading to an increasing complexity 
and requiring a higher precision.
As we are one step ahead of industrials the commercially available testing machines may
not be adapted to our objectives, we thus have to develop our own softwares to 
further improve our tests.

This is the original reason why we created Crappy : to provide a framework for
controlling our tests and all of our hardware.

To this end, we made some choice that are now the keys of the framework:

- **open-source** : it is important for us that everyone can use our work, and bring its own code to the world.

- **modular** : the hardware has to be, as much as possible, separated from the software, to provide re-usable code for different setups.

- **simple** : Python has been chosen for its high level. We are not developpers, and neither are our users, so we cannot afford to use a low level programming language. We work with typical loop time of more than 1 millisecond (10ms most of the time), and Python is enough for that. It is also pretty easy to put a small piece of C/C++ in the Python if we need a speedup.

- **performance** : a great deal of work is made to ensure the performance of the framework. Most tests require a good repetablilty and a stability, and may become hazardous in case of non-handled issue.

- **parallelization** : the key to a good test is the synchronisation between the different sensors. This is why we chose to massively parallelize our framework, ensuring everything can run simultaneously. This is also one of the major difficulties we have to deal with in Python.
