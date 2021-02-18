=================
What is Crappy ?
=================

CRAPPY is an acronym and stands for Command and Real-time Acquisition in
Parallelized PYthon.

Crappy is developped in the "Laboratoire de Mécanique de Lille", a
mechanical research laboratory based in Lille, France to provide a powerful
and easy-to-use framework for material testing.

In order to understand the mechanical behaviour of materials, we tend
to setup tests with more sensors, more precision and more complexity.
As we are one step ahead of industrials, the testing machines we can buy may
not be adapted to our objectives and we have to develop our own softwares to
improve our tests.

This is the original reason why we created Crappy : provide a framework to
control our tests and all of our hardware.

To this end, we made some choice that are now the keys of the framework:

- **open-source** : it is important for us that everyone can use our work, and bring its own code to the world.

- **modular** : the hardware has to be, as much as possible, separated from the software, to provide re-usable code for different setup.

- **simple** : Python as been chosen for its high level. We are not developpers, and neither are our users , so we cannot afford to use a low level programming language. We work with typical loop time of more than 1 millisecond (10ms most of the time), and Python is enough for this. It is also pretty easy to put a small piece of C/C++ in the Python if we need a speedup.

- **performance** : a great deal of work is made to ensure the performance of the framework. Most tests requires a good repetablilty and a stability, and may become hazardous in case of not-handled issue.

- **parallelization** : the key to a good test is the synchronisation between the different sensors. This is why we chose to massively parallelize our framework, to ensure everything can run at the same time. This is also one of the major difficulties we have to deal with in Python.