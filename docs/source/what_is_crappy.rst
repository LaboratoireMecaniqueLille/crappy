=================
What is Crappy ?
=================

|Downloads|
|Documentation status|
|PyPi version|
|PyPI pyversions|

Overview
--------

CRAPPY is an acronym and stands for **C**\ommand and **R**\eal-time
**A**\cquisition in **P**\arallelized **PY**\thon.

Its aim is to provide an easy-to-use software environment for **controlling**
**tests and driving hardware**. It targets an audience of **experimental**
**researchers and R&D engineers**, and provides a framework that manages the
**operation, the parallelization and the synchronization** of the test
equipment. Any device that can be controlled from Python can be integrated into
Crappy, regardless of its manufacturer. In addition to interfacing with
hardware, Crappy also comes with **a rich set of signal and image processing**
**solutions** that can be combined to build and **drive arbitrarily complex**
**experimental setups** !

Crappy is developed at the `LaMCube <https://lamcube.univ-lille.fr/>`_, a
mechanical research laboratory based in Lille, France. It was originally
intended for material mechanics, but **can be used in any domain** that
requires to run experimental tests.

Key features of Crappy
----------------------

- **open-source** :
  It is natural for us to make our work available to anyone, and to keep it
  open to outside contributions. All the code base is freely hosted on GitHub.

- **modular** :
  The software basis we provide can be easily extended and fine-tuned to drive
  new hardware or perform custom operations on data.

- **simple** :
  `Python <https://www.python.org/>`_ was chosen for Crappy because it is one
  of the most accessible languages. We are not professional developers, and
  neither are our users ! The module was also designed so that most of its
  complexity is hidden from users.

- **performance** :
  Under the hood, a complex code ensures that the computer running Crappy
  operates at full power to maximize the performance of the framework. We're
  well aware that experimental tests require a good repeatability and
  stability, and may become hazardous in case of non-handled issues.

- **parallelization** :
  One of the keys to a good test is the synchronisation between the different
  sensors. Therefore, we chose to massively parallelize our framework,
  ensuring that every device can operate simultaneously on a same time basis.
  This is truly one of Crappy's main strengths !

Is Crappy for me ?
------------------

Crappy is **the right solution** for you if :

- You want to drive sensors and actuators in a synchronized and parallelized
  way.

- You want a modular solution in which you can easily add new hardware,
  functions and write your own test protocols.

- You don't want to bother coding in a low-level language.

- You want to remain independent from commercial software environments.

As Crappy's scope is well-defined, there are also situations in which Crappy
won't be able to help you. So Crappy is **NOT for you** if :

- You need deterministic sampling at frequencies higher than a few dozen Hz
  (those who need it will know what that means !).

- Your devices cannot be driven from Python, e.g. if they can only be driven
  from a proprietary software.

- You don't want to code a single line in Python (you should really give it a
  try, it's great and easy to learn !).

.. |Downloads| image:: https://static.pepy.tech/badge/crappy
   :target: https://static.pepy.tech/badge/crappy

.. |Documentation status| image:: https://readthedocs.org/projects/crappy/badge/?version=latest
   :target: https://crappy.readthedocs.io/en/latest/?badge=latest

.. |PyPi version| image:: https://badgen.net/pypi/v/crappy/
   :target: https://pypi.org/project/crappy

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/crappy.svg
   :target: https://pypi.python.org/pypi/crappy/
