=================
What is Crappy?
=================

|Downloads|
|Documentation status|
|PyPi version|
|PyPI pyversions|

Overview
--------

CRAPPY is an acronym and stands for **C**\ommand and **R**\eal-time
**A**\cquisition in **P**\arallelized **PY**\thon.

Crappy provides a software environment for controlling tests and driving
hardware. It is intended for experimental researchers and research and
development (R&D) engineers. The framework manages the operation,
parallelization, and synchronization of test equipment. A device can be
integrated in Crappy if it can be controlled with Python, regardless of its
manufacturer. Crappy also provides ready-to-use signal-processing and
image-processing features that can be combined for complex experimental setups.

Crappy is developed at the `LaMCube <https://lamcube.univ-lille.fr/>`_, a
mechanical research laboratory based in Lille, France. It was originally
intended for material mechanics, but it can be used in any domain that runs
experimental tests.

Key features of Crappy
----------------------

- **Open source:**
  The source code and contribution workflow are hosted on GitHub.

- **Modular:**
  Components can be extended to drive new hardware or perform custom data
  operations.

- **Python-based:**
  Tests are written as regular `Python <https://www.python.org/>`_ scripts
  using ready-to-use components for acquisition, control, and data handling.

- **Performance-oriented:**
  Crappy is designed to make effective use of the computer so that all the
  tasks can run smoothly together, including advanced real-time processing.

- **Built for complete experiments:**
  A single script can coordinate hardware, process measurements, display
  results, and save data throughout a test.

When to use Crappy
------------------

Consider Crappy when:

- You want to drive sensors and actuators in an simple and efficient way.

- You want to add your own hardware integrations, processing functions, or test
  protocols to a modular framework.

- You want to define a test in Python rather than a low-level or specialized
  language.

- You want to remain independent from commercial software environments.

Choose a different tool when:

- You need deterministic sampling or hard real-time guarantees. Crappy does
  not provide them.

- Your devices cannot be driven from Python, e.g. if they can only be driven
  from a proprietary software.

- You need a graphical application for configuring and running tests without
  writing code.

.. |Downloads| image:: https://static.pepy.tech/badge/crappy
   :target: https://static.pepy.tech/badge/crappy

.. |Documentation status| image:: https://readthedocs.org/projects/crappy/badge/?version=latest
   :target: https://crappy.readthedocs.io/en/latest/?badge=latest

.. |PyPi version| image:: https://badgen.net/pypi/v/crappy/
   :target: https://pypi.org/project/crappy/

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/crappy.svg
   :target: https://pypi.org/project/crappy/
