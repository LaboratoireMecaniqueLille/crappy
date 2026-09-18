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
development (R&D) engineers. A single test can acquire measurements, send
commands, process data, display results, and save files. Crappy provides
ready-to-use components for these tasks, and users can add custom components
for their own equipment or procedures.

A device can be integrated with Crappy if it can be controlled from Python,
regardless of its manufacturer. Ready-to-use signal-processing and
image-processing features can be combined with hardware control when an
experiment requires them.

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
  Crappy aims to use the computer efficiently when acquisition, commands,
  display, processing, and recording all run during the same test.

- **Built for complete experiments:**
  A single script can coordinate hardware, process measurements, display
  results, and save data throughout a test.

When to use Crappy
------------------

Consider Crappy when:

- You want to acquire measurements from sensors and drive actuators from one
  test script.

- You want to add your own hardware integrations, processing functions, or test
  protocols to a modular framework.

- You want to define a test in Python rather than a low-level or specialized
  language.

- You want to remain independent from commercial software environments.

Choose a different tool when:

- You need deterministic sampling or hard real-time guarantees. Crappy does
  not provide them.

- Your devices cannot be driven from Python, e.g. if they can only be driven
  by proprietary software.

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
