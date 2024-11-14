Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=================================================================

[![Downloads](https://static.pepy.tech/badge/crappy)](https://www.pepy.tech/projects/crappy)
[![Documentation Status](https://readthedocs.org/projects/crappy/badge/?version=latest)](https://crappy.readthedocs.io/en/latest/)
[![PyPi version](https://badgen.net/pypi/v/crappy/)](https://pypi.org/project/crappy/)
[![Python version](https://img.shields.io/pypi/pyversions/crappy.svg)](https://pypi.org/project/crappy/)

CRAPPY aims to provide a free and open-source software canvas for driving 
experimental setups in a versatile and accessible way.

Presentation
------------

Setups in experimental research tend to get increasingly complex, and require 
to drive a variety of actuators, sensors, and cameras from various suppliers. 
However, as researchers are one step ahead of industrials, the commercially 
available testing solutions are not always well-suited to their objectives. 
Developing a custom software interface is also not always an option, as the 
synchronization of the devices and the optimization of the computer resources
can prove challenging even to experienced developers.

The purpose of CRAPPY is to provide a framework for driving experimental 
setups, in which even the most complex designs can be controlled in usually 
less than a hundred lines of code. CRAPPY is:

- A free and open source Python module
- Written in pure Python, to make it easily understandable and editable by a 
large audience
- Highly modular and versatile, and can adapt to almost any setup
- Distributed with a wide collection of ready-to-run [examples](https://github.com/LaboratoireMecaniqueLille/crappy/examples)
- Heavily optimized, to make the most of your computer's resources
- Distributed with a collection of powerful tools for performing real-time data
and image processing

Crappy is developed at the [LaMCube](https://lamcube.univ-lille.fr/), a
mechanical research laboratory based in Lille, France, where it is used mainly 
for materials testing.

Requirements
------------

CRAPPY can run with Python 3.9 to 3.13, and has been tested on Windows, Linux, 
Raspberry Pi and macOS. It can probably run on other operating systems 
supporting the required Python versions. 

CRAPPY has only one requirement: [Numpy](https://numpy.org/) (1.21 or higher).
In addition, other modules can be necessary depending on which features you 
want to use. The main ones are [Matplotlib](https://matplotlib.org/),
[openCV](https://opencv.org/), [pyserial](https://pypi.org/project/pyserial/)
and [Pillow](https://python-pillow.org/).

Installation
------------

CRAPPY is distributed on PyPI, and can be installed on the supported operating 
systems simply by running the following command in a terminal:

    python -m pip install crappy

You'll find more details in the dedicated [installation section](https://crappy.readthedocs.io/en/latest/installation.html) 
of the documentation, as well as alternative installation methods.

Citing Crappy
-------------

If Crappy has been of help in your research, please reference it in your 
academic publications by citing one or both of the following articles:

- Couty V., Witz J-F., Martel C. et al., *Command and Real-Time Acquisition in 
Parallelized Python, a Python module for experimental setups*, SoftwareX 16, 
2021, DOI: 10.1016/j.softx.2021.100848. 
([link to Couty et al.](https://www.sciencedirect.com/science/article/pii/S2352711021001278))
- Weisrock A., Couty V., Witz J-F. et al., *CRAPPY goes embedded: Including 
low-cost hardware in experimental setups*, SoftwareX 22, 2023, DOI: 
10.1016/j.softx.2023.101348. 
([link to Weisrock et al.](https://www.sciencedirect.com/science/article/pii/S2352711023000444))

Documentation
-------------

The latest versions of the documentation can be accessed on our
[ReadTheDocs](https://crappy.readthedocs.io/) page. It contains a description 
of Crappy's features, tutorials, and other useful information.

License
-------

[GNU GPLv2](https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/LICENSE) 
&copy; 2015, Laboratoire Mécanique de Lille & contributors
