Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=================================================================

[![Downloads](https://pepy.tech/badge/crappy)](https://pepy.tech/project/crappy)
[![Documentation Status](https://readthedocs.org/projects/crappy/badge/?version=latest)](https://crappy.readthedocs.io/en/latest/?badge=latest)
[![PyPi version](https://badgen.net/pypi/v/crappy/)](https://pypi.org/project/crappy)

This package aims to provide an open-source software canvas for developing 
experimental tests in a versatile and accessible way.

Presentation
------------

Crappy is developed at the [LaMCube](https://lamcube.univ-lille.fr/), a
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

Requirements
------------

To install Crappy you will need Python 3 (3.6 or higher) with the following 
modules :
- [Numpy](https://numpy.org/) (1.19.0 or higher)

In addition, other modules are necessary for a wide range of applications in Crappy 
without being mandatory for installing the module. The main ones are [Matplotlib](https://matplotlib.org/),
[openCV](https://opencv.org/), [pyserial](https://pypi.org/project/pyserial/)
and [Tk](https://docs.python.org/3/library/tkinter.html).

Installation
------------

Tested on Windows 10, Ubuntu 18.04 and 20.04, and MacOS Sierra.
Simply run in a terminal (with Python installed) :

    pip install crappy

or

    pip3 install crappy

Refer to the dedicated [installation section](https://crappy.readthedocs.io/en/latest/installation.html) 
of the documentation for more details.

Documentation
-------------

The latest versions of the documentation can be accessed on our
[ReadTheDocs](https://crappy.readthedocs.io/) page. It contains descriptions of
Crappy's features, tutorials, and other useful information.

Bug reports
-----------

Please report bugs, issues, ask for help or give feedback in the [dedicated github section](https://github.com/LaboratoireMecaniqueLille/crappy/issues).

License information
-------------------

Refer to the file [``LICENSE.txt``](https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/LICENSE) 
for information on the history of this software, terms & conditions for usage, 
and a disclaimer of all warranties.
