Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=================================================================

This package aims to provide an open-source software canvas for developing 
experimental tests in a versatile and accessible way.

Requirements
------------

To install Crappy you will need Python 3 (3.6 or higher) with the following 
modules :
- [Numpy](https://numpy.org/) (1.19.0 or higher)

These modules are not mandatory but will provide additional functionalities:
- [Matplotlib](https://matplotlib.org/) (1.5.3 or higher, for plotting graphs 
  and displaying images)
- [openCV](https://opencv.org/) (3.0 or higher, to perform image acquisition and 
  analysis)
- [pyserial](https://pypi.org/project/pyserial/) (to interface with serial 
  sensors and actuators)
- [Tk](https://docs.python.org/3/library/tkinter.html) (For the configuration
  interface of cameras)
- [scikit-image](https://scikit-image.org/) (0.11 or higher)
- [Ximea API](https://www.ximea.com/support/wiki/apis/xiapi) (for ximea cameras)
- [Labjack LJM](https://labjack.com/support/software/examples/ljm/python) (for 
  labjack support)
- [Simple-ITK](https://simpleitk.org/) (for faster image saving)
- [pycuda](https://documen.tician.de/pycuda/) (for real-time correlation)
- [Comedi](https://www.comedi.org/) driver (Linux Only, for comedi acquisition 
  boards)
- [PyDaqmx](https://pythonhosted.org/PyDAQmx/) (Windows only, for NI boards)
- [openDAQ](https://pypi.org/project/opendaq/) (for opendaq board)
- [niFgen](https://www.ni.com/fr-fr/support/downloads/drivers/download.ni-fgen.html#346233) 
  (package from National Instrument, Windows only)

Installation
------------

Tested on Windows 10, Ubuntu 18.04 and 20.04, and MacOS Sierra :

    pip install crappy

or

    pip3 install crappy

See [documentation](https://crappy.readthedocs.io/en/latest/installation.html) 
for more details.

Documentation
-------------

The latest version of the documentation can be accessed 
[here](https://crappy.readthedocs.io/).

Bug reports
-----------

Please report bugs in the [dedicated github section](https://github.com/LaboratoireMecaniqueLille/crappy/issues).

License information
-------------------

Refer to the file [``LICENSE.txt``](https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/LICENSE) 
for information on the history of this software, terms & conditions for usage, 
and a DISCLAIMER OF ALL WARRANTIES.
