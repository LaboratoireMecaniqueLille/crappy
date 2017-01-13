=======================
Command and Real-time Acquisition in Parallelized PYthon (CRAPPY){#mainpage}
=======================

This package aims to provide easy-to-use tools for command and acquisition on 
complex experimental setups.

.. contents::

Description
-----------

See ``DESCRIPTION.rst`` for a more complete description.

Requirements
------------

To install Crappy you will need:

- Python 2 (2.6 or higher)
- Numpy
- Scipy
- Matplotlib
- openCV
- Pandas
- SimpleITK
- scikit-image
- pyserial

These packages are not mandatory but will provide additionnal functions:
- Ximea API (for ximea cameras)
- Labjack LJM (for labjack support)
- pycuda (for real-time correlation)
- openDAQ (for opendaq board)

Installation
------------

Only tested on Ubuntu 14.04 / 15.10 / 16.04:

       git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
       
       cd crappy

       sudo python setup install


Documentation
-------------

The documentation is generated with doxywizard:

Please install doxygen and pluggins as follow:

sudo apt-get install doxygen doxygen-gui
sudo apt-get instal doxypy

Then, download and install the doxypypy project.

then, load the Doxyfile located in doc with doxywizard:

    doxywizard doc/Doxyfile

Finally, run doxygen from the run tab, and show html output.


Bug reports
-----------

Please reports bugs at:

https://github.com/LaboratoireMecaniqueLille/crappy/issues


License information
-------------------

See the file ``LICENSE.txt`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
