Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=======================

This package aims to provide easy-to-use tools for command and acquisition on
complex experimental setups.

Requirements
------------

To install Crappy you will need Python 3 (3.4 or higher)
with the following modules :
- Numpy
- Matplotlib
- openCV
- pyserial
- scikit-image

These modules are not mandatory but will provide additionnal functionalities:
- SimpleITK (to save images to disk)
- Ximea API (for ximea cameras)
- Labjack LJM (for labjack support)
- pycuda (for real-time correlation)
- Comedi driver (Linux Only, for comedi acquisition boards)
- PyDaqmx (Windows only, for NI boards)
- openDAQ (for opendaq board)

Installation
------------

Only tested on Ubuntu 16.04 / 18.04 and 20.04:

       git clone https://github.com/LaboratoireMecaniqueLille/crappy.git

       cd crappy

       sudo python setup install

Se documentation for more details


Documentation
-------------

The latest version for the branch master can be accessed at
https://laboratoiremecaniquelille.github.io/crappy/

To build it yourself, install doxygen, doxypy and doxygen-gui:

    sudo apt-get install doxygen doxypy doxygen-gui

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
