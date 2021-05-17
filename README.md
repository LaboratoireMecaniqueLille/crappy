Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=======================

This package aims to provide easy-to-use tools for command and acquisition on
complex experimental setups.

Requirements
------------

To install Crappy you will need Python 3 (3.6 or higher)
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

Only tested on Ubuntu 18.04 and 20.04:

    git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
    cd crappy
    sudo python3 setup.py install

See documentation for more details


Documentation
-------------

The latest version of the documentation can be accessed at
https://crappy.readthedocs.io/

To build it locally, please install Sphinx and sphinx\_rtd\_theme

    sudo apt install python3-sphinx
    pip3 install sphinx_rtd_theme

and use the Makefile in docs/ to build the html pages

    cd docs/
    make html


Bug reports
-----------

Please reports bugs at:

https://github.com/LaboratoireMecaniqueLille/crappy/issues


License information
-------------------

See the file ``LICENSE.txt`` for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
