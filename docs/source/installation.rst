============
Installation
============

Requirements
-------------
To install Crappy, you will need Python 3 (3.4 or higher) with the following modules :

	- Numpy (2.7 or higher)
	- matplotlib (1.5.3 or higher)
	- Opencv (3.0 or higher)
	- Scipy (0.17 or higher)
	- pyserial
	- Scikit-image (0.11 or higher)

These modules are not mandatory but will provide additional functionalities :

	- Simple-ITK (to save images to disk)
	- xiApi (for Ximea cameras)
	- labjack (for Labjack support)
	- PyCUDA (for real-time correlation)
	- Comedi driver (for Comedi acquisition boards, Linux only)
	- PyDaqmx (for National Instrument boards, Windows only)
	- niFgen (package from National Instrument, Windows only)
	- openDAQ (for opendaq boards)

.. note::	- *"If you have Python 2 also installed, you should remplace* ``python`` *and* ``pip`` *by* ``python3`` *and* ``pip3`` *in all the following steps."*
		- *"Replace* ``module-name`` *by the name of the module you want to install."*
		- *"See* :ref:`Documentation` *for more details."*

A. For Linux users
-------------------
These steps have been tested for Ubuntu 14.04, 15.10 and 16.04,  but should work with other distros as well.

1. First, you should install all the required python modules using pip. ::

	sudo apt-get update
	sudo apt-get install python-pip
	sudo pip install module-name

2. Then you may need the dev packages for python and nump and also python-imaging-tk: ::

	sudo apt-get install python-dev python-imaging-tk

3. You can now install crappy. Get the sources using git and use setup script: ::

	git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
	cd crappy
	sudo python setup.py install

B. For Windows users
---------------------
These steps have been tested for Windows 8.1 but should work with other versions as well. Make sure you are using the x64 version of python or the C++ modules will not compile properly.

1. Install the dependencies: ::

	pip install module-name

   This will works for most modules, but some may fail and need a wheel file built for windows. We had to do this for numpy (with mkl) and scikit-image. Just find the correct version at http://www.lfd.uci.edu/~gohlke/pythonlibs/ and simply run: ::

	pip install wheel_file.whl

2. Also, you will need Visual C++ for Python 3.x (your version of python) in order to compile C++ modules.
If you want to use Ximea cameras, don't forget to install XiAPI and add ``c:\XIMEA\API\x64`` to your path.

3. Then you can get the source code and install it: ::

	git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
	cd crappy
	setup.py install

C. For macOS users
-------------------
These steps have been tested on macOS Catalina (10.15.7), but should work with other versions as well.

1. You should install the required modules using pip. ::

	python install pip
	pip install module-name
