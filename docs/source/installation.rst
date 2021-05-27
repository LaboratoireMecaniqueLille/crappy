============
Installation
============

Requirements
------------
To install Crappy, you will need Python 3 (3.6 or higher) with the following
modules :

	- Numpy (2.7 or higher)

These modules are not mandatory but will provide additional functionalities :

        - matplotlib (1.5.3 or higher, for plotting graphs and displaying images)
        - Opencv (3.0 or higher, to perform image acquisition and analysis)
	- pyserial (To interface with serial sensors and actuators)
	- Tk (For the configuration interface of cameras)
	- Scikit-image (0.11 or higher)
	- xiApi (for Ximea cameras)
	- labjack (for Labjack support)
	- Simple-ITK (for faster image saving)
	- PyCUDA (for real-time correlation)
	- Comedi driver (for Comedi acquisition boards, Linux only)
	- PyDaqmx (for National Instrument boards, Windows only)
	- niFgen (package from National Instrument, Windows only)
	- openDAQ (for opendaq boards)

.. note::	- *"If you have Python 2 also installed, you should remplace* ``python`` *and* ``pip`` *by* ``python3`` *and* ``pip3`` *in all the following steps."*
		- *"Replace* ``module-name`` *by the name of the module you want to install."*
		- *"See* :ref:`Documentation` *for more details."*

A. For Linux users
------------------
These steps have been tested for Ubuntu 16.04, 18.04 and 20.04  but should work
with other distros as well.

1a. Install the dependencies in a virtualenv (recommended) ::

  workon myenv
  pip install <module>

1b. OR Installing the required Python modules on the system ::

	sudo apt update
	sudo apt install python-pip
	sudo pip install <module>


2. You can now install crappy. Get the sources using git and use setup script: ::

    git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
    cd crappy
    python setup.py install
    sudo python setup.py install

B. For Windows users
--------------------
These steps have been tested for Windows 8.1 but should work with other
versions as well. Make sure you are using the x64 version of python or the C++
modules will not compile properly.

1. Install the dependencies: ::

	pip install module-name

   This will works for most modules, but some may fail and need a wheel file
   built for windows. We had to do this for numpy (with mkl) and scikit-image.
   Just find the correct version at http://www.lfd.uci.edu/~gohlke/pythonlibs/
   and simply run: ::

	pip install wheel_file.whl

2. Also, you will need Visual C++ for Python 3.x (your version of python) in
   order to compile C++ modules.  If you want to use Ximea cameras, don't
   forget to install XiAPI and add ``c:\XIMEA\API\x64`` to your path.

3. Then you can get the source code and install it: ::

	git clone https://github.com/LaboratoireMecaniqueLille/crappy.git
	cd crappy
	setup.py install

C. For macOS users
------------------
These steps have been tested on macOS Catalina (10.15.7), but should work with
other versions as well.

1. You should install the required modules using pip. ::

	python install pip
	pip install module-name

D. Troubleshooting
------------------

The imaging module is not natively included in Tk. Some user may have to
install it manually to us the camera configuration GUI

For Ubuntu, you can do ::

  sudo apt install python3-pil.imagetk

Also, the matplotlib backend may have some troubles initializing multiple
windows on some desktop environnement. It can be fixed easily by using an other
backend. Simply specify a functionnal backed in the grapher to fix this issue
i.e.: ::

  graph = crappy.bocks.Grapher(('t(s)','F(N)'),backend='TkAgg')

Or simply edit the default backend in crappy/blocks/grapher.py by replacing
None with the desired backend.
