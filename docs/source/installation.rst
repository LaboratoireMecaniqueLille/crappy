============
Installation
============

.. role:: shell(code)
  :language: shell-session
  :class: highlight

Requirements
------------

Crappy has been installed and tested on Linux, Windows, macOS, and Raspberry Pi
computers. Other systems that support a compatible Python version may work but
are not covered by project testing.

.. note::
  We develop Crappy on recent OS versions, and no particular effort is made to
  ensure compatibility with older OS versions.

Crappy requires Python 3.10 or later and the following package:

- `numpy <https://numpy.org/>`_ (2.0.0 or higher)

The following optional packages enable additional features:

- `matplotlib <https://matplotlib.org/>`_ (3.3.0 or higher, for plotting graphs
  and displaying images)
- `opencv <https://opencv.org/>`_ (4.0 or higher, for image acquisition
  and processing)
- `pyserial <https://pypi.org/project/pyserial/>`_ (3.4 or higher, to interface with serial
  sensors and actuators)
- `Tk <https://docs.python.org/3/library/tkinter.html>`_ (for the configuration
  interface of cameras)
- `scikit-image <https://scikit-image.org/>`_ (0.18.0 or higher)
- `SimpleITK <https://simpleitk.org/>`_ (2.0.0 or higher, for image recording)
- `PyCUDA <https://documen.tician.de/pycuda/>`_ (for GPU accelerated features)

.. note::
  Optional dependencies are imported only by the features that need them.
  Crappy reports a missing dependency when the corresponding feature starts.

1. Check your Python version
----------------------------

Before installing Crappy, check that you have a compatible version of Python.
You can get the current version by running :shell:`python --version` in a
console. The version should then be displayed, e.g. :shell:`Python 3.10.1`.

.. note::
  Windows does not include Python by default. If it is unavailable, the command
  displays an error message.

If the current version of Python is not compatible with Crappy (requires Python
>=3.10), or if Python is not installed, first install a compatible version of
Python. The precise installation steps for each OS are beyond the scope of this
documentation.

.. note::
  On Linux and macOS, install a new Python version alongside the system Python.
  Removing the system Python can damage operating-system tools.

2. Deploy a virtual environment (optional)
------------------------------------------

Install Crappy in a `virtual environment
<https://docs.python.org/3/library/venv.html>`_ to avoid conflicts with Python
packages installed at the user or system level. A user-level installation is
also supported.

To create a virtual environment called ``venv_crappy``, run the following
command at the location of your choice.

.. code-block:: shell-session

   python -m venv venv_crappy

This should create a new folder called `venv_crappy` at the location of your
console, containing an independent install of Python.

3. Install Crappy
-----------------

After installing a compatible Python version and optionally creating a virtual
environment, install Crappy with ``pip``.

**Without a virtual environment**

.. code-block:: shell-session

   python -m pip install crappy

**In a virtual environment**

On Linux and macOS, assuming your console is at the location of the virtual
environment:

.. code-block:: shell-session

   venv_crappy/bin/python -m pip install crappy

On Windows, assuming your console is at the location of the virtual
environment:

.. code-block:: shell-session

   venv_crappy\Scripts\python.exe -m pip install crappy

Use the same interpreter to install any optional package required by your
script. For example:

.. code-block:: shell-session

   python -m pip install matplotlib

4. Check your install
---------------------

After installing Crappy, import it and print its version:

**Without a virtual environment**

.. code-block:: shell-session

   python -c "import crappy;print(crappy.__version__)"

**In a virtual environment**

On Linux and macOS, assuming your console is at the location of the virtual
environment:

.. code-block:: shell-session

   venv_crappy/bin/python -c "import crappy;print(crappy.__version__)"

On Windows, assuming your console is at the location of the virtual
environment:

.. code-block:: shell-session

   venv_crappy\Scripts\python.exe -c "import crappy;print(crappy.__version__)"

This command should return without an error and print the installed version of
Crappy. If that is not the case, please refer to the
:ref:`Troubleshooting <troubleshooting:troubleshooting>` page of the
documentation.

If you can successfully import Crappy, you can then try to run a few examples
to confirm that Crappy operates as expected. The `examples folder
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/master/examples>`_ of
the GitHub repository contains a wide collection of readily-runnable examples.
To execute a test script called :file:`example.py`, run the following lines in
a console:

**Without a virtual environment**

.. code-block:: shell-session

   python example.py

**In a virtual environment**

On Linux and macOS, assuming your console is at the location of the virtual
environment and that :file:`example.py` is at the same level as the virtual
environment:

.. code-block:: shell-session

   venv_crappy/bin/python example.py

On Windows, assuming your console is at the location of the virtual environment
and that :file:`example.py` is at the same level as the virtual environment:

.. code-block:: shell-session

   venv_crappy\Scripts\python.exe example.py

The installation is ready when the import check and a suitable example both
complete without errors.
