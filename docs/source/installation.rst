============
Installation
============

Install a published release
---------------------------

Crappy requires Python ``>=3.10``. Its only base package dependency is:

- ``numpy>=2.0.0``

.. note::
   Individual hardware drivers, image-processing features, or Blocks can have
   additional requirements. They are imported only when the corresponding
   feature starts, so a basic installation does not need packages for unused
   hardware. Check the :doc:`hardware matrix <hardware>` for the dependencies
   of the included hardware drivers.

.. note::
   The Python package is designed to run on Linux, Windows, macOS, and
   RaspberryPi OS. Compatibility with particular hardware can still depend on
   an operating-system driver or vendor library. The hardware matrix records
   these constraints separately from package installation.

1. Check Python
~~~~~~~~~~~~~~~

Run this command in a terminal:

.. code-block:: shell-session

   python --version

If the reported version does not satisfy ``>=3.10``, install a newer Python
before continuing. On Windows, Python may need to be installed first. On Linux
and macOS, install another Python version alongside the system Python instead
of removing the system interpreter.

2. Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A virtual environment keeps Crappy and its dependencies separate from other
Python projects. Creating one is recommended but not required.

.. code-block:: shell-session

   python -m venv venv_crappy

The following commands use the environment's interpreter directly, so no
activation step is required.

3. Install Crappy from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux and macOS, run:

.. code-block:: shell-session

   venv_crappy/bin/python -m pip install --upgrade pip
   venv_crappy/bin/python -m pip install crappy

On Windows, run:

.. code-block:: shell-session

   venv_crappy\Scripts\python.exe -m pip install --upgrade pip
   venv_crappy\Scripts\python.exe -m pip install crappy

Without a virtual environment, replace the interpreter path with ``python``:

.. code-block:: shell-session

   python -m pip install --upgrade pip
   python -m pip install crappy

Install optional packages only for the tasks that need them. Common examples
include:

- ``matplotlib`` for live plots and Matplotlib image display
- ``opencv-python`` and ``Pillow`` for many image-acquisition and display tasks
- ``scikit-image`` for video extensometry
- ``SimpleITK`` for an additional image-reading and writing backend
- ``tables`` for HDF5 streaming-data recording
- ``pyserial`` for serial devices
- ``pyusb`` for drivers that communicate directly over USB
- ``PyCUDA`` and a compatible CUDA installation for GPU image processing

For example, install Matplotlib with the same interpreter used for Crappy:

.. code-block:: shell-session

   venv_crappy/bin/python -m pip install matplotlib

Use the plain ``python`` interpreter instead when appropriate. Driver-specific
dependencies and backend choices are listed in :doc:`hardware` and in each
driver's :doc:`API entry <api>`.

4. Check the installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Import Crappy and print the installed version:

.. code-block:: shell-session

   venv_crappy/bin/python -c "import crappy; print(crappy.__version__)"

On Windows, use ``venv_crappy\Scripts\python.exe``. Without a virtual
environment, use ``python``. A successful check prints the version without a
traceback.

If the import fails, copy the complete error message and continue with
:doc:`troubleshooting`.

5. Run a hardware-free test
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download the quickstart script
</downloads/getting_started/quickstart.py>` and save it as ``quickstart.py``.
From the directory containing the file, run:

.. code-block:: shell-session

   venv_crappy/bin/python quickstart.py

The script requires no physical hardware, graphical interface, or optional
package. It prints simulated measurements and stops automatically after three
seconds. A final ``Generator Path exhausted`` warning marks its planned end.

The :doc:`quickstart tutorial <tutorials/quickstart>` explains the script.
Continue with :doc:`tutorials` for guided tasks or :doc:`examples` for the
complete examples index.

Development installation
------------------------

The commands above install a published release for normal use. A source
checkout is intended for contributing code or documentation and uses a
different setup. Follow :ref:`the development setup
<developers:building the documentation>` when working on the repository.
