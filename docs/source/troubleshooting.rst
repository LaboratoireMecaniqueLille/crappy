===============
Troubleshooting
===============

In case of an error, always consider the complete exception traceback and focus
on the first error raised by a Block. Later errors can be consequences of the
same failure while the other Blocks stop and clean up.

Crappy cannot be imported
-------------------------

``ModuleNotFoundError: No module named 'crappy'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python interpreter running the script cannot see the Crappy installation.
Check which interpreter is running and whether Crappy is installed for it:

.. code-block:: shell-session

   python -c "import sys; print(sys.executable)"
   python -m pip show crappy

If ``pip show`` reports that the package is absent, install it with that same
interpreter:

.. code-block:: shell-session

   python -m pip install crappy

When using a virtual environment, replace ``python`` with its full interpreter
path as shown in :doc:`installation`. Do not use a bare ``pip`` command from a
different environment.

``ImportError: cannot import name ... from 'crappy'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First check the imported file and version:

.. code-block:: shell-session

   python -c "import crappy; print(crappy.__file__); print(crappy.__version__)"

An unexpected path or an older version usually means that another environment
or installation is being used. A script named ``crappy.py`` or a directory
named ``crappy`` beside the script can also shadow the installed package.
Rename the local file or directory, then run the check again.

An optional dependency is missing
---------------------------------

``RuntimeError: Missing module: ...``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crappy uses placeholders for optional dependencies, so importing ``crappy``
can succeed even when a package needed by one feature is absent. The error is
raised when that feature starts. Install the missing dependency with the same
interpreter that runs the script:

.. code-block:: shell-session

   python -m pip install PACKAGE_NAME

The import name in a traceback and the package name accepted by ``pip`` can be
different. For example, ``cv2`` is provided by ``opencv-python`` and ``serial``
is provided by ``pyserial``. Check the affected object's :doc:`API entry
<api>` or, for distributed hardware drivers, the
:doc:`hardware matrix <hardware>`.

``ModuleNotFoundError: No module named ...`` from a backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some features select one of several backends. Install the package for the
selected backend, or choose another backend documented by that object. Common
examples include OpenCV or Matplotlib for image display and SimpleITK, Pillow,
OpenCV, or NumPy for image files.

If the Python package is installed but importing it still fails, read the
complete traceback. Hardware and GUI packages can also depend on a system
driver or shared library that ``pip`` does not install.

The script starts again or fails during startup
-----------------------------------------------

``An attempt has been made to start a new process before the current process has finished its bootstrapping phase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put Block creation, Link creation, and ``crappy.start()`` inside a function,
then call that function behind Python's main guard:

.. code-block:: python

   import crappy


   def main() -> None:
     # Create Blocks and Links here.
     crappy.start()


   if __name__ == '__main__':
     main()

This structure is required when Python starts workers with the ``spawn``
method, which is the default on Windows and can also be selected on other
platforms. Without the guard, each worker imports the script and executes its
top-level startup code again.

If the script is being packaged as a frozen executable, follow Python's
``multiprocessing.freeze_support`` guidance in addition to keeping the main
guard. For initial diagnosis, run the saved ``.py`` file from a terminal
instead of an interactive shell or notebook.

A graphical window does not open
--------------------------------

``ModuleNotFoundError: No module named 'tkinter'`` or ``'_tkinter'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tk is supplied by the Python installation or operating system, not by Crappy.
Do not try to install a package named ``tkinter`` from PyPI. Check whether the
current interpreter can create a Tk window:

.. code-block:: shell-session

   python -c "import tkinter; root = tkinter.Tk(); root.destroy()"

If the import fails, install Tk support using the instructions for the Python
distribution or operating system that supplied the interpreter. Make sure the
test command uses the same interpreter as the Crappy script.

``TclError: no display name and no $DISPLAY environment variable``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program is running without access to a graphical display, which is common
in a remote shell, container, service, or continuous-integration environment.
Setting an arbitrary ``DISPLAY`` value does not create a display server.
Either provide a working graphical session or run the test without graphical
Blocks and configuration windows.

``ImportError: Cannot load backend 'TkAgg'`` or a display backend crashes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib, OpenCV, and their graphical backends have separate runtime
requirements. First test the selected library outside Crappy using the same
interpreter. If more than one backend is supported by the Block, select one
that is installed and usable on the current system. The relevant backend
argument and accepted values are documented in the Block's API entry.

Run a test without a graphical interface
----------------------------------------

For headless execution:

- Replace Grapher, Dashboard, Canvas, Button, and StopButton with non-graphical
  Blocks where needed. LinkReader and Recorder are common alternatives for
  inspecting or saving regular data.
- Omit ImageDisplayer from VisionBlock pipelines. ImageRecorder can save
  images without opening a display window.
- Set ``config=False`` on CameraSource or an all-in-one Camera Block and
  provide ``img_shape`` and ``img_dtype`` explicitly. Image processors that
  normally request an interactive region or spots also need those values
  supplied directly. Check each processor's API before disabling
  configuration.
- Do not add a GUI merely to stop the test. A finite Generator path or a
  StopBlock can stop it from data instead.

The :doc:`quickstart tutorial <tutorials/quickstart>` is a useful baseline
because it requires no graphical interface or optional dependency.

Hardware cannot be opened
-------------------------

``SerialException``, ``Permission denied``, or ``Access is denied``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First list the ports visible to the interpreter that runs Crappy:

.. code-block:: shell-session

   python -m serial.tools.list_ports -v

Check that the configured port matches the device, then close serial monitors
and any earlier test that may still own it. Reconnect the device and run a
small test that opens only that port. If the port is visible but access is
denied, follow the operating-system or device-vendor instructions for granting
your user access. Do not work around the error by running the complete
experiment as an administrator or by making every serial device writable.
The `pySerial documentation
<https://pyserial.readthedocs.io/en/stable/appendix.html>`_ describes the
platform-specific permission issue.

``USBError: Access denied`` or ``No backend available``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These messages identify different failures:

- ``Access denied`` usually means that the operating system found the device
  but did not grant the current user access. On Linux, install the udev rule
  supplied by the hardware vendor when one is available.
- ``No backend available`` means that PyUSB cannot load a supported native USB
  library. Installing the Python package alone may not install that library or
  the required Windows driver.
- A result such as ``device not found`` can instead indicate a wrong USB
  identifier, a disconnected device, or a device already claimed by another
  application.

Use the installation instructions for the selected Crappy driver and the
`PyUSB project documentation <https://github.com/pyusb/pyusb>`_. Avoid copying
udev rules or replacing Windows USB drivers from an unrelated device model.

A camera is not detected or cannot deliver frames
-------------------------------------------------

Close every application that may be using the camera, verify its device index
or path, and test the selected camera backend outside Crappy. A camera visible
to one application may still be unavailable through another backend. Also
check that the requested pixel format, image size, and frame rate are
supported by that camera.

On Linux, cameras exposed through Video4Linux can be inspected with:

.. code-block:: shell-session

   v4l2-ctl --list-devices
   v4l2-ctl -d /dev/video0 --all
   v4l2-ctl -d /dev/video0 --list-formats-ext

Replace ``/dev/video0`` with the reported device. If ``v4l2-ctl`` is missing,
install the ``v4l-utils`` package using the instructions for the operating
system. A permission error on the device path must be corrected using the
operating-system or camera-vendor guidance, not by running the whole
experiment with elevated privileges.

For a GStreamer camera, test discovery and a minimal pipeline before using a
custom pipeline in Crappy:

.. code-block:: shell-session

   gst-inspect-1.0 --version
   gst-launch-1.0 autovideosrc ! videoconvert ! autovideosink

Then use ``gst-inspect-1.0 PLUGIN_NAME`` for each element needed by the custom
pipeline. A working command-line pipeline with a failing Python import points
to the PyGObject installation or environment. Follow the upstream
`GStreamer installation guide
<https://gstreamer.freedesktop.org/documentation/installing/>`_ and
`PyGObject setup guide <https://pygobject.gnome.org/getting_started.html>`_
for the current platform.

Data or images are not written
------------------------------

``PermissionError`` or ``Read-only file system``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolve the configured output path to an absolute path and check that its
parent directory can be created and written by the current user. Container,
network, and removable-media mounts can be read-only even when their files are
visible. On Windows, another application can also hold a file open. Test a
small output in a known writable temporary directory to distinguish a path
problem from a Recorder configuration problem.

Recorder and HDFRecorder do not overwrite an existing file. They add a suffix
such as ``_00001`` instead. ImageRecorder similarly selects another directory
when its target already contains recordings. Check the run's log for the final
path before concluding that no output was produced.

Crappy did not stop cleanly
---------------------------

Blocks remain active after the main script exits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During normal shutdown, Crappy asks every Block to stop and terminates Blocks
that do not finish promptly. Closing a terminal or stopping a script from an
IDE can bypass part of that cleanup. Before starting another hardware test:

- put actuators and other hazardous equipment in a safe state
- inspect the operating system's process list for Python workers started by
  the previous run
- terminate only workers that you have identified as belonging to that run
- close applications that may still own serial ports, cameras, or output
  files

If ownership is unclear, restarting the Python session or the computer is
safer than terminating an unrelated Python program. Repeated stale workers
usually indicate that a Block is stuck in a driver call. Preserve the log and
reduce that Block to a minimal script before reporting it.

``resource_tracker: There appear to be ... leaked shared_memory objects``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VisionBlocks exchange images through named shared-memory segments. A forced
termination can prevent the owning Block from closing and unlinking its
segment, which causes this warning at interpreter shutdown.

First ensure that no worker from the run is still active, then exit the Python
or IDE session completely. The resource tracker normally removes abandoned
segments. If the warning persists, restart the computer before another
hardware run. Do not delete arbitrary entries from the operating system's
shared-memory directory. Other applications may own them, and a running
Crappy pipeline may still be using a segment whose name begins with
``crappy_``.

Collect useful logs
-------------------

Crappy writes a new log file for each run at ``/tmp/crappy/logs.txt`` on Linux
and macOS, or
``C:\Users\<User>\AppData\Local\Temp\crappy\logs.txt`` on Windows. Copy the
file immediately after the failing run because the next run overwrites it.

Set ``debug=True`` on the faulty Block and, when the source is uncertain, on
the Blocks directly connected to it. Reproduce the failure once to capture
their detailed messages, and leave the ``log_level`` argument of
``crappy.start()`` at its default. Keep the complete traceback and terminal
output as well as ``logs.txt``. Before sharing these files, inspect them for
passwords, tokens, network addresses, private paths, and sensitive
experimental data.

.. _bug-report-checklist:

Report a bug
------------

The :doc:`support` page explains where to ask a usage question, report hardware
compatibility, or submit a reproducible defect. Search for the exact error text
in the documentation before opening a new issue.

A useful bug report should let another person reproduce the failure without
guessing. Include:

- What you wanted to achieve and the behavior you expected
- What you did, what happened instead, and whether the failure is consistent
- A minimal, complete script that fails
- The full traceback, terminal output, and ``logs.txt`` from a run with
  ``debug=True`` on the faulty Blocks
- The operating-system name and version, CPU architecture, Python version,
  and Crappy version
- In case the problem happens during communication with hardware, the relevant
  hardware model, connection type, driver, backend, or firmware version, and
  optional dependency versions
- Whether the hardware-free :doc:`quickstart <tutorials/quickstart>` works in
  the same environment and any troubleshooting already attempted

Remove unrelated Blocks, hardware, credentials, and private data from the
reproducer. Do not replace the minimal script with only a screenshot or a
fragment that cannot be run. If the problem cannot be reproduced safely,
describe the safety constraints and provide the smallest diagnostic output
that can be collected.
