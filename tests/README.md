Tests
=====

This directory contains Crappy's unit and integration test suites. The tests
use Python's built-in ``unittest`` package and are organized by the package
or feature they cover.

Installing the test dependencies
--------------------------------

From the repository root, install Crappy and the dependencies needed by the
test suite:

    python -m pip install .
    python -m pip install -r tests/requirements.txt

The graphical tests require PyQt6 for the default Button, Canvas, Dashboard,
StopButton, and camera-configuration interfaces. Grapher additionally requires
PyQtGraph. Both packages are included in ``tests/requirements.txt``.

The Tkinter compatibility tests also require Tk. It is included with the
standard Python installers on Windows and macOS, but may need to be installed
separately on Linux.

Running the tests
-----------------

Run the complete suite from the repository root with:

    python -m unittest -v tests

An individual package or test module can also be run directly, for example:

    python -m unittest -v tests.modifier
    python -m unittest -v tests.modifier.test_mean

Camera-configuration tests are grouped into ``tests.camera_configuration.base``
(headless shared behavior), ``tests.camera_configuration.tkinter``, and
``tests.camera_configuration.pyqt`` (backend integration). The aggregate
``tests.camera_configuration`` entry point runs all three layers.

The ``tests.blocks.test_gui_backends`` module checks backend selection,
argument validation, Link restrictions, optional imports, application setup,
and callback failures without a display. The ``tests.blocks_gui`` package
checks actual Tkinter and PyQt6 windows, widget callbacks, received values,
Canvas overlays, and cleanup. Common behavior is exercised on both backends.

The ``blocks_gui``, camera-configuration backend suites,
``camera_processes_gui``, and ``vision_gui`` packages open graphical interfaces
and therefore require a display. On a headless Linux system, run them through
Xvfb, for example:

    xvfb-run --auto-servernum python -m unittest -v tests.blocks_gui

The tests run every non-graphical package with the Matplotlib ``Agg`` backend.
