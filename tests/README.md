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

The graphical tests also require Tk. It is included with the standard Python
installers on Windows and macOS, but may need to be installed separately on
Linux.

Running the tests
-----------------

Run the complete suite from the repository root with:

    python -m unittest -v tests

An individual package or test module can also be run directly, for example:

    python -m unittest -v tests.modifier
    python -m unittest -v tests.modifier.test_mean

The ``blocks_gui``, ``camera_configuration``, and ``camera_processes_gui``
packages open graphical interfaces and therefore require a display. On a
headless Linux system, run them through Xvfb, for example:

    xvfb-run --auto-servernum python -m unittest -v tests.blocks_gui

The tests run every non-graphical package with the Matplotlib ``Agg`` backend.
