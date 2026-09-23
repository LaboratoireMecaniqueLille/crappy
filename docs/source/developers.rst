======================
Contributing to Crappy
======================

This page explains how to contribute to Crappy, build its documentation, and
understand its runtime architecture.

Prepare changes on the `develop branch
<https://github.com/LaboratoireMecaniqueLille/crappy/tree/develop>`_ and submit
them through a `GitHub pull request
<https://github.com/LaboratoireMecaniqueLille/crappy/compare>`_. The ``master``
branch contains published releases and is not committed to directly.

Code contributions
------------------

Crappy generally follows `PEP 8 <https://peps.python.org/pep-0008/>`_, with
two spaces instead of four for indentation. Use `Google-style docstrings
<https://google.github.io/styleguide/pyguide.html>`_ and document public
behavior, arguments, return values, exceptions, and lifecycle constraints.

Keep each commit focused and give it a concise title that describes the change.
In the pull request, explain why the change is needed, identify any
compatibility impact, and list the tests that were run. Update tests and
documentation whenever public behavior changes.

Testing changes
---------------

Crappy's tests use Python's built-in :mod:`unittest` framework. Install the
package and test dependencies from the repository root:

.. code-block:: console

   $ python -m pip install .
   $ python -m pip install -r tests/requirements.txt

During development, run the smallest relevant module or package first. For
example:

.. code-block:: console

   $ python -m unittest -v tests.modifier.test_mean
   $ python -m unittest -v tests.modifier

Run the complete suite before submitting a change that affects several parts
of Crappy:

.. code-block:: console

   $ python -m unittest -v tests

The standard test suite does not require physical hardware. Hardware APIs are
tested with simulated devices, test doubles, or patched dependencies. New
hardware-driver tests should remain runnable on an ordinary development
machine and in continuous integration.

The ``blocks_gui``, ``camera_configuration``, ``camera_processes_gui``, and
``vision_gui`` suites open graphical windows and require Tk and a display. On
a headless Linux system, run a graphical suite through Xvfb:

.. code-block:: console

   $ xvfb-run --auto-servernum python -m unittest -v tests.vision_gui

Set the Matplotlib backend to ``Agg`` when running non-graphical suites in a
headless environment. Documentation changes have their own checks:

.. code-block:: console

   $ python -m unittest -v tests.documentation
   $ make -C docs clean html

Continuous integration runs the non-graphical and graphical suites on every
supported Python version and operating system. See the repository's
`test-suite README
<https://github.com/LaboratoireMecaniqueLille/crappy/blob/develop/tests/README.md>`_
for the current package layout and additional examples.

Writing documentation
---------------------

Write for a reader trying to complete a task or understand observable
behavior. Explain the outcome and prerequisites first, then keep limitations,
safety information, hardware requirements, and file or graphical side effects
close to the step where they matter. Tutorials should contain a complete
runnable example. API pages should describe the interface without repeating
general concepts already covered elsewhere.

Use the established names for Blocks, Links, and hardware objects. In
particular, distinguish a Camera object, the all-in-one Camera Block,
CameraSource, and VisionBlock. The :doc:`image-pipeline concepts
<concepts/image_pipelines>` define these terms and the supported architectures.

Use American English, expand abbreviations when first introduced. Add genuine
technical names to ``docs/spelling_wordlist.txt`` rather than adding ordinary
misspellings.

After installing ``codespell``, run the editorial spelling check from the
repository root:

.. code-block:: console

   $ codespell docs/source docs/README.md --skip='*.py,*.cu' \
       --ignore-words=docs/spelling_wordlist.txt

Building the documentation
--------------------------

Sphinx uses the `Graphviz <https://graphviz.org/download/>`_ system package to
render the documentation diagrams. Install it with your operating system's
package manager and check that the ``dot`` executable is available:

.. code-block:: console

   $ dot -V

Build the documentation in a clean virtual environment with a supported
Python version. From the repository root, run:

.. code-block:: console

   $ python -m venv .venv-docs
   $ source .venv-docs/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install -r docs/source/requirements.txt
   $ python -m pip install .
   $ make -C docs clean html

The final command performs a fresh, nitpicky Sphinx build and treats warnings
as errors. The generated site is available in ``docs/build/html``.

You can run an external hypertext link audit locally with
``make -C docs checklinks``. Its categorized report is written to
``docs/build/linkcheck/output.txt``.

Runtime concepts and architecture
---------------------------------

The user-facing runtime contracts now live in the :doc:`concepts` section.
Use these pages when a tutorial or API entry needs to explain shared behavior:

- :doc:`concepts/blocks_links_labels`
- :doc:`concepts/lifecycle_shutdown`
- :doc:`concepts/regular_links_and_image_links`
- :doc:`concepts/image_pipelines`
- :doc:`concepts/choosing_custom_object_type`

The :doc:`architecture` guide documents process ownership, startup
coordination, transport internals, error propagation, cleanup, and the
all-in-one Camera internals for contributors. Keep details that may change
without affecting application code in that guide rather than in tutorials.
