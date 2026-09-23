=====================
Support and reporting
=====================

Search the documentation and existing reports before opening a new one. The
:doc:`troubleshooting` page covers common error messages and recovery steps.
The :doc:`hardware` page records the drivers distributed with Crappy and their
known platforms and backends.

Choose the right channel
------------------------

Usage question
~~~~~~~~~~~~~~

Open a `GitHub Discussion
<https://github.com/LaboratoireMecaniqueLille/crappy/discussions/new/choose>`_
when you need help designing a test, selecting an object, or using an API. Say
what you want to achieve, include the smallest relevant script, and link any
documentation you already tried.

Reproducible defect
~~~~~~~~~~~~~~~~~~~

Open a `GitHub issue
<https://github.com/LaboratoireMecaniqueLille/crappy/issues/new/choose>`_ when
Crappy behaves incorrectly in a script that another person can run. Follow the
:ref:`bug-report checklist <bug-report-checklist>` and search the `existing
issues <https://github.com/LaboratoireMecaniqueLille/crappy/issues>`_ first.

Hardware compatibility report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with the :doc:`hardware matrix <hardware>`.

- If a device or platform combination is not listed, or you want to ask
  whether it is likely to work, open a GitHub Discussion.
- If a distributed driver fails for hardware that the matrix lists, reduce the
  failure to a minimal script and open a GitHub issue.
- If you tested an unverified combination successfully, report the exact
  combination in a GitHub Discussion so that maintainers can confirm and
  record it.

Include the device manufacturer, exact model and revision, connection type,
operating system, architecture, selected Crappy driver and backend, and any
vendor driver or firmware version. Describe what was tested and the observed
result. Do not infer compatibility from device-family names alone.

Feature or integration proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a GitHub Discussion before investing in a substantial new feature or
hardware integration. Describe the use case and proposed public interface.
For a focused code or documentation correction, follow the :doc:`contributor
guide <developers>` and submit a pull request against the ``develop`` branch.

Collect environment information
-------------------------------

Run the following command with the same Python interpreter as the failing
script:

.. code-block:: shell-session

   python -c "import crappy; print('Crappy:', crappy.__version__)"
   python --version
   python -c "import platform; print('OS:', platform.platform()); print('Architecture:', platform.machine())"

Also record the backend and hardware versions relevant to the report. For an
optional Python dependency, obtain the installed version with:

.. code-block:: shell-session

   python -m pip show PACKAGE_NAME

Remove passwords, access tokens, private network addresses, and sensitive
experimental data before posting. If a report cannot be made public safely,
do not open an issue containing the sensitive material.
