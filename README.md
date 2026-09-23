Command and Real-time Acquisition in Parallelized PYthon (CRAPPY)
=================================================================

![Crappy](https://raw.githubusercontent.com/LaboratoireMecaniqueLille/crappy/master/docs/source/_static/branding/banner_1024.svg)

[![Downloads](https://static.pepy.tech/badge/crappy)](https://www.pepy.tech/projects/crappy)
[![Documentation Status](https://readthedocs.org/projects/crappy/badge/?version=latest)](https://crappy.readthedocs.io/en/latest/)
[![PyPI version](https://badgen.net/pypi/v/crappy/)](https://pypi.org/project/crappy/)
[![Python versions](https://img.shields.io/pypi/pyversions/crappy.svg)](https://pypi.org/project/crappy/)
[![Test Python Package](https://github.com/LaboratoireMecaniqueLille/crappy/actions/workflows/test_python_package.yml/badge.svg)](https://github.com/LaboratoireMecaniqueLille/crappy/actions/workflows/test_python_package.yml)

Crappy is an open-source Python framework for command and data acquisition on
experimental setups. A Crappy test is a Python script assembled from **Blocks**
that acquire measurements, drive actuators, process data, display signals, or
save results. **Links** carry labeled data between these Blocks.

Crappy is developed at [LaMCube](https://lamcube.univ-lille.fr/), a mechanical
research laboratory in Lille, France, where it is used mainly for materials
testing.

## Why use Crappy?

As experimental setups grow, coordinating instruments from different vendors,
organizing acquisition and commands, and reproducing the same test procedure
can become increasingly difficult. Commercial packages may tie a setup to a
specific ecosystem, while writing a complete control application from scratch
requires time and software expertise.

Crappy provides a common, modular framework for these tasks. Ready-to-use
Blocks can be combined and replaced as an experiment evolves, and users can use
included hardware drivers or integrate custom devices. Because each test is an 
ordinary Python script, its complete workflow can be saved, shared, and 
repeated. Crappy is also designed to use computer resources efficiently, 
helping beginners and experts build responsive experiments without having to
write the underlying software.

## What can Crappy do?

- Acquire measurements from sensors and laboratory instruments
- Drive actuators and build feedback loops
- Capture, process, display, and record images
- Display and save experimental data as it is acquired
- Integrate custom hardware and processing code for user-specific needs
- Run on Linux, Windows, macOS, and Raspberry Pi

## Installation

Crappy requires Python 3.10 or newer and NumPy 2.0 or newer. Install it from
PyPI with:

```shell
python -m pip install crappy
```

See the [installation guide](https://crappy.readthedocs.io/en/latest/installation.html)
for platform-specific instructions and optional dependencies.

## Example script

This example reads the computer's memory usage, displays it live, records it in
`data.csv`, and stops automatically after ten seconds. It requires no  physical 
hardware, but uses Matplotlib for the graph and psutil for the simulated input:

```shell
python -m pip install matplotlib psutil
```

```python
import crappy

if __name__ == '__main__':
  acquisition = crappy.blocks.IOBlock('FakeInOut', labels=('t(s)', 'ram(%)'))
  graph = crappy.blocks.Grapher(('t(s)', 'ram(%)'))
  recorder = crappy.blocks.Recorder('data.csv', labels=('t(s)', 'ram(%)'))
  stop = crappy.blocks.StopBlock('t(s) > 10')

  crappy.link(acquisition, graph)
  crappy.link(acquisition, recorder)
  crappy.link(acquisition, stop)

  crappy.start()
```

Refer to the documentation for a detailed explanation of each Block's purpose
and capability.

## Documentation and support

Start with [Is Crappy right for you?](https://crappy.readthedocs.io/en/latest/what_is_crappy.html)
for a broader overview of the framework, its intended uses, and its limits.

- The [tutorials](https://crappy.readthedocs.io/en/latest/tutorials.html) provide
  guided introductions to common tasks and custom objects
- The [examples](https://crappy.readthedocs.io/en/latest/examples.html) provide
  complete scripts organized by task, with and without physical hardware
- The [core concepts](https://crappy.readthedocs.io/en/latest/concepts.html)
  explain how Blocks, Links, labels, and test lifecycles fit together
- The [hardware matrix](https://crappy.readthedocs.io/en/latest/hardware.html)
  lists the available drivers and their known platform and backend support
- The [API reference](https://crappy.readthedocs.io/en/latest/api.html) documents
  the arguments and methods of Crappy's public objects

If something does not work as expected, consult the
[troubleshooting guide](https://crappy.readthedocs.io/en/latest/troubleshooting.html)
first. The [support and reporting guide](https://crappy.readthedocs.io/en/latest/support.html)
then explains where and how to ask a question, report a bug, or share hardware
compatibility results.

Usage questions are welcome in
[GitHub Discussions](https://github.com/LaboratoireMecaniqueLille/crappy/discussions).
Use the [issue tracker](https://github.com/LaboratoireMecaniqueLille/crappy/issues)
for reproducible bugs and hardware compatibility reports.

## Citing Crappy

If Crappy contributes to published research, please cite:

> Couty V., Witz J.-F., Martel C. et al. *Command and Real-Time Acquisition in
> Parallelized Python, a Python module for experimental setups*. SoftwareX 16,
> 2021. [doi:10.1016/j.softx.2021.100848](https://doi.org/10.1016/j.softx.2021.100848)

See the [citation guidance](https://crappy.readthedocs.io/en/latest/citing.html)
and [`CITATION.cff`](https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/CITATION.cff)
for structured metadata.

## License

Crappy is distributed under the
[GNU General Public License v2.0 or later](https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/LICENSE).
Copyright &copy; 2015–present, Laboratoire Mécanique de Lille and contributors.
