# Tests

Crappy uses Python's built-in `unittest` framework. Run commands from the
repository root with the interpreter configured for the project.

## Install dependencies

```bash
python -m pip install .
python -m pip install -r tests/requirements.txt
```

`tests/requirements.txt` is the complete dependency file for all tests, 
including CPU image processing, codecs, HDF files, graph generation and
PyQt6.

Tkinter and Tcl/Tk come from Python or the operating system. Linux also needs
native Qt/OpenCV window libraries and a display or Xvfb with xauth. The package
workflow lists the Ubuntu runtime packages. Windows and macOS graphical runs 
need a native desktop session and a Python installation with working Tcl/Tk.

The suite uses software devices, synthetic images, and test extensions. It does
not test hardware drivers, GPU objects, MQTT services, or serial protocols.

## Find a test

| Directory | Coverage |
| --- | --- |
| `tests/unit/` | Feature tests, grouped by package. |
| `tests/integration/` | Real scripts with multi-processing. |
| `tests/gui/tkinter/` | Tkinter Blocks and camera-configuration windows. |
| `tests/gui/pyqt/` | PyQt Blocks, PyQtGraph, and camera-configuration windows. |
| `tests/gui/display/` | CameraProcess and VisionBlock image display. |
| `tests/documentation/` | Examples, downloads, references, and documentation metadata. |
| `tests/support/` | Shared fixtures and resource helpers. |
| `tests/gui/_support/` | GUI behavior helpers shared by the camera configuration backend test classes. |

## Run headless tests

Run all headless tests, integration scenarios, and documentation checks with:

```bash
python -m unittest discover -s tests -t . -v
```

Graphical modules are intentionally skipped unless their group is explicitly
selected. An available display does not automatically enable them. In a
headless environment, set `MPLBACKEND=Agg` before starting Python.

CI runs the same directory discovery through a capability wrapper:

```bash
python -m tests.run tests/unit
python -m tests.run tests/integration
python -m tests.run tests/documentation
```

You can run a test level, feature, module, class, or method directly:

```bash
python -m unittest discover -s tests/unit -t . -v
python -m unittest discover -s tests/integration -t . -v
python -m unittest discover -s tests/documentation -t . -v
python -m unittest discover -s tests/unit/modifier -t . -v
python -m unittest discover -s tests/unit/modifier -t . -p 'test_mean.py' -v
python -m unittest tests.unit.modifier.test_mean -v
python -m unittest tests.unit.modifier.test_mean.TestMean -v
python -m unittest tests.unit.modifier.test_mean.TestMean.test_averages_all_labels_independently -v
```

Keep `-t .` in discovery commands so test modules and multiprocessing targets
have the same importable names. Loading a package by name alone, such as
`python -m unittest tests.unit.modifier`, does not recursively discover it.

## Run graphical tests

Set `CRAPPY_TEST_GUI` to exactly one of `tkinter`, `pyqt`, or `display` and
discover the corresponding directory in a fresh interpreter. Set
`MPLBACKEND=TkAgg` for these commands.

For a native display, these Bash examples run each group separately:

```bash
CRAPPY_TEST_GUI=tkinter python -m tests.run tests/gui/tkinter
CRAPPY_TEST_GUI=pyqt python -m tests.run tests/gui/pyqt
CRAPPY_TEST_GUI=display python -m tests.run tests/gui/display/camera_processes
CRAPPY_TEST_GUI=display python -m tests.run tests/gui/display/vision
```

On Windows PowerShell, set the environment before the corresponding command,
for example `$env:CRAPPY_TEST_GUI = 'pyqt'`, then run
`python -m tests.run tests/gui/pyqt`.
Use a fresh Python process for each group. Do not collect Tk and Qt tests in
the same interpreter.

On headless Linux, prefix each GUI command with Xvfb:

```bash
CRAPPY_TEST_GUI=tkinter xvfb-run --auto-servernum python -m tests.run tests/gui/tkinter
```

On macOS, Tk and image-display groups use the existing Tcl-lifetime wrapper:

```bash
CRAPPY_TEST_GUI=tkinter MPLBACKEND=TkAgg python .github/scripts/run_macos_tk_tests.py discover -s tests/gui/tkinter -t . -v
```
