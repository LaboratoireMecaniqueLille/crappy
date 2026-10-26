# Tests

Crappy uses Python's built-in `unittest` framework. Run commands from the
repository root with the interpreter configured for the project.

## Install dependencies

```bash
python -m pip install .
python -m pip install -r tests/requirements.txt
```

The existing test requirements include the CPU image/file-processing and PyQt6
dependencies. Tkinter also needs Tk, which may require a separate system 
package on Linux. GUI groups require a working display or Xvfb.

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
CRAPPY_TEST_GUI=tkinter MPLBACKEND=TkAgg python -m unittest discover -s tests/gui/tkinter -t . -v
CRAPPY_TEST_GUI=pyqt MPLBACKEND=TkAgg python -m unittest discover -s tests/gui/pyqt -t . -v
CRAPPY_TEST_GUI=display MPLBACKEND=TkAgg python -m unittest discover -s tests/gui/display/camera_processes -t . -v
CRAPPY_TEST_GUI=display MPLBACKEND=TkAgg python -m unittest discover -s tests/gui/display/vision -t . -v
```

On Windows PowerShell, set the environment before the corresponding command,
for example `$env:CRAPPY_TEST_GUI = 'pyqt'` and `$env:MPLBACKEND = 'TkAgg'`.
Use a fresh Python process for each group. Do not collect Tk and Qt tests in
the same interpreter.

On headless Linux, prefix each GUI command with Xvfb:

```bash
CRAPPY_TEST_GUI=tkinter MPLBACKEND=TkAgg xvfb-run --auto-servernum python -m unittest discover -s tests/gui/tkinter -t . -v
```

On macOS, Tk and image-display groups use the existing Tcl-lifetime wrapper:

```bash
CRAPPY_TEST_GUI=tkinter MPLBACKEND=TkAgg python .github/scripts/run_macos_tk_tests.py discover -s tests/gui/tkinter -t . -v
```
