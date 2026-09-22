"""Sphinx configuration for the documentation of Crappy"""

from datetime import date
from importlib.metadata import version as distribution_version
from os import environ


# Project metadata

project = "Crappy"
author = "LaMcube and contributors"
copyright = f"{date.today().year}, {author}"

# Documentation builds install Crappy before loading this configuration.
release = distribution_version("crappy")
version = ".".join(release.split(".")[:2])


# Extensions and source files

needs_sphinx = "8.1"
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.intersphinx",
              "sphinx.ext.viewcode",
              "sphinx.ext.autosectionlabel",
              "sphinx.ext.napoleon",
              "sphinx.ext.mathjax",
              "sphinx.ext.graphviz",
              "sphinx_copybutton",
              "sphinx_rtd_theme"]

source_suffix = {".rst": "restructuredtext"}
language = "en"
highlight_language = "python3"


# Cross-references

nitpicky = True

# DOC-08 will classify and reduce this legacy list. Keep each exception narrow
# until its source annotation or external inventory can be corrected
nitpick_ignore = {
    ("py:mod", "smbus2"),
    ("py:mod", "Adafruit-Blinka"),
    ("py:mod", "adafruit-circuitpython-motorkit"),
    ("py:mod", "Phidget22"),
    ("py:mod", "cv2"),
    ("py:mod", "opencv-python"),
    ("py:mod", "PIL"),
    ("py:mod", "SimpleITK"),
    ("py:mod", "tables"),
    ("py:mod", "PyGObject"),
    ("py:mod", "picamera"),
    ("py:mod", "picamera2"),
    ("py:mod", "ximea"),
    ("py:mod", "PyDAQmx"),
    ("py:mod", "adafruit-circuitpython-busdevice"),
    ("py:mod", "adafruit-circuitpython-mcp9600"),
    ("py:mod", "adafruit-circuitpython-ads1x15"),
    ("py:mod", "adafruit-circuitpython-mprls"),
    ("py:mod", "nidaqmx"),
    ("py:mod", "pijuice"),
    ("py:mod", "smbus"),
    ("py:mod", "spidev"),
    ("py:mod", "pyusb"),
    ("py:mod", "pip"),
    ("py:mod", "pycuda"),
    ("py:mod", "crappy.tool.camera_config"),
    ("py:mod", "crappy.tool.image_processing"),
    ("py:mod", "gphoto2"),
    ("py:class", "pathlib._local.Path"),
    ("py:class", "multiprocessing.synchronize.RLock"),
    ("py:class", "_io.FileIO"),
    ("py:class", "multiprocessing.sharedctypes.Synchronized"),
    ("py:class", "multiprocessing.sharedctypes.SynchronizedArray"),
    ("py:class", "multiprocessing.managers.DictProxy"),
    ("py:class", "multiprocessing.synchronize.Barrier"),
    ("py:class", "multiprocessing.synchronize.Event"),
    ("py:class", "multiprocessing.queues.Queue"),
    ("py:class", "picamera.PiCamera"),
    ("py:class", "smbus2.i2c_msg"),
    ("py:class", "crappy.tool.image_processing.gpu_correl.CorrelStage"),
    ("py:class", "b'RES'"),
    ("py:class", "b'VOLT'"),
    ("py:obj", "tables.Atom"),
    ("py:obj", "multiprocessing.Connection"),
    ("py:meth", "crappy.blocks.UController.send_to_pc"),
    ("py:meth", "crappy.camera.Camera.__getattribute__"),
    ("py:meth", "nidaqmx.task.add_ai"),
    ("py:meth", "nidaqmx.task.add_ao_voltage_chan"),
    ("py:meth", "nidaqmx.task.add_do_chan"),
    ("py:meth", "nidaqmx.task.add_di_chan"),
    ("py:meth", "nidaqmx.task.add_ai_[type]_chan"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "psutil": ("https://psutil.readthedocs.io/stable/", None),
}

# Prefix generated labels so headings in different documents cannot collide
autosectionlabel_prefix_document = True


# API documentation

# API pages select public and exceptional private members explicitly
autodoc_member_order = "bysource"
autodoc_mock_imports = ["PIL",
                        "RPi",
                        "cv2",
                        "gphoto2",
                        "picamera",
                        "pymodbus",
                        "serial",
                        "spidev",
                        "usb",
                        "ue9"]

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_rtype = False


# HTML output

html_theme = "sphinx_rtd_theme"
html_theme_options = {"prev_next_buttons_location": "both",
                      "collapse_navigation": False,
                      "sticky_navigation": False,
                      "includehidden": False}
html_title = f"{project} {release} documentation"
html_baseurl = environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://crappy.readthedocs.io/en/latest").rstrip("/")
html_context = {"display_github": True,
                "github_user": "LaboratoireMecaniqueLille",
                "github_repo": "crappy",
                "github_version": "master/docs/source/"}

# Add only reviewed root-level files, such as llms.txt, in a later work
# package
html_extra_path = []


# Link checking

# Unexpected redirects remain reportable. Timeouts do not fail an audit because
# several hardware-vendor sites throttle automated checks
linkcheck_allowed_redirects = {}
linkcheck_report_timeouts_as_broken = False
