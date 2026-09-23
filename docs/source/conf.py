"""Sphinx configuration for the documentation of Crappy"""

from datetime import date
from importlib.metadata import version as distribution_version
from os import environ
from pathlib import Path
from sys import path as python_path


python_path.insert(0, str(Path(__file__).resolve().parent))


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
              "sphinx.ext.autosummary",
              "sphinx.ext.coverage",
              "sphinx.ext.extlinks",
              "sphinx.ext.intersphinx",
              "sphinx.ext.viewcode",
              "sphinx.ext.autosectionlabel",
              "sphinx.ext.napoleon",
              "sphinx.ext.mathjax",
              "sphinx.ext.graphviz",
              "sphinx_copybutton",
              "sphinx_rtd_theme",
              "_ext.hardware_matrix"]

source_suffix = {".rst": "restructuredtext"}
language = "en"
highlight_language = "python3"


# Cross-references

nitpicky = True

nitpick_ignore = {
    # Python's inventory deliberately does not document these implementations
    ("py:class", "multiprocessing.synchronize.RLock"),
    ("py:class", "multiprocessing.sharedctypes.Synchronized"),
    ("py:class", "multiprocessing.sharedctypes.SynchronizedArray"),
    ("py:class", "multiprocessing.managers.DictProxy"),
    ("py:class", "multiprocessing.synchronize.Barrier"),
    ("py:class", "multiprocessing.synchronize.Event"),
    ("py:class", "multiprocessing.queues.Queue"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "nidaqmx": ("https://nidaqmx-python.readthedocs.io/en/latest/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "picamera": ("https://picamera.readthedocs.io/en/release-1.13/", None),
    "psutil": ("https://psutil.readthedocs.io/stable/", None),
    "pycuda": ("https://documen.tician.de/pycuda/", None),
    "pytables": ("https://pytables.readthedocs.io/en/stable/", None),
    "smbus2": ("https://smbus2.readthedocs.io/en/latest/", None),
}

# Prefix generated labels so headings in different documents cannot collide
autosectionlabel_prefix_document = True

extlinks = {
    "example": (
        "https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/"
        "examples/%s",
        "%s"),
}


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

# The coverage builder measures the reviewed core API modules registered by
# automodule on their reference pages
coverage_ignore_modules = [r"crappy\.tool\.bindings\.(comedi_bind|pyspcm)"]
# Don't count objects with no docstring in coverage
coverage_skip_undoc_in_source = True


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

# Hand-reviewed files copied to the root of the generated documentation
html_extra_path = ["llms.txt"]


# Link checking

# Unexpected redirects remain reportable. Timeouts do not fail an audit because
# several hardware-vendor sites throttle automated checks
linkcheck_allowed_redirects = {}
linkcheck_report_timeouts_as_broken = False
