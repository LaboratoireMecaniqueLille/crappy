# encoding: utf-8

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, Extension
from os import popen, walk
import platform
from pathlib import Path

# Reading version from __version__.py file
with open('src/crappy/__version__.py') as file:
  for line in file:
    if line.startswith('__version__'):
      __version__ = line.split("'")[1]

# Get the long description from the relevant file
long_description = Path('README.md').read_text()

# Getting the current version of Python
py_ver = '.'.join(platform.python_version().split('.')[:2])

# The list of extensions to install
extensions = []

# Now finding the extensions to install
if platform.system() == "Linux":
  # Find the latest runtime version of SiliconSoftware install
  try:
    cl_path = f"/opt/SiliconSoftware/" \
              f"{sorted(next(walk('/opt/SiliconSoftware/'))[1])[-1]}/lib64/"
  except StopIteration:
    # If the software is installed but not found just set cl_path manually here
    cl_path = None

  if cl_path is not None:
    cl_module = Extension(
      'camera.cameralink.clModule',
      sources=['src/ext/Cl_lib/CameraLink.cpp',
               'src/ext/Cl_lib/pyCameraLink.cpp',
               'src/ext/Cl_lib/clSerial.cpp'],
      extra_compile_args=["-std=c++11"],
      extra_link_args=["-l", f"python{py_ver}", "-L", cl_path, "-l", "display",
                       "-l", "clsersis", "-l", "fglib5"],
      include_dirs=[f'/usr/local/lib/python{py_ver}/dist-packages/numpy/'
                    f'core/include'])
    p = popen("lsmod | grep menable")
    if p.read():
      if input("would you like to install CameraLink module? ([y]/n)") != "n":
        print("menable kernel module found, installing CameraLink module.")
        extensions.append(cl_module)
      else:
        print("menable kernel module found, but not installing")
    else:
      print("Cannot find menable kernel module, CameraLink module won't be "
            "available.")

if platform.system() == "Windows":

  py_fgen_module = Extension(
    'tool.pyFgenModule',
    include_dirs=[f"C:\\python{py_ver}\\site-packages\\numpy\\core\\include",
                  "C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\"
                  "include",
                  "C:\\Program Files\\IVI Foundation\\IVI\\Include"],
    sources=['src/ext/niFgen/pyFgen.cpp'], libraries=["niFgen"],
    library_dirs=["C:\\Program Files\\IVI Foundation\\IVI\\Lib_x64\\msc"],
    extra_compile_args=["/EHsc", "/WX"])

  if input("would you like to install pyFgen module? ([y]/n)") != "n":
    extensions.append(py_fgen_module)

  cl_path = "C:\\Program Files\\SiliconSoftware\\Runtime5.2.1\\"
  cl_module = Extension(
    'tool.clModule',
    include_dirs=[f"{cl_path}include",
                  f"C:\\python{py_ver.replace('.', '')}\\Lib\\site-packages\\"
                  f"numpy\\core\\include"],
    sources=['src/ext/Cl_lib/CameraLink.cpp',
             'src/ext/Cl_lib/pyCameraLink.cpp', 'src/ext/Cl_lib/clSerial.cpp'],
    libraries=["clsersis", "fglib5"], library_dirs=[f"{cl_path}lib\\visualc"],
    extra_compile_args=["/EHsc", "/WX"])

  p = popen('driverquery /NH | findstr "me4"')
  if p.read():
    if input("would you like to install CameraLink module? ([y]/n)") != "n":
      extensions.append(cl_module)
  else:
    print("Can't find microEnable4 Device driver, clModule will not be "
          "compiled")

setup(ext_package='crappy', ext_modules=extensions)
