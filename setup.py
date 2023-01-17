# encoding: utf-8

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import find_namespace_packages
from os import popen, walk
from distutils.core import setup, Extension
import platform

# Reading version from __version__.py file
with open('crappy/__version__.py') as file:
  for line in file:
    if line.startswith('__version__'):
      __version__ = line.split("'")[1]

# Get the long description from the relevant file
with open('docs/source/whatiscrappy.rst', encoding='utf-8') as f:
  long_description = f.read()

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
      sources=['sources/Cl_lib/CameraLink.cpp',
               'sources/Cl_lib/pyCameraLink.cpp',
               'sources/Cl_lib/clSerial.cpp'],
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
    sources=['sources/niFgen/pyFgen.cpp'], libraries=["niFgen"],
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
    sources=['sources/Cl_lib/CameraLink.cpp',
             'sources/Cl_lib/pyCameraLink.cpp', 'sources/Cl_lib/clSerial.cpp'],
    libraries=["clsersis", "fglib5"], library_dirs=[f"{cl_path}lib\\visualc"],
    extra_compile_args=["/EHsc", "/WX"])

  p = popen('driverquery /NH | findstr "me4"')
  if p.read():
    if input("would you like to install CameraLink module? ([y]/n)") != "n":
      extensions.append(cl_module)
  else:
    print("Can't find microEnable4 Device driver, clModule will not be "
          "compiled")

# Documentation aliases
SOURCE = 'docs/source'
CRAPPYDOCS = 'docs/source/crappydocs'
TUTORIALS = 'docs/source/tutorials'
# Explicitly listing all documentation files to avoid including unwanted files
docs_files = [
  ('crappy/docs', ['docs/Makefile']),
  (f'crappy/{SOURCE}', [f'{SOURCE}/embedded.rst',
                        f'{SOURCE}/installation.rst',
                        f'{SOURCE}/conf.py',
                        f'{SOURCE}/features.rst',
                        f'{SOURCE}/tutorials.rst',
                        f'{SOURCE}/blocklist.rst',
                        f'{SOURCE}/whatiscrappy.rst',
                        f'{SOURCE}/index.rst',
                        f'{SOURCE}/bugs.rst',
                        f'{SOURCE}/license.rst',
                        f'{SOURCE}/developers.rst',
                        f'{SOURCE}/citing.rst',
                        f'{SOURCE}/documentation.rst']),
  (f'crappy/{CRAPPYDOCS}', [f'{CRAPPYDOCS}/cameras.rst',
                            f'{CRAPPYDOCS}/blocks.rst',
                            f'{CRAPPYDOCS}/modifiers.rst',
                            f'{CRAPPYDOCS}/block.rst',
                            f'{CRAPPYDOCS}/inouts.rst',
                            f'{CRAPPYDOCS}/actuators.rst',
                            f'{CRAPPYDOCS}/tools.rst',
                            f'{CRAPPYDOCS}/links.rst']),
  (f'crappy/{TUTORIALS}', [f'{TUTORIALS}/gettingstarted.rst',
                           f'{TUTORIALS}/custom_blocks.rst',
                           f'{TUTORIALS}/c.rst'])]

# Explicitly listing all the example files to avoid including unwanted files
example_files = [('crappy/Examples',
                  ['Examples/microcontroller_example.py',
                   'Examples/thermocouple_daqmx.py',
                   'Examples/custom_actuator.py',
                   'Examples/ve_fake_test.py',
                   'Examples/drawing.py',
                   'Examples/thermocouple_t7.py',
                   'Examples/tensile_1.py',
                   'Examples/discorrel_basic.py',
                   'Examples/pid.py',
                   'Examples/modifiers.py',
                   'Examples/mean.py',
                   'Examples/videoextenso_simple.py',
                   'Examples/custom_block.py',
                   'Examples/stream_t7.py',
                   'Examples/multiplexer.py',
                   'Examples/spectrum.py',
                   'Examples/correl_basic.py',
                   'Examples/read_write.py',
                   'Examples/generator_steps.py',
                   'Examples/correl_strain_controlled_fake_test.py',
                   'Examples/displayer.py',
                   'Examples/generator.py',
                   'Examples/fake_test.py',
                   'Examples/custom_camera.py',
                   'Examples/correl_fake_test.py',
                   'Examples/custom_in.py',
                   'Examples/furnace_simulation.py',
                   'Examples/custom_out.py',
                   'Examples/photo.py',
                   'Examples/dio_daqmx.py',
                   'Examples/tensile_2.py',
                   'Examples/disve.py',
                   'Examples/gpucorrel_fake_test.py',
                   'Examples/read.py',
                   'Examples/correl_advanced.py'])]

# Explicitly listing all the util files to avoid including unwanted files
util_files = [('crappy/util', ['util/set_ft232h_serial_nr.py',
                               'util/udev_rule_setter.sh'])]

# Explicitly listing all the impact files to avoid including unwanted files
impact_files = [('crappy/impact', ['impact/biaxe.py',
                                   'impact/biotens.py',
                                   'impact/furnace.py',
                                   'impact/video_extenso_full.py',
                                   'impact/video_extenso_auto_drive.py'])]

setup(
  name='crappy',

  version=__version__,

  description='Command and Real-time Acquisition Parallelized in Python',

  long_description=long_description,

  url='https://github.com/LaboratoireMecaniqueLille/crappy',

  project_urls={'Documentation':
                'https://crappy.readthedocs.io/en/latest/index.html'},

  author='LaMcube',
  author_email='antoine.weisrock1@centralelille.fr',

  license='GPL V2',

  zip_safe=False,

  classifiers=['Development Status :: 4 - Beta ',
               'Intended Audience :: Science/Research',
               'Topic :: Software Development :: Build Tools',
               'License :: OSI Approved :: GNU General Public License v2 or '
               'later (GPLv2+)',
               'Programming Language :: Python :: 3.6',
               'Natural Language :: English',
               'Operating System :: OS Independent'],

  keywords='control command acquisition multiprocessing',

  packages=find_namespace_packages(exclude=['contrib', 'docs', 'tests*']),

  python_requires=">=3.6",

  ext_package='crappy',
  ext_modules=extensions,

  install_requires=["numpy>=1.21.0"],

  extras_require={'SBC': ['smbus2',
                          'spidev',
                          'Adafruit-Blinka',
                          'Adafruit-ADS1x15',
                          'adafruit-circuitpython-motorkit',
                          'adafruit-circuitpython-mprls',
                          'adafruit-circuitpython-busdevice'],
                  'image': ['opencv-python>=4.0',
                            'Pillow>=8.0.0',
                            'matplotlib>=3.3.0',
                            'SimpleITK>=2.0.0',
                            'scikit-image>=0.18.0'],
                  'hardware': ['pyusb>=1.1.0',
                               'pyserial>=3.4',
                               'pyyaml>=5.3'],
                  'main': ['matplotlib>=3.3.0',
                           'opencv-python>=4.0',
                           'pyserial>=3.4']},

  include_package_data=True,

  data_files=docs_files + example_files + util_files + impact_files
)
