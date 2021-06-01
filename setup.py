# encoding: utf-8

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import find_packages, __version__
# To use a consistent encoding
from codecs import open
from os import popen, listdir, walk
import sys
from distutils.core import setup, Extension
import platform

# Get the long description from the relevant file
with open('docs/source/whatiscrappy.rst', encoding='utf-8') as f:
    long_description = f.read()

v = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
extensions = []
pyFgenModule = clModule = None


# uncomment following lines for testing extension module
# (see documentation "How to bin C/C++ with Crappy")
# helloModule = Extension('technical.helloModule',
#                         sources=['sources/hello/hello_class.cpp'],
#                         extra_compile_args=["-l", "python%s" % v])
#
# extensions.append(helloModule)

if platform.system() == "Linux":
    try:
      # Find the latest runtime version of SiliconSoftware install
      clPath = '/opt/SiliconSoftware/' + \
          sorted(next(walk('/opt/SiliconSoftware/'))[1])[-1] + '/lib64/'
    except StopIteration:
      print("Silicon Software not found, CameraLink will not be supported.")
      # If the software is installed but not found
      # just set clPath manually in this file
      clPath = None
    if clPath:
      clModule = Extension('camera.clModule',
                         sources=['sources/Cl_lib/CameraLink.cpp',
                           'sources/Cl_lib/pyCameraLink.cpp',
                           'sources/Cl_lib/clSerial.cpp'],
                         extra_compile_args=["-std=c++11"],
                         extra_link_args=["-l", "python%sm" % v, "-L", clPath,
                           "-l", "display", "-l", "clsersis", "-l", "fglib5"],
                         include_dirs=['/usr/local/lib/python%sm\
/dist-packages/numpy/core/include' % v])
      p = popen("lsmod |grep menable")
      if len(p.read()) != 0:
        print("menable kernel module found, installing CameraLink module.")
        extensions.append(clModule)
      else:
        print("Cannot find menable kernel module, "
            "CameraLink module won't be available.")

if platform.system() == "Windows":
    pyFgenModule = Extension('sensor.pyFgenModule',
         include_dirs=[
           "C:\\python%s\\site-packages\\numpy\\core\\include" % v,
           "C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\include",
           "C:\\Program Files\\IVI Foundation\\IVI\\Include"],
       sources=['sources/niFgen/pyFgen.cpp'], libraries=["niFgen"],
       library_dirs=["C:\\Program Files\\IVI Foundation\\IVI\\Lib_x64\\msc"],
       extra_compile_args=["/EHsc", "/WX"])
    if input("would you like to install pyFgen module? ([y]/n)") != "n":
        extensions.append(pyFgenModule)
    clpath = "C:\\Program Files\\SiliconSoftware\\Runtime5.2.1\\"
    clModule = Extension('sensor.clModule',
        include_dirs=[clpath+"include",
          "C:\\python{}\\Lib\\site-packages\\numpy\\core\\include".format(
            v.replace('.', ''))],
        sources=['sources/Cl_lib/CameraLink.cpp',
          'sources/Cl_lib/pyCameraLink.cpp',
          'sources/Cl_lib/clSerial.cpp'],
        libraries=["clsersis", "fglib5"],
        library_dirs=[clpath+"lib\\visualc"],
        extra_compile_args=["/EHsc", "/WX"])

    p = popen('driverquery /NH |findstr "me4"')
    if len(p.read()) != 0:
        extensions.append(clModule)
    else:
        print("Can't find microEnable4 Device driver, "
              "clModule will not be compiled")

setup(
  name='crappy',

  # Versions should comply with PEP440.  For a discussion on single-sourcing
  # the version across setup.py and the project code, see
  # https://packaging.python.org/en/latest/single_source_version.html
  version=__version__,

  description='Command and Real-time Acquisition Parallelized in Python',
  long_description=long_description,

  # The project's main homepage.
  url='https://github.com/LaboratoireMecaniqueLille/crappy',

  # Author details
  author='LaMcube',
  author_email='victor.couty@centralelille.fr',  # Create a mailing list!

  # Choose your license
  license='GPL V2',  # to confirm

  zip_safe=False,
  # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
  classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: \
GNU General Public License v2 or later (GPLv2+)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    # 'Programming Language :: Python :: 2',
    # 'Programming Language :: Python :: 2.6',
    # 'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    # 'Programming Language :: Python :: 3.2',
    # 'Programming Language :: Python :: 3.3',
    'Operating System :: POSIX :: Linux',
  ],

  # What does your project relate to?
  keywords='control command acquisition multiprocessing',

  # You can just specify the packages manually here if your project is
  # simple. Or you can use find_packages().
  packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

  ext_package='crappy',
  ext_modules=extensions,

  # List run-time dependencies here.  These will be installed by pip when
  # your project is installed. For an analysis of "install_requires" vs pip's
  # requirements files see:
  # https://packaging.python.org/en/latest/requirements.html
  install_requires=['numpy'],

  # If there are data files included in your packages that need to be
  # installed, specify them here.
  data_files=[('crappy/data', ['data/'+s for s in listdir('data')])]
)
