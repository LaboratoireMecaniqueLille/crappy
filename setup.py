"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import find_packages, __version__
# To use a consistent encoding
from codecs import open
from os import path, popen, listdir, walk
import sys
from distutils.core import setup, Extension
import platform

here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'about.txt'), encoding='utf-8') as f:
    long_description = f.read()

extentions = []
ximeaModule = pyFgenModule = clModule = None

v = "%d.%d"%(sys.version_info.major,sys.version_info.minor)
# uncomment folowing lines for testing extension module
# (see documentation "How to bin C/C++ with Crappy")
#helloModule = Extension('technical.helloModule',
#                         sources=['sources/hello/hello_class.cpp'],
#                         extra_compile_args=["-l", "python%s"%v])
#
#extentions.append(helloModule)
if platform.system() == "Linux":
    # read the current version in version.py
    exec(compile(open("./crappy/__version__.py").read(),
      "./crappy/__version__.py", 'exec'))
    ximeaModule = Extension('camera.ximeaModule',
            sources=['sources/XimeaLib/ximea.cpp',
              'sources/XimeaLib/pyXimea.cpp'],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-Werror", "-L", "../bin", "-L", "../bin/X64",
          "-L", "../bin/ARM", "-l", "m3api", "-l", "python%sm"%v],
        include_dirs=[
          '/usr/local/lib/python%s/dist-packages/numpy/core/include'%v])
    try:
      # Find the latest runtime version of SiliconSoftware install
      clPath = '/opt/SiliconSoftware/'+\
          sorted(next(walk('/opt/SiliconSoftware/'))[1])[-1]+'/lib64/'
    except StopIteration:
      print("Silicon Software not found, CameraLink won't be available.")
      # If the software is installed but not found
      # just set clPath manually in this file
      clPath = None
    if clPath:
      clModule = Extension('camera.clModule',
                         sources=['sources/Cl_lib/CameraLink.cpp',
                           'sources/Cl_lib/pyCameraLink.cpp',
                           'sources/Cl_lib/clSerial.cpp'],
                         extra_compile_args=["-std=c++11"],
                         extra_link_args=["-l", "python%sm"%v, "-L", clPath,
                           "-l", "display", "-l", "clsersis", "-l", "fglib5"],
                         include_dirs=['/usr/local/lib/python%sm\
/dist-packages/numpy/core/include'%v])
      p = popen("lsmod |grep menable")
      if len(p.read()) != 0:
        extentions.append(clModule)
      else:
        print("WARNING: cannot find menable kernel module, "+
            "CameraLink module won't be available.")

    if input("Install Ximea module? ([y]/n)?").lower() != "n":
        extentions.append(ximeaModule)

if platform.system() == "Windows":
    exec(compile(open(".\crappy\__version__.py").read(),
      ".\crappy\__version__.py", 'exec'))
    pyFgenModule = Extension('sensor.pyFgenModule',
         include_dirs=[
           "C:\\python%s\\site-packages\\numpy\\core\\include"%v,
           "C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\include",
           "C:\\Program Files\\IVI Foundation\\IVI\\Include"],
       sources=['sources/niFgen/pyFgen.cpp'], libraries=["niFgen"],
       library_dirs=["C:\\Program Files\\IVI Foundation\\IVI\\Lib_x64\\msc"],
       extra_compile_args=["/EHsc", "/WX"])
    if input("would you like to install pyFgen module? ([y]/n)") != "n":
        extentions.append(pyFgenModule)
    ximeaModule = Extension('sensor.ximeaModule',
        include_dirs=["c:\\XIMEA\\API",
          "C:\\python34\\Lib\\site-packages\\numpy\\core\\include"],
        sources=['sources/XimeaLib/ximea.cpp', 'sources/XimeaLib/pyximea.cpp'],
        libraries=["m3apiX64"],
        library_dirs=["c:\\XIMEA\\API", "c:\\XIMEA\\API\\x64"],
        extra_compile_args=["/EHsc", "/WX"])
    clModule = Extension('sensor.clModule',
        include_dirs=["C:\Program Files\SiliconSoftware\Runtime5.2.1\include",
          "C:\\python34\\Lib\\site-packages\\numpy\\core\\include"],
        sources=['sources/Cl_lib/CameraLink.cpp',
          'sources/Cl_lib/pyCameraLink.cpp',
          'sources/Cl_lib/clSerial.cpp'],
        libraries=["clsersis", "fglib5"],
        library_dirs=[
          "C:\\Program Files\\SiliconSoftware\\Runtime5.2.1\\lib\\visualc"],
        extra_compile_args=["/EHsc", "/WX"])

    p = popen('driverquery /NH |findstr "mu3camX64"')
    if len(p.read()) != 0:
        extentions.append(ximeaModule)
    else:
        print("Can't find Ximea driver,ximeaModule will not be compiled.")

    p = popen('driverquery /NH |findstr "me4"')
    if len(p.read()) != 0:
        extentions.append(clModule)
    else:
        print("Can't find microEnable4 Device driver, "+
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
  url='https://github.com/LaboratoireMecaniqueLille',

  # Author details
  author='LML',
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
  ext_modules=extentions,

  # List run-time dependencies here.  These will be installed by pip when
  # your project is installed. For an analysis of "install_requires" vs pip's
  # requirements files see:
  # https://packaging.python.org/en/latest/requirements.html
  install_requires=['scipy','numpy', 'matplotlib','pyserial','scikit-image'],

  # List additional groups of dependencies here (e.g. development
  # dependencies). You can install these using the following syntax,
  # for example:
  # $ pip install -e .[dev,test]
  extras_require={
    'dev': ['check-manifest'],
    'test': ['coverage'],
  },
  # If there are data files included in your packages that need to be
  # installed, specify them here.  If using Python 2.6 or less, then these
  # have to be included in MANIFEST.in as well.
  #data_files=[('crappy/data',['data/kernels.cu'])]
  data_files=[('crappy/data',['data/'+s for s in listdir('data')])]
  # package_data={
  # 'sample': ['package_data.dat'],
  # },

  # Although 'package_data' is the preferred approach, in some case you may
  # need to place data files outside of your packages. See:
  # http://docs.python.org/3.5/distutils/setupscript.html#installing-additional-files
  # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
  # data_files=[('my_data', ['data/data_file'])],

  # To provide executable scripts, use entry points in preference to the
  # "scripts" keyword. Entry points provide cross-platform support and allow
  # pip to create the appropriate form of executable for the target platform.
  # entry_points={
  # 'console_scripts': [
  # 'sample=sample:main',
  # ],
  # },
)
