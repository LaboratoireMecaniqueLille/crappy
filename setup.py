# encoding: utf-8

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import find_packages
# To use a consistent encoding
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
                                          "-l", "display", "-l", "clsersis",
                                          "-l", "fglib5"],
                         include_dirs=['/usr/local/lib/python%sm'
                                       '/dist-packages/numpy/core/include' % v])
    p = popen("lsmod |grep menable")
    if len(p.read()) != 0:
      print("menable kernel module found, installing CameraLink module.")
      extensions.append(clModule)
    else:
      print("Cannot find menable kernel module, "
            "CameraLink module won't be available.")

if platform.system() == "Windows":
  pyFgenModule = Extension('sensor.pyFgenModule',
                           include_dirs=["C:\\python%s\\site-packages\\numpy\\"
                                         "core\\include" % v,
                                         "C:\\Program Files (x86)\\"
                                         "IVI Foundation\\VISA\\WinNT\\include",
                                         "C:\\Program Files\\IVI Foundation\\"
                                         "IVI\\Include"],
                           sources=['sources/niFgen/pyFgen.cpp'],
                           libraries=["niFgen"],
                           library_dirs=["C:\\Program Files\\IVI Foundation\\"
                                         "IVI\\Lib_x64\\msc"],
                           extra_compile_args=["/EHsc", "/WX"])
  if input("would you like to install pyFgen module? ([y]/n)") != "n":
    extensions.append(pyFgenModule)
  clpath = "C:\\Program Files\\SiliconSoftware\\Runtime5.2.1\\"
  clModule = Extension('sensor.clModule',
                       include_dirs=[clpath+"include",
                                     "C:\\python{}\\Lib\\site-packages\\numpy\\"
                                     "core\\include".format(
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
    print("Can't find microEnable4 Device driver, clModule will not be "
          "compiled")

docs_files = [('crappy/' + path, [path + '/' + name for name in paths]) for
              (path, _, paths) in walk('docs/source')]

setup(
  name='crappy',

  version='1.4.3',

  description='Command and Real-time Acquisition Parallelized in Python',

  long_description=long_description,

  url='https://github.com/LaboratoireMecaniqueLille/crappy',

  project_urls={'Documentation':
                'https://crappy.readthedocs.io/en/latest/index.html'},

  author='LaMcube',
  author_email='victor.couty@centralelille.fr',

  license='GPL V2',

  zip_safe=False,

  classifiers=['Development Status :: 5 - Production/Stable ',
               'Intended Audience :: Science/Research',
               'Topic :: Software Development :: Build Tools',
               'License :: OSI Approved :: GNU General Public License v2 or '
               'later (GPLv2+)',
               'Programming Language :: Python :: 3.6',
               'Natural Language :: English',
               'Operating System :: OS Independent'],

  keywords='control command acquisition multiprocessing',

  packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

  python_requires=">=3.6",

  ext_package='crappy',
  ext_modules=extensions,

  install_requires=['numpy'],

  data_files=[('crappy/data',
               ['data/' + filename for filename in listdir('data')]),

              ('crappy/docs', ['docs/Makefile']),

              ('crappy/Examples',
               ['Examples/' + filename for filename in listdir('Examples')]),

              ('crappy/impact',
               ['impact/' + filename for filename in listdir('impact')]),

              ('crappy/util',
               ['util/' + filename for filename in listdir('util')])

              ] + docs_files
)
