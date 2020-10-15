from distutils.core import setup
from distutils.extension import Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
from Cython.Distutils import build_ext

ext_modules = [Extension("WHAM.binned", ["WHAM/binned.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("WHAM.lib.potentials", ["WHAM/lib/potentials.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
               Extension("WHAM.lib.timeseries", ["WHAM/lib/timeseries.pyx"],
               include_dirs=[numpy_include],
               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]

setup(
    name='WHAM',
    version='0.0',
    packages=['WHAM'],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
