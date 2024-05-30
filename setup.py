import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("ciso._ciso", ["ciso/_ciso.pyx"], include_dirs=[numpy.get_include()]),
]

setup(ext_modules=cythonize(extensions))
