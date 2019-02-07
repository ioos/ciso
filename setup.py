import os
import sys
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize
import versioneer

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


long_description = '{}\n{}'.format(read('README.rst'), read('CHANGES.txt'))
LICENSE = read('LICENSE.txt')

with open('requirements.txt') as f:
    require = f.readlines()
install_requires = [r.strip() for r in require]

extensions = [Extension("ciso._ciso", ['ciso/_ciso.pyx'],
                        include_dirs=[numpy.get_include()])]

setup(name="ciso",
      version=versioneer.get_version(),
      license=LICENSE,
      long_description=long_description,
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Education',
                   ],
      description='Create isosurfaces from 2D or 3D arrays',
      url='https://github.com/ioos/ciso',
      platforms='any',
      keywords=['oceanography', 'isosurfaces', 'APIRUS'],
      install_requires=install_requires,
      packages=['ciso'],
      tests_require=['pytest'],
      cmdclass=versioneer.get_cmdclass(),
      ext_modules=cythonize(extensions),
      author=["Robert Hetland"],
      author_email="hetland@tamu.edu",
      )
