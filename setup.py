import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("ciso._ciso", ["ciso/_ciso.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    # The package metadata is specified in the pyproject.toml but GitHub's downstream dependency graph
    # does not work unless we put the name this here too.
    name="ciso",
    ext_modules=cythonize(extensions),
    use_scm_version={
        "write_to": "ciso/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
