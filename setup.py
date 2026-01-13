"""Make extension."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

DEFINE_MACROS = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    # 0x030B0000 -> 3.11
    ("Py_LIMITED_API", "0x030B0000"),
    ("CYTHON_LIMITED_API", None),
]

extensions = [
    Extension(
        "ciso._ciso",
        ["ciso/_ciso.pyx"],
        include_dirs=[np.get_include()],
        define_macros=DEFINE_MACROS,
        py_limited_api=True,
    ),
]

setup(
    ext_modules=cythonize(extensions),
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
