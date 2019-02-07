from .ciso import zslice

__all__ = ['zslice']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
