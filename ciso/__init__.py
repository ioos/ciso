"""Cythonized ISOsurface calculation (CISO)."""

from ciso.ciso import zslice

__all__ = ["zslice"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
