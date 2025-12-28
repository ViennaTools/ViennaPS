"""
Utility functions.
"""

from __future__ import annotations
import viennals._core

__all__: list[str] = ["convertSpatialScheme"]

def convertSpatialScheme(arg0: str) -> viennals._core.SpatialSchemeEnum:
    """
    Convert a string to an discretization scheme.
    """
