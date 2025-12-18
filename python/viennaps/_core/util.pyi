"""
Utility functions.
"""

from __future__ import annotations
import viennals._core

__all__: list[str] = ["convertDiscretizationScheme"]

def convertDiscretizationScheme(arg0: str) -> viennals._core.DiscretizationSchemeEnum:
    """
    Convert a string to an discretization scheme.
    """
