"""
Utility functions.
"""

from __future__ import annotations
import viennals._core

__all__: list[str] = ["convertIntegrationScheme"]

def convertIntegrationScheme(arg0: str) -> viennals._core.IntegrationSchemeEnum:
    """
    Convert a string to an integration scheme.
    """
