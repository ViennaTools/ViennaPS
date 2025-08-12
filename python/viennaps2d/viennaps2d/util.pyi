"""
Utility functions.
"""
from __future__ import annotations
import viennals2d.viennals2d
__all__: list[str] = ['convertIntegrationScheme']
def convertIntegrationScheme(arg0: str) -> viennals2d.viennals2d.IntegrationSchemeEnum:
    """
    Convert a string to an integration scheme.
    """
