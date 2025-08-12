"""
Utility functions.
"""
from __future__ import annotations
import viennals3d.viennals3d
__all__: list[str] = ['convertIntegrationScheme']
def convertIntegrationScheme(arg0: str) -> viennals3d.viennals3d.IntegrationSchemeEnum:
    """
    Convert a string to an integration scheme.
    """
