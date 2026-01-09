"""
Utility functions.
"""
from __future__ import annotations
import viennals._core
__all__: list[str] = ['convertIntegrationScheme', 'convertSpatialScheme', 'convertTemporalScheme']
def convertSpatialScheme(arg0: str) -> viennals._core.SpatialSchemeEnum:
    """
    Convert a string to an discretization scheme.
    """
def convertTemporalScheme(arg0: str) -> viennals._core.TemporalSchemeEnum:
    """
    Convert a string to a time integration scheme.
    """
convertIntegrationScheme = convertSpatialScheme
