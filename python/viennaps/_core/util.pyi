"""
Utility functions.
"""
from __future__ import annotations
__all__: list[str] = ['convertIntegrationScheme', 'convertSpatialScheme', 'convertTemporalScheme']
def convertSpatialScheme(arg0: str) -> ...:
    """
    Convert a string to an discretization scheme.
    """
def convertTemporalScheme(arg0: str) -> ...:
    """
    Convert a string to a time integration scheme.
    """
convertIntegrationScheme = convertSpatialScheme
