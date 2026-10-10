"""
Utility functions.
"""
from __future__ import annotations
import viennals._core
import viennaps._core
__all__: list[str] = ['convertBoundaryType', 'convertFluxEngineType', 'convertIntegrationScheme', 'convertOxidantType', 'convertSiliconOrientation', 'convertSpatialScheme', 'convertTemporalScheme']
def convertBoundaryType(arg0: str) -> viennals._core.BoundaryConditionEnum:
    """
    Convert a string to a boundary type.
    """
def convertFluxEngineType(arg0: str) -> viennaps._core.FluxEngineType:
    """
    Convert a string to a flux engine type.
    """
def convertOxidantType(arg0: str) -> viennaps._core.OxidantType:
    """
    Convert a string to an oxidant type.
    """
def convertSiliconOrientation(arg0: str) -> viennaps._core.SiliconOrientation:
    """
    Convert a string to a silicon orientation.
    """
def convertSpatialScheme(arg0: str) -> viennals._core.SpatialSchemeEnum:
    """
    Convert a string to an discretization scheme.
    """
def convertTemporalScheme(arg0: str) -> viennals._core.TemporalSchemeEnum:
    """
    Convert a string to a time integration scheme.
    """
convertIntegrationScheme = convertSpatialScheme
