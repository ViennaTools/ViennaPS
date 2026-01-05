"""
Physical and material constants.
"""
from __future__ import annotations
import typing
__all__: list[str] = ['N_A', 'R', 'celsiusToKelvin', 'gasMeanFreePath', 'gasMeanThermalVelocity', 'kB', 'roomTemperature', 'torrToPascal']
def celsiusToKelvin(arg0: typing.SupportsFloat) -> float:
    """
    Convert temperature from Celsius to Kelvin.
    """
def gasMeanFreePath(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat) -> float:
    """
    Calculate the mean free path of a gas molecule.
    """
def gasMeanThermalVelocity(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat) -> float:
    """
    Calculate the mean thermal velocity of a gas molecule.
    """
def torrToPascal(arg0: typing.SupportsFloat) -> float:
    """
    Convert pressure from torr to pascal.
    """
N_A: float = 6022.1367
R: float = 8.314
kB: float = 8.617333262000001e-05
roomTemperature: float = 300.0
