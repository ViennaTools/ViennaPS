"""
Ray tracing functions.
"""
from __future__ import annotations
import collections.abc
import enum
import typing
__all__: list[str] = ['ReflectionConedCosine', 'ReflectionDiffuse', 'ReflectionSpecular', 'rayTraceDirection']
class rayTraceDirection(enum.IntEnum):
    NEG_X: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_X: 1>
    NEG_Y: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_Y: 3>
    NEG_Z: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_Z: 5>
    POS_X: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_X: 0>
    POS_Y: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_Y: 2>
    POS_Z: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_Z: 4>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
def ReflectionConedCosine(arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg2: ..., arg3: typing.SupportsFloat) -> typing.Annotated[list[float], "FixedSize(3)"]:
    """
    Coned cosine reflection.
    """
def ReflectionDiffuse(arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: ...) -> typing.Annotated[list[float], "FixedSize(3)"]:
    """
    Diffuse reflection.
    """
def ReflectionSpecular(arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> typing.Annotated[list[float], "FixedSize(3)"]:
    """
    Specular reflection,
    """
