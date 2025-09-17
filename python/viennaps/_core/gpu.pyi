"""
GPU support functions.
"""

from __future__ import annotations
import typing

__all__: list[str] = ["Context", "Path"]

class Context:
    @staticmethod
    def createContext(
        modulePath: Path = "",
        deviceID: typing.SupportsInt = 0,
        registerInGlobal: bool = True,
    ) -> Context:
        """
        Create a new context.
        """

    @staticmethod
    def getContextFromRegistry(deviceID: typing.SupportsInt = 0) -> Context:
        """
        Get a context from the global registry by device ID.
        """

    @staticmethod
    def getRegisteredDeviceIDs() -> list[int]:
        """
        Get a list of all device IDs with registered contexts.
        """

    @staticmethod
    def hasContextInRegistry(deviceID: typing.SupportsInt = 0) -> bool:
        """
        Check if a context exists in the global registry by device ID.
        """

    def __init__(self) -> None: ...
    def addModule(self, arg0: str) -> None:
        """
        Add a module to the context.
        """

    def create(
        self,
        modulePath: Path = "",
        deviceID: typing.SupportsInt = 0,
    ) -> None:
        """
        Create a new context.
        """

    def destroy(self) -> None:
        """
        Destroy the context.
        """

    def getModulePath(self) -> str:
        """
        Get the module path.
        """

    @property
    def deviceID(self) -> int:
        """
        Device ID.
        """

    @deviceID.setter
    def deviceID(self, arg0: typing.SupportsInt) -> None: ...

class Path:
    def __init__(self, arg0: str) -> None: ...
