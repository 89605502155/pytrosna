"""Exceptions raised by pytrosna.

Every exception derives from :class:`TrosnaError`, so ``except TrosnaError``
catches all errors reported by the library. Damaged or malformed files raise
:class:`CorruptedError`; errors that also have a natural built-in counterpart
derive from it as well (for example :class:`UnknownDeviceError` is a
:class:`KeyError` and :class:`InvalidArgumentError` is a :class:`ValueError`).
"""

from __future__ import annotations

__all__ = [
    "CorruptedError",
    "DeviceExistsError",
    "InvalidArgumentError",
    "LimitExceededError",
    "LockedError",
    "NotFinalizedError",
    "NotTrosnaError",
    "PointNotFoundError",
    "SchemaError",
    "TrosnaError",
    "TypeMismatchError",
    "UnknownAnnotationError",
    "UnknownColumnError",
    "UnknownCommitError",
    "UnknownDeviceError",
    "UnsupportedError",
    "UnsupportedVersionError",
]


class TrosnaError(Exception):
    """Base class of all errors reported by pytrosna."""


class CorruptedError(TrosnaError):
    """The file contains malformed or damaged data.

    ``offset`` is the file offset of the damaged frame, if known.
    """

    def __init__(self, reason: str, offset: int | None = None) -> None:
        self.reason = reason
        self.offset = offset
        super().__init__(self._message())

    def _message(self) -> str:
        if self.offset is None:
            return f"corrupted data: {self.reason}"
        return f"corrupted data at offset {self.offset}: {self.reason}"

    def at(self, offset: int) -> CorruptedError:
        """Returns this error with ``offset`` attached if it has none yet."""
        if self.offset is None:
            self.offset = offset
            self.args = (self._message(),)
        return self


class NotTrosnaError(CorruptedError):
    """The file does not start with the Trosna magic bytes."""

    def __init__(self) -> None:
        super().__init__("not a Trosna file (bad magic bytes)")

    def _message(self) -> str:
        return "not a Trosna file (bad magic bytes)"


class UnsupportedVersionError(TrosnaError):
    """The file was written with an incompatible major version of the format."""

    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor
        super().__init__(
            f"unsupported format version {major}.{minor}; this library reads version 1.x"
        )


class UnsupportedError(TrosnaError):
    """The file uses a feature (frame kind, encoding, codec) unknown to this version,
    or data cannot be represented in a Trosna file."""


class NotFinalizedError(TrosnaError):
    """The file was not closed properly and strict opening was requested."""

    def __init__(self) -> None:
        super().__init__(
            "the file is not finalized (its writer crashed or is still writing); "
            "open it in non-strict mode or run pytrosna.recover()"
        )


class LockedError(TrosnaError):
    """Another writer holds the lock on the file."""

    def __init__(self) -> None:
        super().__init__("the file is locked by another writer")


class UnknownDeviceError(TrosnaError, KeyError):
    """The named device does not exist."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown device {name!r}")

    def __str__(self) -> str:
        return str(self.args[0])


class DeviceExistsError(TrosnaError):
    """A device with this name already exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"device {name!r} already exists")


class UnknownColumnError(TrosnaError, KeyError):
    """The named column does not exist in the device."""

    def __init__(self, device: str, column: str) -> None:
        self.device = device
        self.column = column
        super().__init__(f"unknown column {column!r} in device {device!r}")

    def __str__(self) -> str:
        return str(self.args[0])


class TypeMismatchError(TrosnaError, TypeError):
    """A column was given values of the wrong type."""


class SchemaError(TrosnaError, ValueError):
    """A schema is invalid or does not match the existing one."""


class PointNotFoundError(TrosnaError, KeyError):
    """There is no point at the given time stamp."""

    def __init__(self, device: str, time: int) -> None:
        self.device = device
        self.time = time
        super().__init__(f"device {device!r} has no point at time {time}")

    def __str__(self) -> str:
        return str(self.args[0])


class UnknownAnnotationError(TrosnaError, KeyError):
    """There is no annotation with the given identifier."""

    def __init__(self, annotation_id: int) -> None:
        self.id = annotation_id
        super().__init__(f"unknown annotation {annotation_id}")

    def __str__(self) -> str:
        return str(self.args[0])


class UnknownCommitError(TrosnaError, KeyError):
    """There is no commit with the given number."""

    def __init__(self, number: int) -> None:
        self.number = number
        super().__init__(f"unknown commit {number}")

    def __str__(self) -> str:
        return str(self.args[0])


class InvalidArgumentError(TrosnaError, ValueError):
    """An argument passed to the API is invalid."""


class LimitExceededError(TrosnaError, ValueError):
    """A size limit of the format was exceeded."""
