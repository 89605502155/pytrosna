"""Schema types: value types, time units, columns and devices."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .errors import InvalidArgumentError, SchemaError, UnsupportedError

__all__ = ["MAX_COLUMNS", "ColumnSchema", "DataType", "DeviceSchema", "TimeUnit"]

MAX_COLUMNS = 65_535
"""Maximum number of value columns of a device."""


class DataType(enum.IntEnum):
    """Type of the values of a column. The value is the code stored in files."""

    BOOL = 0
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    STRING = 5

    @property
    def label(self) -> str:
        """The name used in the API: ``"bool"``, ``"int32"``, …, ``"string"``."""
        return _TYPE_NAMES[self]

    def __str__(self) -> str:
        return self.label

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        """The NumPy dtype of the values (``object`` for strings)."""
        return np.dtype(_NUMPY_DTYPES[self])

    @property
    def plain_width(self) -> int:
        """Size of one value in ``Plain`` encoding (0 for strings, 1 for bools)."""
        return _PLAIN_WIDTHS[self]

    @classmethod
    def from_code(cls, code: int) -> DataType:
        try:
            return cls(code)
        except ValueError:
            msg = f"column type code {code}"
            raise UnsupportedError(msg) from None

    @classmethod
    def parse(cls, value: Any) -> DataType:
        """Accepts a :class:`DataType` or a name such as ``"float64"`` or ``"str"``."""
        if isinstance(value, DataType):
            return value
        if isinstance(value, str):
            try:
                return _TYPE_ALIASES[value.lower()]
            except KeyError:
                pass
        msg = f"unknown data type {value!r}"
        raise InvalidArgumentError(msg)


_TYPE_NAMES = {
    DataType.BOOL: "bool",
    DataType.INT32: "int32",
    DataType.INT64: "int64",
    DataType.FLOAT32: "float32",
    DataType.FLOAT64: "float64",
    DataType.STRING: "string",
}
_NUMPY_DTYPES = {
    DataType.BOOL: np.bool_,
    DataType.INT32: np.int32,
    DataType.INT64: np.int64,
    DataType.FLOAT32: np.float32,
    DataType.FLOAT64: np.float64,
    DataType.STRING: object,
}
_PLAIN_WIDTHS = {
    DataType.BOOL: 1,
    DataType.INT32: 4,
    DataType.INT64: 8,
    DataType.FLOAT32: 4,
    DataType.FLOAT64: 8,
    DataType.STRING: 0,
}
_TYPE_ALIASES = {
    "bool": DataType.BOOL,
    "boolean": DataType.BOOL,
    "int32": DataType.INT32,
    "i32": DataType.INT32,
    "int": DataType.INT32,
    "int64": DataType.INT64,
    "i64": DataType.INT64,
    "long": DataType.INT64,
    "float32": DataType.FLOAT32,
    "f32": DataType.FLOAT32,
    "float": DataType.FLOAT32,
    "float64": DataType.FLOAT64,
    "f64": DataType.FLOAT64,
    "double": DataType.FLOAT64,
    "string": DataType.STRING,
    "str": DataType.STRING,
    "utf8": DataType.STRING,
    "text": DataType.STRING,
}


class TimeUnit(enum.IntEnum):
    """Unit of the time stamps of a device. Time stamps are 64-bit counts of
    this unit since the Unix epoch (1970-01-01 UTC)."""

    SECOND = 0
    MILLISECOND = 1
    MICROSECOND = 2
    NANOSECOND = 3

    @property
    def label(self) -> str:
        """Short name: ``"s"``, ``"ms"``, ``"us"`` or ``"ns"``."""
        return ("s", "ms", "us", "ns")[self]

    def __str__(self) -> str:
        return self.label

    @property
    def per_second(self) -> int:
        """Number of units in one second."""
        return 10 ** (3 * self)

    @property
    def nanos(self) -> int:
        """Number of nanoseconds in one unit."""
        return 10 ** (9 - 3 * self)

    @classmethod
    def from_code(cls, code: int) -> TimeUnit:
        try:
            return cls(code)
        except ValueError:
            msg = f"time unit code {code}"
            raise UnsupportedError(msg) from None

    @classmethod
    def parse(cls, value: Any) -> TimeUnit:
        """Accepts a :class:`TimeUnit` or a name such as ``"ms"`` or ``"seconds"``."""
        if isinstance(value, TimeUnit):
            return value
        if isinstance(value, str):
            try:
                return _UNIT_ALIASES[value.lower()]
            except KeyError:
                pass
        msg = f"unknown time unit {value!r}"
        raise InvalidArgumentError(msg)

    def from_nanos(self, nanos: int) -> int:
        """Converts nanoseconds since the epoch to this unit, rounding down."""
        return nanos // self.nanos

    def to_nanos(self, value: int) -> int:
        """Converts a time stamp in this unit to nanoseconds."""
        return value * self.nanos


_UNIT_ALIASES = {
    "s": TimeUnit.SECOND,
    "sec": TimeUnit.SECOND,
    "second": TimeUnit.SECOND,
    "seconds": TimeUnit.SECOND,
    "ms": TimeUnit.MILLISECOND,
    "millisecond": TimeUnit.MILLISECOND,
    "milliseconds": TimeUnit.MILLISECOND,
    "us": TimeUnit.MICROSECOND,
    "µs": TimeUnit.MICROSECOND,
    "microsecond": TimeUnit.MICROSECOND,
    "microseconds": TimeUnit.MICROSECOND,
    "ns": TimeUnit.NANOSECOND,
    "nanosecond": TimeUnit.NANOSECOND,
    "nanoseconds": TimeUnit.NANOSECOND,
}


@dataclass(frozen=True)
class ColumnSchema:
    """Definition of a value column."""

    name: str
    data_type: DataType
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_type", DataType.parse(self.data_type))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DeviceSchema:
    """Definition of a device: a time column shared by a set of value columns.

    >>> DeviceSchema("vm01", "ms", columns=[ColumnSchema("cpu", "float64")]).time_unit
    <TimeUnit.MILLISECOND: 1>
    """

    name: str
    time_unit: TimeUnit = TimeUnit.MILLISECOND
    columns: tuple[ColumnSchema, ...] = ()
    timezone: str | None = None
    time_name: str = "time"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_unit", TimeUnit.parse(self.time_unit))
        columns = tuple(
            c if isinstance(c, ColumnSchema) else ColumnSchema(*c) for c in self.columns
        )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def build(
        cls,
        name: str,
        time_unit: TimeUnit | str,
        columns: dict[str, DataType | str] | None = None,
        **kwargs: Any,
    ) -> DeviceSchema:
        """Creates a schema from a ``{column: type}`` mapping.

        >>> DeviceSchema.build("room1", "s", {"temperature": "float64"}).column_names
        ['temperature']
        """
        return cls(
            name,
            TimeUnit.parse(time_unit),
            tuple(ColumnSchema(n, DataType.parse(t)) for n, t in (columns or {}).items()),
            **kwargs,
        )

    def with_column(
        self, name: str, data_type: DataType | str, metadata: dict[str, str] | None = None
    ) -> DeviceSchema:
        """A copy of the schema with one more value column."""
        column = ColumnSchema(name, DataType.parse(data_type), metadata or {})
        return replace(self, columns=(*self.columns, column))

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def types(self) -> list[DataType]:
        return [c.data_type for c in self.columns]

    def column_index(self, name: str) -> int | None:
        """Position of the value column with the given name."""
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        return None

    def validate(self) -> None:
        """Checks the constraints of SPEC §5.2."""
        if not self.name:
            msg = "device name must not be empty"
            raise SchemaError(msg)
        if not self.time_name:
            msg = "time column name must not be empty"
            raise SchemaError(msg)
        if not self.columns:
            msg = f"device {self.name!r} must have at least one value column"
            raise SchemaError(msg)
        if len(self.columns) > MAX_COLUMNS:
            msg = (
                f"device {self.name!r} has {len(self.columns)} columns; "
                f"at most {MAX_COLUMNS} are allowed"
            )
            raise SchemaError(msg)
        seen = {self.time_name}
        for column in self.columns:
            if not column.name:
                msg = "column names must not be empty"
                raise SchemaError(msg)
            if column.name in seen:
                msg = f"duplicate column name {column.name!r} in device {self.name!r}"
                raise SchemaError(msg)
            seen.add(column.name)

    def same_layout(self, other: DeviceSchema) -> bool:
        """True if both schemas have the same time unit and columns (names and types)."""
        return self.time_unit == other.time_unit and [
            (c.name, c.data_type) for c in self.columns
        ] == [(c.name, c.data_type) for c in other.columns]
