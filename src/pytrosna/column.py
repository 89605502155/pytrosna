"""In-memory columns and batches.

A :class:`Column` is a NumPy array of values plus an optional validity mask
(``True`` = the value is present). A :class:`Batch` is a set of rows of one
device: an ``int64`` array of raw time stamps and named columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import InvalidArgumentError, TypeMismatchError
from .types import DataType

if TYPE_CHECKING:
    from .types import DeviceSchema

__all__ = ["Batch", "Column"]

_INT_RANGES = {
    DataType.INT32: (-(1 << 31), (1 << 31) - 1),
    DataType.INT64: (-(1 << 63), (1 << 63) - 1),
}


def _null_fill(data_type: DataType) -> Any:
    return "" if data_type is DataType.STRING else 0


class Column:
    """A column of values of one :class:`~pytrosna.DataType` with optional nulls.

    >>> c = Column.from_values("int64", [1, None, 3])
    >>> c.to_list(), c.null_count
    ([1, None, 3], 1)
    """

    __slots__ = ("data_type", "validity", "values")

    def __init__(
        self,
        data_type: DataType | str,
        values: np.ndarray,
        validity: np.ndarray | None = None,
    ) -> None:
        self.data_type = DataType.parse(data_type)
        values = np.asarray(values)
        if values.ndim != 1:
            msg = "column values must be one-dimensional"
            raise InvalidArgumentError(msg)
        if values.dtype != self.data_type.numpy_dtype:
            msg = (
                f"a {self.data_type} column needs {self.data_type.numpy_dtype} values, "
                f"not {values.dtype}"
            )
            raise TypeMismatchError(msg)
        if validity is not None:
            validity = np.asarray(validity, dtype=np.bool_)
            if validity.shape != values.shape:
                msg = "the validity mask must have the length of the values"
                raise InvalidArgumentError(msg)
            if validity.all():
                validity = None
        self.values = values
        self.validity = validity

    # ------------------------------------------------------------ construction

    @classmethod
    def from_values(
        cls, data_type: DataType | str, values: Iterable[Any], *, nan_is_null: bool = False
    ) -> Column:
        """Builds a column from Python values; ``None`` is a null.

        Integers are range-checked, floats become integers only if they are
        whole numbers, booleans and strings must have exactly that type.
        With ``nan_is_null``, NaN values of float columns are stored as nulls.
        """
        data_type = DataType.parse(data_type)
        items = list(values)
        valid = np.fromiter((v is not None for v in items), dtype=np.bool_, count=len(items))
        fill = _null_fill(data_type)
        converted = [fill if v is None else _convert(v, data_type) for v in items]
        if data_type is DataType.STRING:
            array = np.empty(len(converted), dtype=object)
            array[:] = converted
        else:
            array = np.array(converted, dtype=data_type.numpy_dtype)
        if nan_is_null and data_type in (DataType.FLOAT32, DataType.FLOAT64):
            valid &= ~np.isnan(array)
            array[~valid] = 0
        return cls(data_type, array, valid)

    @classmethod
    def nulls(cls, data_type: DataType | str, length: int) -> Column:
        """A column of ``length`` nulls."""
        data_type = DataType.parse(data_type)
        if data_type is DataType.STRING:
            values = np.full(length, "", dtype=object)
        else:
            values = np.zeros(length, dtype=data_type.numpy_dtype)
        return cls(data_type, values, np.zeros(length, dtype=np.bool_))

    @classmethod
    def concat(cls, columns: Sequence[Column]) -> Column:
        """Concatenates columns of the same type."""
        if not columns:
            msg = "nothing to concatenate"
            raise InvalidArgumentError(msg)
        data_type = columns[0].data_type
        if any(c.data_type is not data_type for c in columns):
            msg = "cannot concatenate columns of different types"
            raise TypeMismatchError(msg)
        values = np.concatenate([c.values for c in columns])
        if all(c.validity is None for c in columns):
            return cls(data_type, values)
        return cls(data_type, values, np.concatenate([c.valid_mask() for c in columns]))

    # ------------------------------------------------------------ access

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return f"Column({self.data_type.label}, {self.to_list()!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        if self.data_type is not other.data_type or len(self) != len(other):
            return False
        mine, theirs = self.valid_mask(), other.valid_mask()
        if not np.array_equal(mine, theirs):
            return False
        a, b = self.values[mine], other.values[theirs]
        if self.data_type in (DataType.FLOAT32, DataType.FLOAT64):
            return bool(np.array_equal(a.view(_bits_dtype(a)), b.view(_bits_dtype(b))))
        return bool(np.array_equal(a, b))

    __hash__ = None  # type: ignore[assignment]

    @property
    def null_count(self) -> int:
        return 0 if self.validity is None else int(self.validity.size - self.validity.sum())

    def valid_mask(self) -> np.ndarray:
        """A boolean mask of the rows that hold a value."""
        if self.validity is None:
            return np.ones(len(self), dtype=np.bool_)
        return self.validity

    def is_valid(self, index: int) -> bool:
        return self.validity is None or bool(self.validity[index])

    def value(self, index: int) -> Any:
        """The value at ``index`` as a Python object (``None`` for a null)."""
        if not self.is_valid(index):
            return None
        return _python_value(self.values[index], self.data_type)

    def __getitem__(self, index: int) -> Any:
        return self.value(index)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.to_list())

    def to_list(self) -> list[Any]:
        """The values as Python objects, ``None`` for nulls."""
        out = list(self.values) if self.data_type is DataType.STRING else self.values.tolist()
        if self.validity is not None:
            for i in np.flatnonzero(~self.validity).tolist():
                out[i] = None
        return out

    def to_numpy(self) -> np.ndarray:
        """The values as a NumPy array; nulls are NaN for floats and ``None`` otherwise
        (which makes integer and boolean arrays with nulls ``object`` arrays)."""
        if self.validity is None:
            return self.values.copy()
        if self.data_type in (DataType.FLOAT32, DataType.FLOAT64):
            out = self.values.copy()
            out[~self.validity] = np.nan
            return out
        out = self.values.astype(object)
        out[~self.validity] = None
        return out

    def dense(self) -> np.ndarray:
        """Only the non-null values."""
        return self.values if self.validity is None else self.values[self.validity]

    def slice(self, start: int, stop: int) -> Column:
        validity = None if self.validity is None else self.validity[start:stop]
        return Column(self.data_type, self.values[start:stop], validity)

    def take(self, indices: np.ndarray) -> Column:
        validity = None if self.validity is None else self.validity[indices]
        return Column(self.data_type, self.values[indices], validity)


def _bits_dtype(a: np.ndarray) -> type[np.unsignedinteger[Any]]:
    return np.uint32 if a.dtype == np.float32 else np.uint64


def _python_value(value: Any, data_type: DataType) -> Any:
    if data_type is DataType.STRING:
        return value
    if data_type is DataType.BOOL:
        return bool(value)
    if data_type in (DataType.INT32, DataType.INT64):
        return int(value)
    return float(value)


def _convert(value: Any, data_type: DataType) -> Any:
    """Converts a Python value to the column type if that is lossless."""
    if isinstance(value, np.generic):
        value = value.item()
    if data_type is DataType.BOOL:
        if isinstance(value, bool):
            return value
    elif data_type is DataType.STRING:
        if isinstance(value, str):
            return value
    elif data_type in (DataType.INT32, DataType.INT64):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            low, high = _INT_RANGES[data_type]
            if low <= value <= high:
                return value
            msg = f"{value} does not fit into {data_type}"
            raise TypeMismatchError(msg)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # Any number is stored in a float column, rounded to the nearest float.
        return float(value)
    msg = f"a {type(value).__name__} value cannot be stored in a {data_type} column"
    raise TypeMismatchError(msg)


class Batch:
    """Rows of one device: raw time stamps and named columns of equal length.

    ``schema`` is the :class:`~pytrosna.DeviceSchema` the rows were read from
    (``None`` for a batch built by hand); it gives the time unit and time zone
    used by :meth:`to_pandas`, :meth:`to_arrow` and :meth:`to_polars`.

    >>> b = Batch([1, 2], {"cpu": Column.from_values("float64", [0.5, 0.25])})
    >>> len(b), b.row(1)
    (2, {'cpu': 0.25})
    """

    __slots__ = ("columns", "schema", "time")

    def __init__(
        self,
        time: Sequence[int] | np.ndarray,
        columns: Mapping[str, Column] | None = None,
        schema: DeviceSchema | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=np.int64)
        if self.time.ndim != 1:
            msg = "time stamps must be one-dimensional"
            raise InvalidArgumentError(msg)
        self.columns: dict[str, Column] = dict(columns or {})
        for name, column in self.columns.items():
            if not isinstance(column, Column):
                msg = f"column {name!r} must be a pytrosna.Column"
                raise InvalidArgumentError(msg)
            if len(column) != self.time.size:
                msg = f"column {name!r} has {len(column)} values for {self.time.size} time stamps"
                raise InvalidArgumentError(msg)
        self.schema = schema

    @classmethod
    def empty(cls, columns: Mapping[str, DataType], schema: DeviceSchema | None = None) -> Batch:
        return cls(
            np.empty(0, dtype=np.int64),
            {name: Column.nulls(t, 0) for name, t in columns.items()},
            schema,
        )

    @classmethod
    def concat(cls, batches: Sequence[Batch]) -> Batch:
        if not batches:
            msg = "nothing to concatenate"
            raise InvalidArgumentError(msg)
        names = list(batches[0].columns)
        return cls(
            np.concatenate([b.time for b in batches]),
            {n: Column.concat([b.columns[n] for b in batches]) for n in names},
            batches[0].schema,
        )

    def __len__(self) -> int:
        return int(self.time.size)

    def __repr__(self) -> str:
        device = f", device={self.schema.name!r}" if self.schema else ""
        return f"Batch({len(self)} rows, columns={list(self.columns)}{device})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Batch):
            return NotImplemented
        return (
            np.array_equal(self.time, other.time)
            and list(self.columns) == list(other.columns)
            and all(self.columns[n] == other.columns[n] for n in self.columns)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def __getitem__(self, name: str) -> Column:
        return self.columns[name]

    def row(self, index: int) -> dict[str, Any]:
        """The values of one row by column name."""
        return {name: c.value(index) for name, c in self.columns.items()}

    def rows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """``(time, {column: value})`` for every row."""
        lists = {name: c.to_list() for name, c in self.columns.items()}
        for i, t in enumerate(self.time.tolist()):
            yield t, {name: values[i] for name, values in lists.items()}

    def to_dict(self) -> dict[str, list[Any]]:
        """``{time_name: [raw time stamps], column: [values], …}``."""
        time_name = self.schema.time_name if self.schema else "time"
        out: dict[str, list[Any]] = {time_name: self.time.tolist()}
        for name, column in self.columns.items():
            out[name] = column.to_list()
        return out

    def slice(self, start: int, stop: int) -> Batch:
        return Batch(
            self.time[start:stop],
            {n: c.slice(start, stop) for n, c in self.columns.items()},
            self.schema,
        )

    def take(self, indices: np.ndarray) -> Batch:
        return Batch(
            self.time[indices],
            {n: c.take(indices) for n, c in self.columns.items()},
            self.schema,
        )

    def select(self, names: Iterable[str]) -> Batch:
        """A batch with only the named columns, in the given order."""
        return Batch(self.time, {n: self.columns[n] for n in names}, self.schema)

    # ------------------------------------------------------------ conversions

    def datetimes(self) -> np.ndarray:
        """The time stamps as ``numpy.datetime64`` values (UTC)."""
        unit = self.schema.time_unit.label if self.schema else "ns"
        return self.time.astype(f"datetime64[{unit}]")

    def to_arrow(self) -> Any:
        """Converts the batch to a ``pyarrow.Table`` (needs ``pyarrow``)."""
        from .adapters import batch_to_arrow

        return batch_to_arrow(self)

    def to_pandas(self, *, dtype_backend: str = "numpy") -> Any:
        """Converts the batch to a ``pandas.DataFrame`` (needs ``pandas``).

        ``dtype_backend="numpy_nullable"`` uses pandas' nullable dtypes, so
        integer and boolean columns with nulls keep their type.
        """
        from .adapters import batch_to_pandas

        return batch_to_pandas(self, dtype_backend=dtype_backend)

    def to_polars(self) -> Any:
        """Converts the batch to a ``polars.DataFrame`` (needs ``polars``)."""
        from .adapters import batch_to_polars

        return batch_to_polars(self)
