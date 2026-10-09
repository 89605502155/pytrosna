"""Adapters between Trosna batches and the Python data stack.

Reading: :func:`batch_to_arrow`, :func:`batch_to_pandas` and
:func:`batch_to_polars` turn a :class:`~pytrosna.Batch` into a
``pyarrow.Table``, ``pandas.DataFrame`` or ``polars.DataFrame``.

Writing: :func:`normalize` accepts a pandas or Polars DataFrame, a PyArrow
Table, RecordBatch or RecordBatchReader, an object exporting an Arrow stream,
a :class:`~pytrosna.Batch` or a mapping ``{column: values}`` and yields
:class:`SourceTable` objects; :func:`infer_device` derives a device schema
from one and :func:`to_batch` converts one into rows of a device.

pandas, Polars and PyArrow are optional: each is imported only when data of
that kind is converted.

Type mapping (Trosna → Arrow / pandas / Polars):

=========  =====================  ===========================  ==================
Trosna     Arrow                  pandas                       Polars
=========  =====================  ===========================  ==================
time       timestamp(unit, tz)    datetime64[unit, tz]         Datetime(unit, tz)
bool       bool                   bool / object / boolean      Boolean
int32      int32                  int32 / float64 / Int32      Int32
int64      int64                  int64 / float64 / Int64      Int64
float32    float32                float32 / Float32            Float32
float64    float64                float64 / Float64            Float64
string     string                 object or str / string       String
=========  =====================  ===========================  ==================
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .column import Batch, Column
from .errors import InvalidArgumentError, SchemaError, TypeMismatchError, UnsupportedError
from .timeconv import _datetime_ns, parse_text, tzinfo
from .types import ColumnSchema, DataType, DeviceSchema, TimeUnit

__all__ = [
    "SourceColumn",
    "SourceTable",
    "batch_to_arrow",
    "batch_to_pandas",
    "batch_to_polars",
    "guess_time_column",
    "infer_device",
    "normalize",
    "to_batch",
]

TIME_NAMES = ("time", "timestamp", "ts", "datetime", "date", "t", "время")
"""Column names recognised as the time column, in order of preference."""

_UNITS = {"s": TimeUnit.SECOND, "ms": TimeUnit.MILLISECOND, "us": TimeUnit.MICROSECOND}
_UNITS["ns"] = TimeUnit.NANOSECOND


def _require(module: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError:
        msg = f"this needs {module}; install it with: pip install 'pytrosna[{extra}]'"
        raise ImportError(msg) from None


# ====================================================================== reading


def _time_unit(batch: Batch) -> TimeUnit:
    return batch.schema.time_unit if batch.schema else TimeUnit.NANOSECOND


def _time_name(batch: Batch) -> str:
    return batch.schema.time_name if batch.schema else "time"


def _timezone(batch: Batch) -> str | None:
    return batch.schema.timezone if batch.schema else None


def _column_metadata(batch: Batch, name: str) -> dict[str, str]:
    if batch.schema is None:
        return {}
    index = batch.schema.column_index(name)
    return {} if index is None else batch.schema.columns[index].metadata


def batch_to_arrow(batch: Batch) -> Any:
    """Converts a batch to a ``pyarrow.Table``. The schema metadata holds the
    device metadata and the device name under ``trosna.device``."""
    pa = _require("pyarrow", "arrow")
    unit = _time_unit(batch)
    fields = [
        pa.field(_time_name(batch), pa.timestamp(unit.label, _timezone(batch)), nullable=False)
    ]
    arrays = [pa.array(batch.time, type=pa.int64()).cast(fields[0].type)]
    types = {
        DataType.BOOL: pa.bool_(),
        DataType.INT32: pa.int32(),
        DataType.INT64: pa.int64(),
        DataType.FLOAT32: pa.float32(),
        DataType.FLOAT64: pa.float64(),
        DataType.STRING: pa.string(),
    }
    for name, column in batch.columns.items():
        arrow_type = types[column.data_type]
        mask = None if column.validity is None else ~column.validity
        if column.data_type is DataType.STRING:
            array = pa.array(column.values.tolist(), type=arrow_type, mask=mask)
        else:
            array = pa.array(column.values, type=arrow_type, mask=mask)
        fields.append(pa.field(name, arrow_type, metadata=_column_metadata(batch, name) or None))
        arrays.append(array)
    metadata = {}
    if batch.schema is not None:
        metadata = dict(batch.schema.metadata)
        metadata["trosna.device"] = batch.schema.name
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata or None))


def batch_to_pandas(batch: Batch, *, dtype_backend: str = "numpy") -> Any:
    """Converts a batch to a ``pandas.DataFrame`` with the time as its first
    column (``datetime64`` in the device's unit, aware if the device has a time zone).

    With ``dtype_backend="numpy"`` (default) nulls become NaN in numeric
    columns (integer columns with nulls become ``float64``), ``None`` in
    boolean and string columns. With ``"numpy_nullable"`` pandas' nullable
    dtypes (``Int64``, ``boolean``, ``Float64``, ``string``) keep the types.
    """
    pd = _require("pandas", "pandas")
    if dtype_backend not in ("numpy", "numpy_nullable"):
        msg = "dtype_backend must be 'numpy' or 'numpy_nullable'"
        raise InvalidArgumentError(msg)
    unit = _time_unit(batch)
    time = pd.Series(batch.time.astype(f"datetime64[{unit.label}]"), name=_time_name(batch))
    zone = tzinfo(_timezone(batch))
    if zone is not None:
        time = time.dt.tz_localize("UTC").dt.tz_convert(zone)
    data: dict[str, Any] = {time.name: time}
    nullable = {
        DataType.BOOL: "boolean",
        DataType.INT32: "Int32",
        DataType.INT64: "Int64",
        DataType.FLOAT32: "Float32",
        DataType.FLOAT64: "Float64",
        DataType.STRING: "string",
    }
    for name, column in batch.columns.items():
        if dtype_backend == "numpy_nullable":
            values = column.values.copy()
            if column.data_type is DataType.STRING:
                values = values.astype(object)
            series = pd.Series(values, dtype=nullable[column.data_type])
            if column.validity is not None:
                series[~column.validity] = pd.NA
        elif column.validity is not None and column.data_type in (DataType.INT32, DataType.INT64):
            # Like pyarrow: integers with nulls become floats with NaN.
            values = column.values.astype(np.float64)
            values[~column.validity] = np.nan
            series = pd.Series(values)
        else:
            series = pd.Series(column.to_numpy())
        data[name] = series.rename(name)
    return pd.DataFrame(data)


def _polars_timezone(tz: str | None) -> str | None:
    """Polars accepts only IANA time zones; fixed offsets become ``Etc/GMT±H``
    when whole hours, otherwise UTC (the instants are unchanged)."""
    if tz is None:
        return None
    import re

    m = re.match(r"^([+-])(\d{2})(?::?(\d{2}))?$", tz)
    if m is None:
        return "UTC" if tz.upper() == "Z" else tz
    hours, minutes = int(m.group(2)), int(m.group(3) or 0)
    if minutes == 0 and hours == 0:
        return "UTC"
    limit = 14 if m.group(1) == "+" else 12  # Etc/GMT-14 … Etc/GMT+12 exist
    if minutes == 0 and 0 < hours <= limit:
        # POSIX-style names have the opposite sign.
        return f"Etc/GMT{'-' if m.group(1) == '+' else '+'}{hours}"
    return "UTC"


def batch_to_polars(batch: Batch) -> Any:
    """Converts a batch to a ``polars.DataFrame``. Polars has no second
    resolution, so such time stamps become milliseconds."""
    pl = _require("polars", "polars")
    unit = _time_unit(batch)
    raw = batch.time
    if unit is TimeUnit.SECOND:
        raw = raw * 1000
        unit = TimeUnit.MILLISECOND
    time = pl.Series(_time_name(batch), raw, dtype=pl.Int64).cast(pl.Datetime(unit.label))
    tz = _polars_timezone(_timezone(batch))
    if tz is not None:
        time = time.dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
    types = {
        DataType.BOOL: pl.Boolean,
        DataType.INT32: pl.Int32,
        DataType.INT64: pl.Int64,
        DataType.FLOAT32: pl.Float32,
        DataType.FLOAT64: pl.Float64,
        DataType.STRING: pl.String,
    }
    series = [time]
    for name, column in batch.columns.items():
        values = column.values.tolist() if column.data_type is DataType.STRING else column.values
        s = pl.Series(name, values, dtype=types[column.data_type])
        if column.validity is not None:
            s = s.scatter(np.flatnonzero(~column.validity), None)
        series.append(s)
    return pl.DataFrame(series)


# ====================================================================== writing


@dataclass
class SourceColumn:
    """A value column of input data, normalized to NumPy."""

    name: str
    data_type: DataType | None
    """The Trosna type that stores the values losslessly (``None``: only nulls)."""
    values: np.ndarray
    valid: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SourceTable:
    """Input data normalized to NumPy: a time column and value columns.

    ``time`` holds integers in ``time_unit``; ``time_unit`` is ``None`` for an
    integer time column of unknown unit. Times given as text or ``datetime``
    objects are kept in nanoseconds, and ``device_unit`` is then the coarsest
    unit that represents them exactly (the unit of a device created from
    them). ``naive`` marks wall-clock times without a time zone.
    """

    time_name: str
    time: np.ndarray
    time_unit: TimeUnit | None
    timezone: str | None
    naive: bool
    columns: list[SourceColumn] = field(default_factory=list)
    device_unit: TimeUnit | None = None

    def __post_init__(self) -> None:
        if self.device_unit is None:
            self.device_unit = self.time_unit

    def __len__(self) -> int:
        return int(self.time.size)


def guess_time_column(names: Sequence[str], timestamp_names: Sequence[str] = ()) -> str:
    """The first timestamp column, or the first column named like a time."""
    if timestamp_names:
        return timestamp_names[0]
    lowered = {name.lower(): name for name in reversed(names)}
    for candidate in TIME_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    msg = "cannot tell which column holds the time; pass time_column=..."
    raise InvalidArgumentError(msg)


def _numpy_type(dtype: np.dtype[Any]) -> DataType | None:
    kind = dtype.kind
    if kind == "b":
        return DataType.BOOL
    if kind in "iu":
        if dtype.itemsize < 4 or (kind == "i" and dtype.itemsize == 4):
            return DataType.INT32
        return DataType.INT64
    if kind == "f":
        return DataType.FLOAT32 if dtype.itemsize <= 4 else DataType.FLOAT64
    if kind in "US":
        return DataType.STRING
    return None


def _object_type(values: Sequence[Any], name: str) -> DataType | None:
    """The type of Python objects (``None`` and NaN-free)."""
    kinds = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            kinds.add(DataType.BOOL)
        elif isinstance(v, (int, np.integer)):
            kinds.add(DataType.INT64)
        elif isinstance(v, (float, np.floating)):
            kinds.add(DataType.FLOAT64)
        elif isinstance(v, str):
            kinds.add(DataType.STRING)
        else:
            msg = f"column {name!r} holds a {type(v).__name__}, which Trosna cannot store"
            raise UnsupportedError(msg)
    if not kinds:
        return None
    if kinds == {DataType.INT64, DataType.FLOAT64}:
        return DataType.FLOAT64
    if len(kinds) > 1:
        msg = f"column {name!r} mixes values of types {sorted(k.label for k in kinds)}"
        raise UnsupportedError(msg)
    return kinds.pop()


def _from_objects(name: str, values: Sequence[Any], nan_is_null: bool = False) -> SourceColumn:
    items = list(values)
    if nan_is_null:
        items = [None if isinstance(v, float) and v != v else v for v in items]  # noqa: PLR0124
    data_type = _object_type(items, name)
    valid = np.fromiter((v is not None for v in items), dtype=np.bool_, count=len(items))
    if data_type is None:
        return SourceColumn(name, None, np.zeros(len(items)), valid)
    column = Column.from_values(data_type, items)
    return SourceColumn(name, data_type, column.values, column.validity)


def _time_from_objects(
    name: str, values: Sequence[Any]
) -> tuple[np.ndarray, TimeUnit, str | None, bool]:
    """Times given as text, ``datetime``/``date`` or ``numpy.datetime64`` objects."""
    nanos: list[int] = []
    timezone: str | None = None
    naive_seen = aware_seen = False
    for v in values:
        if v is None:
            msg = "the time column contains nulls"
            raise UnsupportedError(msg)
        if isinstance(v, str):
            text = v.strip()
            has_offset = text.endswith(("Z", "z")) or (
                len(text) > 10 and ("+" in text[10:] or "-" in text[10:])
            )
            nanos.append(parse_text(text, None))
            if has_offset:
                aware_seen = True
                if timezone is None:
                    timezone = _offset_name(text)
            else:
                naive_seen = True
        elif isinstance(v, dt.datetime):
            if v.tzinfo is None:
                naive_seen = True
            else:
                aware_seen = True
                if timezone is None:
                    timezone = _tz_name(v)
            nanos.append(_datetime_ns(v, None))
        elif isinstance(v, dt.date):
            naive_seen = True
            nanos.append(_datetime_ns(dt.datetime(v.year, v.month, v.day), None))
        elif isinstance(v, np.datetime64):
            naive_seen = True
            nanos.append(int(v.astype("datetime64[ns]").astype(np.int64)))
        else:
            msg = f"cannot use a {type(v).__name__} as a time (column {name!r})"
            raise TypeMismatchError(msg)
    if naive_seen and aware_seen:
        msg = "the time column mixes times with and without a UTC offset"
        raise InvalidArgumentError(msg)
    array = np.array(nanos, dtype=np.int64)
    return array, _coarsest_unit(array), timezone, naive_seen


def _coarsest_unit(nanos: np.ndarray) -> TimeUnit:
    # Like the reference CLI: milliseconds, or finer if needed.
    for unit in (TimeUnit.MILLISECOND, TimeUnit.MICROSECOND):
        if bool(np.all(nanos % unit.nanos == 0)):
            return unit
    return TimeUnit.NANOSECOND


def _offset_name(text: str) -> str:
    if text.endswith(("Z", "z")):
        return "UTC"
    tail = text[-6:] if text[-3] == ":" else text[-5:]
    return tail if ":" in tail else f"{tail[:3]}:{tail[3:]}"


def _tz_name(value: dt.datetime) -> str:
    zone = value.tzinfo
    key = getattr(zone, "key", None)
    if key:
        return str(key)
    if zone is dt.UTC:
        return "UTC"
    offset = value.utcoffset() or dt.timedelta(0)
    if offset == dt.timedelta(0):
        return "UTC"
    sign = "-" if offset < dt.timedelta(0) else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def _datetime64_time(values: np.ndarray) -> tuple[np.ndarray, TimeUnit]:
    unit_name = np.datetime_data(values.dtype)[0]
    if unit_name in _UNITS:
        return values.astype(np.int64), _UNITS[unit_name]
    if unit_name in ("D", "W", "M", "Y"):
        return values.astype("datetime64[ms]").astype(np.int64), TimeUnit.MILLISECOND
    return values.astype("datetime64[s]").astype(np.int64), TimeUnit.SECOND


# ---------------------------------------------------------------- mappings


def _from_mapping(data: Mapping[str, Any], time_column: str | None) -> SourceTable:
    data = dict(data)
    names = list(data)
    stamps = [n for n in names if isinstance(data[n], np.ndarray) and data[n].dtype.kind == "M"]
    time_name = time_column or guess_time_column(names, stamps)
    if time_name not in data:
        msg = f"the data has no column {time_name!r}"
        raise KeyError(msg)
    raw_time = data.pop(time_name)
    if isinstance(raw_time, Column):
        raw_time = raw_time.to_list()
    array = (
        raw_time if isinstance(raw_time, np.ndarray) else np.asarray(list(raw_time), dtype=object)
    )
    if array.dtype.kind == "M":
        time, unit = _datetime64_time(array)
        table = SourceTable(time_name, time, unit, None, True, [])
    elif array.dtype.kind in "iu" or (
        array.dtype == object
        and array.size
        and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in array.tolist()
        )
    ):
        table = SourceTable(time_name, array.astype(np.int64), None, None, False, [])
    elif array.size == 0:
        table = SourceTable(time_name, np.empty(0, dtype=np.int64), None, None, False, [])
    else:
        time, unit, tz, naive = _time_from_objects(time_name, array.tolist())
        table = SourceTable(time_name, time, TimeUnit.NANOSECOND, tz, naive, [], unit)
    for name, values in data.items():
        table.columns.append(_value_column(name, values))
    return table


def _value_column(name: str, values: Any) -> SourceColumn:
    if isinstance(values, Column):
        return SourceColumn(name, values.data_type, values.values, values.validity)
    if isinstance(values, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(values)
        inner = _value_column(name, np.asarray(values.filled(values.fill_value)))
        valid = ~mask if inner.valid is None else inner.valid & ~mask
        return SourceColumn(name, inner.data_type, inner.values, valid)
    if isinstance(values, np.ndarray) and values.dtype != object:
        data_type = _numpy_type(values.dtype)
        if data_type is None:
            msg = f"column {name!r} has dtype {values.dtype}, which Trosna cannot store"
            raise UnsupportedError(msg)
        if data_type is DataType.STRING:
            return _from_objects(name, values.tolist())
        return SourceColumn(name, data_type, values)
    return _from_objects(name, list(values))


# ---------------------------------------------------------------- pandas


def _from_pandas(frame: Any, time_column: str | None) -> SourceTable:
    pd = _require("pandas", "pandas")
    if time_column is None and isinstance(frame.index, pd.DatetimeIndex):
        name = frame.index.name or "time"
        frame = frame.rename_axis(name).reset_index()
        time_column = name
    names = [str(c) for c in frame.columns]
    if len(set(names)) != len(names):
        msg = "the DataFrame has duplicate column names"
        raise SchemaError(msg)
    stamps = [str(c) for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    time_name = time_column or guess_time_column(names, stamps)
    if time_name not in names:
        msg = f"the data has no column {time_name!r}"
        raise KeyError(msg)
    columns = []
    table: SourceTable | None = None
    for column_name in frame.columns:
        series = frame[column_name]
        name = str(column_name)
        if name == time_name:
            table = _pandas_time(pd, series, name)
        else:
            columns.append(_pandas_column(pd, series, name))
    assert table is not None  # noqa: S101
    table.columns = columns
    return table


def _pandas_time(pd: Any, series: Any, name: str) -> SourceTable:
    if series.isna().any():
        msg = "the time column contains nulls"
        raise UnsupportedError(msg)
    if pd.api.types.is_datetime64_any_dtype(series):
        tz = getattr(series.dtype, "tz", None)
        unit_name = np.datetime_data(series.dtype.base if tz is not None else series.dtype)[0]
        values = series.dt.tz_convert("UTC").dt.tz_localize(None) if tz is not None else series
        time, unit = _datetime64_time(values.to_numpy(dtype=f"datetime64[{unit_name}]"))
        timezone = None if tz is None else _pandas_tz_name(tz)
        return SourceTable(name, time, unit, timezone, tz is None, [])
    if pd.api.types.is_integer_dtype(series):
        return SourceTable(name, series.to_numpy(dtype=np.int64), None, None, False, [])
    time, unit, tz, naive = _time_from_objects(name, series.tolist())
    return SourceTable(name, time, TimeUnit.NANOSECOND, tz, naive, [], unit)


def _pandas_tz_name(tz: Any) -> str:
    key = getattr(tz, "key", None) or getattr(tz, "zone", None)
    if key:
        return str(key)
    probe = dt.datetime(2000, 1, 1, tzinfo=tz) if isinstance(tz, dt.tzinfo) else None
    if probe is not None:
        return _tz_name(probe)
    return str(tz)  # pragma: no cover - exotic time zone objects


def _pandas_column(pd: Any, series: Any, name: str) -> SourceColumn:
    mask = series.isna().to_numpy()
    valid = None if not mask.any() else ~mask
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _from_objects(name, series.astype(object).where(~mask, None).tolist())
    if pd.api.types.is_bool_dtype(dtype):
        return SourceColumn(
            name, DataType.BOOL, series.to_numpy(dtype=np.bool_, na_value=False), valid
        )
    if pd.api.types.is_integer_dtype(dtype):
        numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
        data_type = _numpy_type(np.dtype(numpy_dtype))
        values = series.to_numpy(dtype=numpy_dtype, na_value=0)
        return SourceColumn(name, data_type, values, valid)
    if pd.api.types.is_float_dtype(dtype):
        numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
        data_type = _numpy_type(np.dtype(numpy_dtype))
        values = series.to_numpy(dtype=numpy_dtype, na_value=0)
        if valid is not None:
            values = values.copy()
            values[mask] = 0
        return SourceColumn(name, data_type, values, valid)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        msg = f"column {name!r} holds time stamps; only one time column is supported"
        raise UnsupportedError(msg)
    objects = series.astype(object).where(~mask, None).tolist()
    return _from_objects(name, objects)


# ---------------------------------------------------------------- Arrow


def _arrow_type(pa: Any, arrow_type: Any) -> DataType | None:
    types = pa.types
    if types.is_boolean(arrow_type):
        return DataType.BOOL
    if types.is_int8(arrow_type) or types.is_int16(arrow_type) or types.is_int32(arrow_type):
        return DataType.INT32
    if types.is_uint8(arrow_type) or types.is_uint16(arrow_type):
        return DataType.INT32
    if types.is_int64(arrow_type) or types.is_uint32(arrow_type) or types.is_uint64(arrow_type):
        return DataType.INT64
    if types.is_float16(arrow_type) or types.is_float32(arrow_type):
        return DataType.FLOAT32
    if types.is_float64(arrow_type):
        return DataType.FLOAT64
    if types.is_string(arrow_type) or types.is_large_string(arrow_type):
        return DataType.STRING
    if hasattr(types, "is_string_view") and types.is_string_view(arrow_type):
        return DataType.STRING
    if (
        types.is_dictionary(arrow_type)
        and _arrow_type(pa, arrow_type.value_type) is DataType.STRING
    ):
        return DataType.STRING
    return None


def _from_arrow(table: Any, time_column: str | None) -> SourceTable:
    pa = _require("pyarrow", "arrow")
    schema = table.schema
    names = list(schema.names)
    stamps = [f.name for f in schema if pa.types.is_timestamp(f.type)]
    time_name = time_column or guess_time_column(names, stamps)
    if time_name not in names:
        msg = f"the data has no column {time_name!r}"
        raise KeyError(msg)
    source: SourceTable | None = None
    columns = []
    for f in schema:
        array = table.column(f.name)
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks() if array.num_chunks else pa.array([], type=f.type)
        if f.name == time_name:
            source = _arrow_time(pa, array, f.name)
            continue
        metadata = {k.decode(): v.decode() for k, v in (f.metadata or {}).items()}
        columns.append(_arrow_column(pa, array, f.name, metadata))
    assert source is not None  # noqa: S101
    source.columns = columns
    return source


def _arrow_time(pa: Any, array: Any, name: str) -> SourceTable:
    if array.null_count:
        msg = "the time column contains nulls"
        raise UnsupportedError(msg)
    arrow_type = array.type
    if pa.types.is_timestamp(arrow_type):
        values = array.cast(pa.int64()).to_numpy(zero_copy_only=False)
        tz = arrow_type.tz
        return SourceTable(
            name, values.astype(np.int64), _UNITS[arrow_type.unit], tz, tz is None, []
        )
    if pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
        values = array.cast(pa.date64()).cast(pa.int64()).to_numpy(zero_copy_only=False)
        return SourceTable(name, values.astype(np.int64), TimeUnit.MILLISECOND, None, True, [])
    if pa.types.is_integer(arrow_type):
        values = array.cast(pa.int64()).to_numpy(zero_copy_only=False)
        return SourceTable(name, values.astype(np.int64), None, None, False, [])
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        time, unit, tz, naive = _time_from_objects(name, array.to_pylist())
        return SourceTable(name, time, TimeUnit.NANOSECOND, tz, naive, [], unit)
    msg = f"cannot use a column of type {arrow_type} as time"
    raise UnsupportedError(msg)


def _arrow_column(pa: Any, array: Any, name: str, metadata: dict[str, str]) -> SourceColumn:
    data_type = _arrow_type(pa, array.type)
    valid = None
    if array.null_count:
        valid = array.is_valid().to_numpy(zero_copy_only=False)
    if pa.types.is_null(array.type):
        return SourceColumn(name, None, np.zeros(len(array)), np.zeros(len(array), dtype=np.bool_))
    if data_type is None:
        msg = f"column {name!r} has Arrow type {array.type}, which Trosna cannot store"
        raise UnsupportedError(msg)
    if data_type is DataType.STRING:
        strings = array.cast(pa.string()).to_pylist()
        values = np.empty(len(strings), dtype=object)
        values[:] = ["" if s is None else s for s in strings]
        return SourceColumn(name, data_type, values, valid, metadata)
    if pa.types.is_float16(array.type):
        array = array.cast(pa.float32())
    filled = (
        array.fill_null(False if data_type is DataType.BOOL else 0) if array.null_count else array
    )
    values = filled.to_numpy(zero_copy_only=False)
    return SourceColumn(name, data_type, values, valid, metadata)


# ---------------------------------------------------------------- Polars


def _polars_type(pl: Any, dtype: Any) -> DataType | None:
    if dtype == pl.Boolean:
        return DataType.BOOL
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16):
        return DataType.INT32
    if dtype in (pl.Int64, pl.UInt32, pl.UInt64):
        return DataType.INT64
    if dtype == pl.Float32:
        return DataType.FLOAT32
    if dtype == pl.Float64:
        return DataType.FLOAT64
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return DataType.STRING
    return None


def _from_polars(frame: Any, time_column: str | None) -> SourceTable:
    pl = _require("polars", "polars")
    names = list(frame.columns)
    stamps = [n for n in names if isinstance(frame.schema[n], pl.Datetime)]
    time_name = time_column or guess_time_column(names, stamps)
    if time_name not in names:
        msg = f"the data has no column {time_name!r}"
        raise KeyError(msg)
    source: SourceTable | None = None
    columns = []
    for name in names:
        series = frame.get_column(name)
        if name == time_name:
            source = _polars_time(pl, series, name)
        else:
            columns.append(_polars_column(pl, series, name))
    assert source is not None  # noqa: S101
    source.columns = columns
    return source


def _polars_time(pl: Any, series: Any, name: str) -> SourceTable:
    if series.null_count():
        msg = "the time column contains nulls"
        raise UnsupportedError(msg)
    dtype = series.dtype
    if isinstance(dtype, pl.Datetime):
        unit = _UNITS[dtype.time_unit]
        values = series.dt.epoch(dtype.time_unit).to_numpy().astype(np.int64)
        tz = dtype.time_zone
        return SourceTable(name, values, unit, tz, tz is None, [])
    if dtype == pl.Date:
        values = series.dt.epoch("ms").to_numpy().astype(np.int64)
        return SourceTable(name, values, TimeUnit.MILLISECOND, None, True, [])
    if dtype.is_integer():
        return SourceTable(name, series.cast(pl.Int64).to_numpy(), None, None, False, [])
    if dtype == pl.String:
        time, unit, tz, naive = _time_from_objects(name, series.to_list())
        return SourceTable(name, time, TimeUnit.NANOSECOND, tz, naive, [], unit)
    msg = f"cannot use a column of type {dtype} as time"
    raise UnsupportedError(msg)


def _polars_column(pl: Any, series: Any, name: str) -> SourceColumn:
    if series.dtype == pl.Null:
        n = len(series)
        return SourceColumn(name, None, np.zeros(n), np.zeros(n, dtype=np.bool_))
    data_type = _polars_type(pl, series.dtype)
    if data_type is None:
        msg = f"column {name!r} has Polars type {series.dtype}, which Trosna cannot store"
        raise UnsupportedError(msg)
    valid = None
    if series.null_count():
        valid = series.is_not_null().to_numpy()
    if data_type is DataType.STRING:
        strings = series.cast(pl.String).to_list()
        values = np.empty(len(strings), dtype=object)
        values[:] = ["" if s is None else s for s in strings]
        return SourceColumn(name, data_type, values, valid)
    fill = False if data_type is DataType.BOOL else 0
    values = series.fill_null(fill).to_numpy() if valid is not None else series.to_numpy()
    return SourceColumn(name, data_type, values, valid)


# ---------------------------------------------------------------- dispatch


def _from_batch(batch: Batch, time_column: str | None) -> SourceTable:
    schema = batch.schema
    name = time_column or (schema.time_name if schema else "time")
    unit = schema.time_unit if schema else None
    tz = schema.timezone if schema else None
    columns = [
        SourceColumn(n, c.data_type, c.values, c.validity, _column_metadata(batch, n))
        for n, c in batch.columns.items()
    ]
    return SourceTable(name, batch.time, unit, tz, False, columns)


def _arrow_stream(pa: Any, reader: Any, time_column: str | None) -> Iterator[SourceTable]:
    empty = True
    for batch in reader:
        empty = False
        yield _from_arrow(batch, time_column)
    if empty:
        # No rows at all: the schema still describes the device.
        yield _from_arrow(reader.schema.empty_table(), time_column)


def normalize(data: Any, time_column: str | None = None) -> Iterator[SourceTable]:
    """Turns supported input data into :class:`SourceTable` objects (one per
    record batch for Arrow streams, otherwise one)."""
    module = type(data).__module__.split(".")[0]
    if isinstance(data, Batch):
        yield _from_batch(data, time_column)
    elif module == "pandas":
        yield _from_pandas(data, time_column)
    elif module == "polars":
        if type(data).__name__ == "LazyFrame":
            data = data.collect()
        yield _from_polars(data, time_column)
    elif module == "pyarrow":
        pa = _require("pyarrow", "arrow")
        if isinstance(data, pa.RecordBatchReader):
            yield from _arrow_stream(pa, data, time_column)
        else:
            yield _from_arrow(data, time_column)
    elif isinstance(data, Mapping):
        yield _from_mapping(data, time_column)
    elif hasattr(data, "__arrow_c_stream__"):
        pa = _require("pyarrow", "arrow")
        yield from _arrow_stream(pa, pa.RecordBatchReader.from_stream(data), time_column)
    else:
        msg = (
            f"cannot write a {type(data).__name__}; give a pandas or Polars DataFrame, "
            "a PyArrow Table/RecordBatch/RecordBatchReader, a pytrosna.Batch or a "
            "mapping of column names to values"
        )
        raise TypeError(msg)


def infer_device(
    name: str, source: SourceTable, unit: TimeUnit | str | None = None
) -> DeviceSchema:
    """Derives a device schema from input data. Integer time columns need ``unit``."""
    if source.time_unit is None:
        if unit is None:
            msg = "an integer time column needs an explicit time unit (unit='ms', …)"
            raise UnsupportedError(msg)
        time_unit = TimeUnit.parse(unit)
    else:
        time_unit = source.device_unit or source.time_unit
    columns = []
    for column in source.columns:
        if column.data_type is None:
            msg = (
                f"column {column.name!r} contains only nulls, so its type is unknown; "
                "create the device with an explicit schema first"
            )
            raise UnsupportedError(msg)
        columns.append(ColumnSchema(column.name, column.data_type, column.metadata))
    schema = DeviceSchema(name, time_unit, tuple(columns), source.timezone, source.time_name)
    schema.validate()
    return schema


def _convert_unit(values: np.ndarray, source: TimeUnit, target: TimeUnit) -> np.ndarray:
    """Converts time stamps between units; going to a coarser unit must be exact."""
    if source == target:
        return values
    if target > source:
        factor = target.per_second // source.per_second
        limit = (1 << 63) // factor
        if values.size and (values.max() >= limit or values.min() < -limit):
            msg = f"time stamps in {source} overflow in {target}"
            raise UnsupportedError(msg)
        return values * factor
    factor = source.per_second // target.per_second
    if bool(np.any(values % factor)):
        bad = int(values[np.flatnonzero(values % factor)[0]])
        msg = f"time stamp {bad} {source} is not a whole number of {target}"
        raise UnsupportedError(msg)
    return values // factor


def _localize(nanos: np.ndarray, tz: str) -> np.ndarray:
    """Interprets naive nanoseconds as wall-clock times of ``tz`` (the earlier
    moment in a repeated hour; an error for skipped times)."""
    zone = tzinfo(tz)
    if isinstance(zone, dt.timezone):
        offset = zone.utcoffset(None)
        return nanos - int(offset.total_seconds()) * 1_000_000_000
    minute = 60 * 1_000_000_000
    starts = nanos // minute * minute
    unique, inverse = np.unique(starts, return_inverse=True)
    epoch = dt.datetime(1970, 1, 1)
    shifted = np.array(
        [
            _datetime_ns(epoch + dt.timedelta(microseconds=int(s) // 1000), tz)
            for s in unique.tolist()
        ],
        dtype=np.int64,
    )
    return shifted[inverse] + (nanos - starts)


def _source_time(
    source: SourceTable, schema: DeviceSchema, unit: TimeUnit | str | None
) -> np.ndarray:
    if source.time_unit is None:
        int_unit = TimeUnit.parse(unit) if unit is not None else schema.time_unit
        return _convert_unit(source.time, int_unit, schema.time_unit)
    time, time_unit = source.time, source.time_unit
    if source.naive and schema.timezone is not None:
        time = _localize(_convert_unit(time, time_unit, TimeUnit.NANOSECOND), schema.timezone)
        time_unit = TimeUnit.NANOSECOND
    return _convert_unit(time, time_unit, schema.time_unit)


def _cast_column(column: SourceColumn, target: DataType) -> Column:
    n = column.values.size
    if column.data_type is None:
        return Column.nulls(target, n)
    source = column.data_type
    values = column.values
    valid = column.valid
    if (
        target is DataType.BOOL
        or target is DataType.STRING
        or source in (DataType.BOOL, DataType.STRING)
    ):
        if source is not target:
            msg = (
                f"column {column.name!r} has type {target}, so values of type {source} "
                "cannot be stored in it"
            )
            raise TypeMismatchError(msg)
        if target is DataType.BOOL:
            return Column(target, values.astype(np.bool_), valid)
        if values.dtype != object:
            values = values.astype(object)
        return Column(target, values, valid)
    if target in (DataType.INT32, DataType.INT64):
        dense = values if valid is None else values[valid]
        if values.dtype.kind == "f" and not bool(
            np.all(np.isfinite(dense) & (dense == np.round(dense)))
        ):
            bad = dense[~(np.isfinite(dense) & (dense == np.round(dense)))][0]
            msg = (
                f"column {column.name!r} has type {target}, "
                f"but the value {bad} is not a whole number"
            )
            raise TypeMismatchError(msg)
        info = np.iinfo(target.numpy_dtype)
        if dense.size and (dense.min() < info.min or dense.max() > info.max):
            msg = f"column {column.name!r} has values outside the {target} range"
            raise TypeMismatchError(msg)
        clean = values if valid is None else np.where(valid, values, 0)
        return Column(target, clean.astype(target.numpy_dtype), valid)
    return Column(target, values.astype(target.numpy_dtype), valid)


def to_batch(
    source: SourceTable, schema: DeviceSchema, unit: TimeUnit | str | None = None
) -> Batch:
    """Converts input data into rows of ``schema``: the time stamps are
    converted to the device's unit (integer time stamps are read in ``unit``,
    by default the device's unit; naive times are wall-clock times of the
    device's time zone) and every value column to the type of the device
    column of the same name."""
    time = _source_time(source, schema, unit)
    columns: dict[str, Column] = {}
    for column in source.columns:
        index = schema.column_index(column.name)
        if index is None:
            from .errors import UnknownColumnError

            raise UnknownColumnError(schema.name, column.name)
        columns[column.name] = _cast_column(column, schema.columns[index].data_type)
    return Batch(time, columns, schema)


def iter_source(data: Iterable[Any], time_column: str | None = None) -> Iterator[SourceTable]:
    """:func:`normalize` over several inputs."""
    for item in data:
        yield from normalize(item, time_column)
