"""Adapters to pandas, Polars and PyArrow."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import pytrosna
from pytrosna import Batch, Column, DataType, DeviceSchema, TimeUnit, adapters
from pytrosna.errors import (
    InvalidArgumentError,
    SchemaError,
    TypeMismatchError,
    UnknownColumnError,
    UnsupportedError,
)

from .conftest import SCHEMA, T0


def sample_batch() -> Batch:
    return Batch(
        np.array([T0, T0 + 1000, T0 + 2000]),
        {
            "cpu": Column.from_values("float64", [0.5, None, float("nan")]),
            "ram_mb": Column.from_values("int64", [1, None, 3]),
            "swap": Column.from_values("int32", [1, 2, 3]),
            "ok": Column.from_values("bool", [True, None, False]),
            "state": Column.from_values("string", ["a", None, "в"]),
            "temp": Column.from_values("float32", [1.5, 2.5, None]),
        },
        SCHEMA,
    )


# ---------------------------------------------------------------- reading


def test_to_arrow() -> None:
    table = sample_batch().to_arrow()
    assert table.schema.field("time").type == pa.timestamp("ms", "UTC")
    assert not table.schema.field("time").nullable
    assert table.schema.metadata[b"trosna.device"] == b"vm01"
    assert table.column("ram_mb").to_pylist() == [1, None, 3]
    assert table.column("ok").to_pylist() == [True, None, False]
    assert table.column("state").to_pylist() == ["a", None, "в"]
    assert table.column("temp").type == pa.float32()
    cpu = table.column("cpu").to_pylist()
    assert cpu[0] == 0.5
    assert cpu[1] is None
    assert np.isnan(cpu[2])


def test_column_metadata_reaches_arrow() -> None:
    schema = DeviceSchema("d", "s", (pytrosna.ColumnSchema("v", DataType.INT64, {"unit": "MB"}),))
    batch = Batch([1], {"v": Column.from_values("int64", [1])}, schema)
    assert batch.to_arrow().schema.field("v").metadata == {b"unit": b"MB"}


def test_to_pandas() -> None:
    df = sample_batch().to_pandas()
    assert list(df.columns) == ["time", *SCHEMA.column_names]
    assert str(df["time"].dtype) == "datetime64[ms, UTC]"
    assert df["ram_mb"].dtype == np.float64
    assert np.isnan(df["ram_mb"][1])
    assert df["swap"].dtype == np.int32
    assert df["ok"].tolist() == [True, None, False]
    assert df["temp"].dtype == np.float32
    nullable = sample_batch().to_pandas(dtype_backend="numpy_nullable")
    assert str(nullable["ram_mb"].dtype) == "Int64"
    assert nullable["ram_mb"].isna().tolist() == [False, True, False]
    assert str(nullable["ok"].dtype) == "boolean"
    assert str(nullable["state"].dtype).startswith("string")
    with pytest.raises(InvalidArgumentError):
        sample_batch().to_pandas(dtype_backend="pyarrow")


def test_to_polars() -> None:
    df = sample_batch().to_polars()
    assert df.schema["time"] == pl.Datetime("ms", "UTC")
    assert df["ram_mb"].to_list() == [1, None, 3]
    assert df["ok"].to_list() == [True, None, False]
    assert df["state"].to_list() == ["a", None, "в"]
    assert df.schema["swap"] == pl.Int32


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        ("+03:00", "Etc/GMT-3"),
        ("-05:00", "Etc/GMT+5"),
        ("+05:30", "UTC"),
        ("+00:00", "UTC"),
        ("Z", "UTC"),
        ("Europe/Moscow", "Europe/Moscow"),
        (None, None),
    ],
)
def test_polars_time_zones(tz: str | None, expected: str | None) -> None:
    assert adapters._polars_timezone(tz) == expected


def test_polars_has_no_seconds() -> None:
    schema = DeviceSchema.build("d", "s", {"v": "int64"}, timezone="+03:00")
    df = Batch([1], {"v": Column.from_values("int64", [1])}, schema).to_polars()
    assert df.schema["time"] == pl.Datetime("ms", "Etc/GMT-3")
    assert df["time"].dt.epoch("ms").to_list() == [1000]


def test_batches_without_a_schema() -> None:
    batch = Batch([5], {"v": Column.from_values("int64", [1])})
    assert batch.to_arrow().schema.field("time").type == pa.timestamp("ns")
    assert batch.to_pandas()["time"].dtype == np.dtype("datetime64[ns]")
    assert batch.to_polars().schema["time"] == pl.Datetime("ns")


# ---------------------------------------------------------------- writing


def source(data: object, **kw: object) -> adapters.SourceTable:
    tables = list(adapters.normalize(data, **kw))  # type: ignore[arg-type]
    assert len(tables) == 1
    return tables[0]


def test_pandas_input() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-10-08 10:00", periods=3, freq="s", tz="Europe/Moscow"),
            "i8": np.array([1, 2, 3], dtype=np.int8),
            "u32": np.array([1, 2, 3], dtype=np.uint32),
            "f": [1.0, np.nan, 3.0],
            "f32": np.array([1, 2, 3], dtype=np.float32),
            "b": [True, False, True],
            "nb": pd.array([True, None, False], dtype="boolean"),
            "ni": pd.array([1, None, 3], dtype="Int64"),
            "s": ["a", None, "c"],
            "cat": pd.Categorical(["x", "y", "x"]),
            "ns": pd.array(["a", pd.NA, "b"], dtype="string"),
        }
    )
    src = source(df)
    schema = adapters.infer_device("d", src)
    assert schema.time_unit is TimeUnit.parse(np.datetime_data(df["time"].dtype.base)[0])
    assert schema.timezone == "Europe/Moscow"
    types = dict(zip(schema.column_names, schema.types, strict=True))
    assert types == {
        "i8": DataType.INT32,
        "u32": DataType.INT64,
        "f": DataType.FLOAT64,
        "f32": DataType.FLOAT32,
        "b": DataType.BOOL,
        "nb": DataType.BOOL,
        "ni": DataType.INT64,
        "s": DataType.STRING,
        "cat": DataType.STRING,
        "ns": DataType.STRING,
    }
    batch = adapters.to_batch(src, schema)
    assert batch["f"].to_list() == [1.0, None, 3.0]  # NaN in pandas is a missing value
    assert batch["ni"].to_list() == [1, None, 3]
    assert batch["nb"].to_list() == [True, None, False]
    assert batch["ns"].to_list() == ["a", None, "b"]
    assert batch["cat"].to_list() == ["x", "y", "x"]


def test_pandas_datetime_index_and_naive_times() -> None:
    df = pd.DataFrame(
        {"v": [1, 2]}, index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], name="ts")
    )
    src = source(df)
    assert src.time_name == "ts"
    assert src.naive
    schema = DeviceSchema.build("d", "s", {"v": "int64"}, timezone="+02:00")
    batch = adapters.to_batch(src, schema)
    assert batch.time.tolist() == [1767218400, 1767304800]  # wall clock at +02:00


def test_pandas_errors() -> None:
    with pytest.raises(UnsupportedError, match="nulls"):
        source(pd.DataFrame({"time": pd.to_datetime(["2026-01-01", None]), "v": [1, 2]}))
    with pytest.raises(UnsupportedError, match="only one time column"):
        source(
            pd.DataFrame(
                {"time": pd.to_datetime(["2026-01-01"]), "t2": pd.to_datetime(["2026-01-01"])}
            )
        )
    with pytest.raises(UnsupportedError, match="mixes"):
        source(pd.DataFrame({"time": [1, 2], "v": ["a", 1]}))
    with pytest.raises(UnsupportedError, match="cannot store"):
        source(pd.DataFrame({"time": [1], "v": [b"bytes"]}))
    with pytest.raises(InvalidArgumentError, match="time_column"):
        source(pd.DataFrame({"x": [1], "v": [1]}))
    with pytest.raises(KeyError):
        source(pd.DataFrame({"time": [1]}), time_column="when")
    frame = pd.DataFrame([[1, 2]], columns=["time", "time"])
    with pytest.raises(SchemaError, match="duplicate"):
        source(frame)


def test_pandas_text_and_integer_times() -> None:
    text = source(pd.DataFrame({"time": ["2026-10-08T10:00:00.5+03:00"], "v": [1]}))
    assert text.timezone == "+03:00"
    assert text.device_unit is TimeUnit.MILLISECOND
    ints = source(pd.DataFrame({"time": [1, 2], "v": [1, 2]}))
    assert ints.time_unit is None
    with pytest.raises(UnsupportedError, match="explicit time unit"):
        adapters.infer_device("d", ints)
    assert adapters.infer_device("d", ints, "us").time_unit is TimeUnit.MICROSECOND


def test_arrow_input() -> None:
    table = pa.table(
        {
            "time": pa.array([1, 2], type=pa.timestamp("us", "UTC")),
            "u8": pa.array([1, None], type=pa.uint8()),
            "f16": pa.array(np.array([1.5, 2.5], dtype=np.float16)),
            "large": pa.array(["a", "b"], type=pa.large_string()),
            "dict": pa.array(["x", "y"]).dictionary_encode(),
            "nulls": pa.nulls(2),
            "flag": pa.array([True, None]),
        }
    )
    src = source(table)
    types = {c.name: c.data_type for c in src.columns}
    assert types == {
        "u8": DataType.INT32,
        "f16": DataType.FLOAT32,
        "large": DataType.STRING,
        "dict": DataType.STRING,
        "nulls": None,
        "flag": DataType.BOOL,
    }
    with pytest.raises(UnsupportedError, match="only nulls"):
        adapters.infer_device("d", src)
    schema = DeviceSchema.build(
        "d",
        "us",
        {
            "u8": "int32",
            "f16": "float64",
            "large": "string",
            "dict": "string",
            "nulls": "int64",
            "flag": "bool",
        },
    )
    batch = adapters.to_batch(src, schema)
    assert batch["u8"].to_list() == [1, None]
    assert batch["nulls"].to_list() == [None, None]
    assert batch["flag"].to_list() == [True, None]
    assert batch["f16"].to_list() == [1.5, 2.5]
    one = source(table.to_batches()[0])
    assert len(one) == 2


def test_arrow_time_columns() -> None:
    dates = source(pa.table({"time": pa.array([dt.date(1970, 1, 2)]), "v": [1]}))
    assert dates.time.tolist() == [86_400_000]
    ints = source(pa.table({"time": pa.array([5], type=pa.int32()), "v": [1]}))
    assert ints.time_unit is None
    text = source(pa.table({"time": ["1970-01-01T00:00:00.000001Z"], "v": [1]}))
    assert text.device_unit is TimeUnit.MICROSECOND
    with pytest.raises(UnsupportedError, match="as time"):
        source(pa.table({"time": [1.5], "v": [1]}))
    with pytest.raises(UnsupportedError, match="nulls"):
        source(pa.table({"time": pa.array([1, None], type=pa.timestamp("s")), "v": [1, 2]}))
    with pytest.raises(UnsupportedError, match="cannot store"):
        source(pa.table({"time": pa.array([1], type=pa.timestamp("s")), "v": pa.array([[1]])}))


def test_arrow_streams() -> None:
    table = pa.table({"time": pa.array([1, 2, 3], type=pa.timestamp("s")), "v": [1, 2, 3]})
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=1))
    assert [len(s) for s in adapters.normalize(reader)] == [1, 1, 1]
    empty = pa.RecordBatchReader.from_batches(table.schema, [])
    assert [len(s) for s in adapters.normalize(empty)] == [0]

    class Exporter:
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            return table.__arrow_c_stream__(requested_schema)

    assert sum(len(s) for s in adapters.normalize(Exporter())) == 3


def test_polars_input() -> None:
    df = pl.DataFrame(
        {
            "time": pl.Series([0, 1000], dtype=pl.Int64).cast(pl.Datetime("ms", "Europe/Moscow")),
            "i16": pl.Series([1, None], dtype=pl.Int16),
            "u64": pl.Series([1, 2], dtype=pl.UInt64),
            "f": [1.0, float("nan")],
            "f32": pl.Series([1.0, 2.0], dtype=pl.Float32),
            "b": [True, None],
            "s": ["a", None],
            "cat": pl.Series(["x", "y"], dtype=pl.Categorical),
            "nothing": pl.Series([None, None], dtype=pl.Null),
        }
    )
    src = source(df)
    assert src.timezone == "Europe/Moscow"
    assert src.time_unit is TimeUnit.MILLISECOND
    types = {c.name: c.data_type for c in src.columns}
    assert types["i16"] is DataType.INT32
    assert types["u64"] is DataType.INT64
    assert types["cat"] is DataType.STRING
    assert types["nothing"] is None
    schema = DeviceSchema.build(
        "d",
        "ms",
        {
            "i16": "int32",
            "u64": "int64",
            "f": "float64",
            "f32": "float32",
            "b": "bool",
            "s": "string",
            "cat": "string",
            "nothing": "bool",
        },
    )
    batch = adapters.to_batch(src, schema)
    assert batch["i16"].to_list() == [1, None]
    f = batch["f"].to_list()
    assert f[0] == 1.0
    assert np.isnan(f[1])  # NaN is a value in Polars
    assert batch["s"].to_list() == ["a", None]
    assert batch["nothing"].to_list() == [None, None]
    lazy = source(df.lazy())
    assert len(lazy) == 2


def test_polars_time_columns() -> None:
    dates = source(pl.DataFrame({"time": [dt.date(1970, 1, 2)], "v": [1]}))
    assert dates.time.tolist() == [86_400_000]
    ints = source(pl.DataFrame({"time": [5], "v": [1]}))
    assert ints.time_unit is None
    text = source(pl.DataFrame({"time": ["1970-01-01 00:00:01"], "v": [1]}))
    assert text.naive
    with pytest.raises(UnsupportedError, match="as time"):
        source(pl.DataFrame({"time": [1.5], "v": [1]}))
    with pytest.raises(UnsupportedError, match="nulls"):
        source(pl.DataFrame({"time": [1, None], "v": [1, 2]}))
    with pytest.raises(UnsupportedError, match="cannot store"):
        source(pl.DataFrame({"time": [1], "v": [[1, 2]]}))


def test_mapping_input() -> None:
    src = source(
        {
            "when": np.array(["2026-01-01T00:00:00", "2026-01-01T00:00:01"], dtype="datetime64[s]"),
            "v": [1, None],
            "f": np.array([1.0, 2.0]),
            "s": np.array(["a", "b"]),
            "m": np.ma.masked_array([1, 2], mask=[False, True]),
            "c": Column.from_values("bool", [True, False]),
            "mixed": [1, 2.5],
        }
    )
    assert src.time_name == "when"
    assert src.time_unit is TimeUnit.SECOND
    types = {c.name: c.data_type for c in src.columns}
    assert types == {
        "v": DataType.INT64,
        "f": DataType.FLOAT64,
        "s": DataType.STRING,
        "m": DataType.INT64,
        "c": DataType.BOOL,
        "mixed": DataType.FLOAT64,
    }
    m = next(c for c in src.columns if c.name == "m")
    assert m.valid is not None
    assert m.valid.tolist() == [True, False]


def test_mapping_time_forms() -> None:
    aware = source({"time": [dt.datetime(2026, 1, 1, tzinfo=dt.UTC)], "v": [1]})
    assert aware.timezone == "UTC"
    moscow = source(
        {"time": [dt.datetime(2026, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=3)))], "v": [1]}
    )
    assert moscow.timezone == "+03:00"
    zone = source(
        {
            "time": [
                dt.datetime(2026, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=-3, minutes=-30)))
            ],
            "v": [1],
        }
    )
    assert zone.timezone == "-03:30"
    dates = source({"time": [dt.date(1970, 1, 2)], "v": [1]})
    assert dates.time.tolist() == [86_400 * 10**9]
    numpy_objects = source({"time": [np.datetime64(5, "ns")], "v": [1]})
    assert numpy_objects.device_unit is TimeUnit.NANOSECOND
    ints = source({"time": [1, 2], "v": [1, 2]})
    assert ints.time_unit is None
    column = source({"time": Column.from_values("int64", [1]), "v": [1]})
    assert column.time.tolist() == [1]
    empty = source({"time": [], "v": []})
    assert len(empty) == 0
    with pytest.raises(InvalidArgumentError, match="mixes"):
        source({"time": ["2026-01-01T00:00:00Z", "2026-01-01 00:00:00"], "v": [1, 2]})
    with pytest.raises(TypeMismatchError):
        source({"time": [1.5], "v": [1]})
    with pytest.raises(UnsupportedError, match="nulls"):
        source({"time": ["2026-01-01", None], "v": [1, 2]})
    with pytest.raises(KeyError):
        source({"time": [1]}, time_column="x")
    with pytest.raises(UnsupportedError):
        source({"time": [1], "v": np.array([1 + 2j])})
    with pytest.raises(UnsupportedError):
        source({"time": [1], "v": [object()]})


def test_unsupported_inputs() -> None:
    with pytest.raises(TypeError, match="cannot write"):
        list(adapters.normalize(42))


def test_batch_input_round_trip() -> None:
    batch = sample_batch()
    src = source(batch)
    assert adapters.infer_device("vm01", src).same_layout(SCHEMA)
    assert adapters.to_batch(src, SCHEMA) == batch


def test_casting_rules() -> None:
    schema = DeviceSchema.build(
        "d", "s", {"i": "int32", "f": "float32", "b": "bool", "s": "string"}
    )
    ok = source({"time": [1, 2], "i": [1.0, 2.0], "f": [1, 2], "b": [True, False], "s": ["x", "y"]})
    batch = adapters.to_batch(ok, schema)
    assert batch["i"].to_list() == [1, 2]
    assert batch["f"].to_list() == [1.0, 2.0]
    with pytest.raises(TypeMismatchError, match="whole number"):
        adapters.to_batch(source({"time": [1], "i": [1.5]}), schema)
    with pytest.raises(TypeMismatchError, match="range"):
        adapters.to_batch(source({"time": [1], "i": [2**40]}), schema)
    with pytest.raises(TypeMismatchError, match="cannot be stored"):
        adapters.to_batch(source({"time": [1], "b": [1]}), schema)
    with pytest.raises(TypeMismatchError, match="cannot be stored"):
        adapters.to_batch(source({"time": [1], "s": [1]}), schema)
    with pytest.raises(UnknownColumnError):
        adapters.to_batch(source({"time": [1], "zzz": [1]}), schema)
    with_nulls = adapters.to_batch(source({"time": [1, 2], "i": [1.0, None]}), schema)
    assert with_nulls["i"].to_list() == [1, None]


def test_time_unit_conversion() -> None:
    schema = DeviceSchema.build("d", "s", {"v": "int64"})
    ms = source(pa.table({"time": pa.array([2000, 3000], type=pa.timestamp("ms")), "v": [1, 2]}))
    assert adapters.to_batch(ms, schema).time.tolist() == [2, 3]
    bad = source(pa.table({"time": pa.array([2500], type=pa.timestamp("ms")), "v": [1]}))
    with pytest.raises(UnsupportedError, match="whole number"):
        adapters.to_batch(bad, schema)
    fine = DeviceSchema.build("d", "ns", {"v": "int64"})
    seconds = source(pa.table({"time": pa.array([2**62], type=pa.timestamp("s")), "v": [1]}))
    with pytest.raises(UnsupportedError, match="overflow"):
        adapters.to_batch(seconds, fine)
    with pytest.raises(UnsupportedError, match="whole number"):
        adapters.to_batch(source({"time": [5], "v": [1]}), schema, unit="ms")
    assert adapters.to_batch(
        source({"time": [5000], "v": [1]}), schema, unit="ms"
    ).time.tolist() == [5]


def test_localization_of_naive_times() -> None:
    schema = DeviceSchema.build("d", "s", {"v": "int64"}, timezone="Europe/Berlin")
    naive = source(pa.table({"time": pa.array([0, 3600], type=pa.timestamp("s")), "v": [1, 2]}))
    assert adapters.to_batch(naive, schema).time.tolist() == [-3600, 0]
    gap = source({"time": ["2026-03-29 02:30"], "v": [1]})
    with pytest.raises(InvalidArgumentError, match="does not exist"):
        adapters.to_batch(gap, schema)


def test_guess_time_column() -> None:
    assert adapters.guess_time_column(["a", "Время"]) == "Время"
    assert adapters.guess_time_column(["x", "Timestamp", "time"]) == "time"
    assert adapters.guess_time_column(["a"], ["a"]) == "a"


def test_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name == "polars":
            raise ImportError(name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(ImportError, match=r"pytrosna\[polars\]"):
        sample_batch().to_polars()


def test_write_and_read_with_every_library(path: Path) -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "v": [1.5, 2.5, None, 4.0],
        }
    )
    pytrosna.write(path, df, "d")
    assert pytrosna.read_pandas(path)["v"].tolist()[:2] == [1.5, 2.5]
    pytrosna.write(path, pl.from_pandas(df), "d", mode="a")
    pytrosna.write(path, pa.Table.from_pandas(df, preserve_index=False), "d", mode="a")
    assert pytrosna.read_polars(path)["v"].null_count() == 1
    assert pytrosna.read_arrow(path).num_rows == 4
