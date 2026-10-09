"""Schema types."""

from __future__ import annotations

import numpy as np
import pytest

from pytrosna import ColumnSchema, DataType, DeviceSchema, TimeUnit
from pytrosna.errors import InvalidArgumentError, SchemaError, UnsupportedError
from pytrosna.types import MAX_COLUMNS


def test_codes_and_names_round_trip() -> None:
    for t in DataType:
        assert DataType.from_code(int(t)) is t
        assert DataType.parse(t.label) is t
        assert str(t) == t.label
    for u in TimeUnit:
        assert TimeUnit.from_code(int(u)) is u
        assert TimeUnit.parse(u.label) is u
        assert str(u) == u.label
    with pytest.raises(UnsupportedError):
        DataType.from_code(6)
    with pytest.raises(UnsupportedError):
        TimeUnit.from_code(4)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("boolean", DataType.BOOL),
        ("i32", DataType.INT32),
        ("long", DataType.INT64),
        ("float", DataType.FLOAT32),
        ("double", DataType.FLOAT64),
        ("str", DataType.STRING),
        ("UTF8", DataType.STRING),
    ],
)
def test_type_aliases(alias: str, expected: DataType) -> None:
    assert DataType.parse(alias) is expected


def test_unknown_names() -> None:
    with pytest.raises(InvalidArgumentError):
        DataType.parse("decimal")
    with pytest.raises(InvalidArgumentError):
        DataType.parse(5)
    with pytest.raises(InvalidArgumentError):
        TimeUnit.parse("minutes")


def test_type_properties() -> None:
    assert DataType.FLOAT32.numpy_dtype == np.float32
    assert DataType.STRING.numpy_dtype == np.dtype(object)
    assert DataType.INT64.plain_width == 8
    assert DataType.BOOL.plain_width == 1


def test_time_units() -> None:
    assert TimeUnit.MILLISECOND.per_second == 1000
    assert TimeUnit.MICROSECOND.nanos == 1000
    assert TimeUnit.MILLISECOND.from_nanos(1_500_000) == 1
    assert TimeUnit.MILLISECOND.from_nanos(-1) == -1
    assert TimeUnit.SECOND.to_nanos(2) == 2_000_000_000
    assert TimeUnit.parse("µs") is TimeUnit.MICROSECOND
    assert TimeUnit.parse("Seconds") is TimeUnit.SECOND


def test_schema_building() -> None:
    schema = DeviceSchema.build("vm01", "ms", {"cpu": "float64"}, timezone="UTC")
    schema = schema.with_column("состояние", "string", {"unit": "-"})
    assert schema.column_names == ["cpu", "состояние"]
    assert schema.types == [DataType.FLOAT64, DataType.STRING]
    assert schema.column_index("состояние") == 1
    assert schema.column_index("missing") is None
    assert schema.columns[1].metadata == {"unit": "-"}
    schema.validate()
    tuple_form = DeviceSchema("x", "s", (("a", "bool"),))
    assert tuple_form.columns == (ColumnSchema("a", DataType.BOOL),)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (DeviceSchema("", "s", (("a", "bool"),)), "name must not be empty"),
        (DeviceSchema("x", "s", (("a", "bool"),), time_name=""), "time column name"),
        (DeviceSchema("x", "s"), "at least one value column"),
        (DeviceSchema("x", "s", (("a", "bool"), ("a", "bool"))), "duplicate"),
        (DeviceSchema("x", "s", (("time", "bool"),)), "duplicate"),
        (DeviceSchema("x", "s", (("", "bool"),)), "must not be empty"),
    ],
)
def test_validation(schema: DeviceSchema, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        schema.validate()


def test_too_many_columns() -> None:
    columns = tuple(ColumnSchema(f"c{i}", DataType.BOOL) for i in range(MAX_COLUMNS + 1))
    with pytest.raises(SchemaError, match="at most"):
        DeviceSchema("x", "s", columns).validate()


def test_same_layout() -> None:
    a = DeviceSchema.build("x", "ms", {"a": "int64"})
    assert a.same_layout(DeviceSchema.build("x", "ms", {"a": "int64"}, timezone="UTC"))
    assert not a.same_layout(DeviceSchema.build("x", "s", {"a": "int64"}))
    assert not a.same_layout(DeviceSchema.build("x", "ms", {"a": "int32"}))
